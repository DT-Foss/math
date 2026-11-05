# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.7.1",
#   "transformers==4.57.1",
#   "safetensors>=0.5.3",
#   "huggingface-hub>=0.35.0",
#   "numpy>=2.0,<3",
# ]
# ///
from __future__ import annotations

from collections import Counter, defaultdict
import gc
import json
import math
import os
import platform
import statistics
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT = "Describe a new computer architecture that reuses persistent mathematical state instead of recomputing every layer."
MAX_NEW_TOKENS = 48
BITS = 5
THREADS = 8
REPEATS = 3
PREDICTORS = ((1, 0), (2, -1), (0, 1), (0, 0), (3, -2), (-1, 2), (2, 0), (1, -1))


def median_ms(fn, repeats=REPEATS):
    fn()
    values=[]
    for _ in range(repeats):
        start=time.perf_counter_ns();fn();values.append((time.perf_counter_ns()-start)/1e6)
    return float(statistics.median(values))


def pctl(values,q):
    if not values:return 0.0
    xs=sorted(float(x) for x in values);pos=(len(xs)-1)*q
    lo=int(math.floor(pos));hi=int(math.ceil(pos))
    return xs[lo] if lo==hi else xs[lo]*(hi-pos)+xs[hi]*(pos-lo)


def quantize_per_coordinate(x:torch.Tensor,bits:int):
    qmax=(1<<(bits-1))-1
    scale=x.abs().amax(dim=0)/max(qmax,1)
    scale=scale.clamp_min(torch.finfo(torch.float32).tiny)
    codes=torch.round(x/scale).clamp(-qmax,qmax).to(torch.int16)
    return codes,codes.float()*scale,scale


def fit_predictors(codes:torch.Tensor,train_end:int):
    # Each coordinate selects the exact integer temporal predictor with the
    # lowest mismatch count on the calibration prefix.
    target=codes[2:train_end].to(torch.int32)
    prev=codes[1:train_end-1].to(torch.int32)
    prev2=codes[:train_end-2].to(torch.int32)
    scores=[]
    for a,b in PREDICTORS:
        scores.append((target!=(a*prev+b*prev2)).sum(dim=0))
    score=torch.stack(scores)
    assignment=torch.argmin(score,dim=0)
    return assignment


def predictor_geometry(codes:torch.Tensor,assignment:torch.Tensor,eval_start:int):
    hold=[];pred=[]
    by_pair=Counter()
    for g,pair in enumerate(PREDICTORS):
        by_pair[str(pair)]=int((assignment==g).sum())
    for t in range(eval_start,len(codes)):
        current=codes[t].to(torch.int32)
        previous=codes[t-1].to(torch.int32)
        previous2=codes[t-2].to(torch.int32)
        hold.append(int((current!=previous).sum()))
        predicted=torch.empty_like(current)
        for g,(a,b) in enumerate(PREDICTORS):
            mask=assignment==g
            predicted[mask]=a*previous[mask]+b*previous2[mask]
        pred.append(int((current!=predicted).sum()))
    width=codes.shape[1]
    return {
        "hold_event_fraction_median":statistics.median(hold)/width,
        "hold_event_fraction_p90":pctl([x/width for x in hold],.9),
        "predictor_event_fraction_median":statistics.median(pred)/width,
        "predictor_event_fraction_p90":pctl([x/width for x in pred],.9),
        "event_reduction":statistics.median(hold)/max(1,statistics.median(pred)),
        "predictor_usage":dict(by_pair),
    }


def replay_predictor(weight,bias,codes,scale,assignment,eval_start):
    W=weight.detach().cpu().float().contiguous()
    C=W.t().contiguous()
    bias=None if bias is None else bias.detach().cpu().float().contiguous()
    groups=[]
    for g,(a,b) in enumerate(PREDICTORS):
        idx=torch.nonzero(assignment==g,as_tuple=False).flatten()
        if idx.numel():groups.append((a,b,idx))

    decoded=codes.float()*scale
    def full_replay():
        y=None
        for t in range(eval_start,len(codes)):
            y=torch.mv(W,decoded[t])
            if bias is not None:y=y+bias
        return y

    # Partial outputs for t-2 and t-1. Bias is added once to the total output,
    # not to each predictor group.
    prev2=[];prev=[]
    for _,_,idx in groups:
        pages=C.index_select(0,idx)
        prev2.append(torch.matmul(decoded[eval_start-2].index_select(0,idx),pages))
        prev.append(torch.matmul(decoded[eval_start-1].index_select(0,idx),pages))

    def predictor_replay():
        y=None
        older=[x.clone() for x in prev2]
        last=[x.clone() for x in prev]
        for t in range(eval_start,len(codes)):
            current_codes=codes[t].to(torch.int32)
            previous_codes=codes[t-1].to(torch.int32)
            previous2_codes=codes[t-2].to(torch.int32)
            new=[]
            total=None
            for gi,(a,b,idx) in enumerate(groups):
                predicted_codes=a*previous_codes.index_select(0,idx)+b*previous2_codes.index_select(0,idx)
                actual_codes=current_codes.index_select(0,idx)
                residual_codes=actual_codes-predicted_codes
                event=torch.nonzero(residual_codes!=0,as_tuple=False).flatten()
                partial=a*last[gi]+b*older[gi]
                if event.numel():
                    global_idx=idx.index_select(0,event)
                    residual=(residual_codes.index_select(0,event).float()*scale.index_select(0,global_idx))
                    partial=partial+torch.matmul(residual,C.index_select(0,global_idx))
                new.append(partial)
                total=partial if total is None else total+partial
            if bias is not None:total=total+bias
            older,last=last,new
            y=total
        return y

    ref=full_replay();got=predictor_replay()
    full_ms=median_ms(full_replay);pred_ms=median_ms(predictor_replay)
    result={
        "input_width":int(W.shape[1]),"output_width":int(W.shape[0]),
        "predictor_groups":len(groups),"full_replay_ms":full_ms,
        "predictor_replay_ms":pred_ms,"speedup":full_ms/pred_ms,
        "max_abs_error":float((ref-got).abs().max()),
        "route":"predictor" if pred_ms<full_ms else "full",
    }
    del W,C,decoded,ref,got
    gc.collect()
    return result


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(THREADS)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True)
    enc=tok(text,return_tensors="pt");prompt_len=enc.input_ids.shape[1]
    generated=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)

    captures={};hooks=[];layers=model.model.layers
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    for i,l in enumerate(layers):
        hooks.append(l.self_attn.q_proj.register_forward_pre_hook(cap(f"L{i}.attn_norm")))
        hooks.append(l.self_attn.o_proj.register_forward_pre_hook(cap(f"L{i}.attn_out")))
        hooks.append(l.mlp.gate_proj.register_forward_pre_hook(cap(f"L{i}.mlp_norm")))
        hooks.append(l.mlp.down_proj.register_forward_pre_hook(cap(f"L{i}.mlp_hidden")))
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()

    start=max(0,prompt_len-1)
    source_rows=[];states={}
    for name,x0 in sorted(captures.items()):
        x=x0[start:]
        codes,decoded,scale=quantize_per_coordinate(x,BITS)
        train_end=max(4,2+len(codes)//2)
        eval_start=train_end
        assignment=fit_predictors(codes,train_end)
        geometry=predictor_geometry(codes,assignment,eval_start)
        geometry.update({
            "source":name,"layer":int(name.split('.')[0][1:]),"source_type":name.split('.')[1],
            "width":x.shape[1],"tokens":len(x),"train_end":train_end,
            "quantized_relative_l2_error":float(torch.linalg.vector_norm(decoded-x)/torch.linalg.vector_norm(x)),
        })
        source_rows.append(geometry);states[name]=(codes,scale,assignment,eval_start)

    selected_layers={0,len(layers)//2,len(layers)-1}
    module_rows=[]
    for i,l in enumerate(layers):
        if i not in selected_layers:continue
        specs=[
            ("q_proj",f"L{i}.attn_norm",l.self_attn.q_proj),
            ("o_proj",f"L{i}.attn_out",l.self_attn.o_proj),
            ("gate_proj",f"L{i}.mlp_norm",l.mlp.gate_proj),
            ("down_proj",f"L{i}.mlp_hidden",l.mlp.down_proj),
        ]
        for typ,source,module in specs:
            codes,scale,assignment,eval_start=states[source]
            row=replay_predictor(module.weight,module.bias,codes,scale,assignment,eval_start)
            row.update({"layer":i,"module_type":typ,"source":source})
            module_rows.append(row)

    by_type=defaultdict(list)
    for row in source_rows:by_type[row["source_type"]].append(row)
    geometry_summary={}
    for typ,rows in by_type.items():
        geometry_summary[typ]={
            "sources":len(rows),
            "hold_event_fraction_median":statistics.median(x["hold_event_fraction_median"] for x in rows),
            "predictor_event_fraction_median":statistics.median(x["predictor_event_fraction_median"] for x in rows),
            "event_reduction_median":statistics.median(x["event_reduction"] for x in rows),
            "quantized_relative_l2_error_median":statistics.median(x["quantized_relative_l2_error"] for x in rows),
        }
    full=sum(x["full_replay_ms"] for x in module_rows);pred=sum(x["predictor_replay_ms"] for x in module_rows)
    routed=sum(min(x["full_replay_ms"],x["predictor_replay_ms"]) for x in module_rows)
    result={
        "schema":"smollm2-temporal-predictor/v1","model":MODEL_ID,
        "parameters":sum(p.numel() for p in model.parameters()),"layers":len(layers),
        "prompt_tokens":prompt_len,"generated_tokens":generated.shape[1]-prompt_len,
        "bits":BITS,"predictors":[list(x) for x in PREDICTORS],
        "environment":{"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"torch":torch.__version__,"platform":platform.platform()},
        "geometry_by_source_type":geometry_summary,
        "selected_module_replay":{
            "modules":module_rows,"full_ms":full,"predictor_ms":pred,"routed_ms":routed,
            "predictor_speedup":full/pred,"routed_speedup":full/routed,
            "predictor_route_modules":sum(x["route"]=="predictor" for x in module_rows),
        },
        "source_examples":source_rows[:12],
        "decoded_tail":tok.decode(generated[0,-64:],skip_special_tokens=True),
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps({
        "geometry":geometry_summary,
        "selected_modules":{
            "predictor_speedup":full/pred,"routed_speedup":full/routed,
            "predictor_route_modules":sum(x["route"]=="predictor" for x in module_rows),
            "modules":len(module_rows),
        },
    },indent=2),flush=True)


if __name__=="__main__":main()
