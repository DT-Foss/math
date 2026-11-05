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

from collections import defaultdict
import gc
import json
import math
import os
import platform
import statistics
import time

import torch
from transformers import AutoModelForCausalLM,AutoTokenizer


MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT="Explain how a stateful inference engine can avoid repeated dense computation while preserving model behavior."
MAX_NEW_TOKENS=int(os.environ.get("MAX_NEW_TOKENS","48"))
THREADS=int(os.environ.get("TORCH_THREADS","8"))
REPEATS=int(os.environ.get("REPEATS","2"))
CONFIGS=[
    (6,.5),(6,1.0),(6,2.0),(6,4.0),(6,8.0),
    (8,.5),(8,1.0),(8,2.0),(8,4.0),(8,8.0),(8,16.0),
]
THRESHOLDS=(.03,.05,.08,.12)


def median_ms(fn,repeats=REPEATS):
    fn();rows=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();fn();rows.append((time.perf_counter_ns()-t)/1e6)
    return float(statistics.median(rows))


def pctl(xs,q):
    if not xs:return 0.0
    values=sorted(float(x) for x in xs);p=(len(values)-1)*q
    lo=int(math.floor(p));hi=int(math.ceil(p))
    return values[lo] if lo==hi else values[lo]*(hi-p)+values[hi]*(p-lo)


def replay_hysteretic(x:torch.Tensor,bits:int,deadband:float,calibration_end:int):
    qmax=(1<<(bits-1))-1
    calibration=x[:calibration_end]
    scale=calibration.abs().amax(dim=0)/max(1,qmax)
    scale=scale.clamp_min(torch.finfo(torch.float32).tiny)
    code=torch.round(x[calibration_end-1]/scale).clamp(-qmax,qmax).to(torch.int16)
    decoded=code.float()*scale
    decoded_rows=[];events=[];counts=[]
    encode_ns=0
    for t in range(calibration_end,len(x)):
        start=time.perf_counter_ns()
        value=x[t]
        candidate_mask=(value-decoded).abs()>=deadband*scale
        candidates=torch.nonzero(candidate_mask,as_tuple=False).flatten()
        if candidates.numel():
            new_codes=torch.round(value.index_select(0,candidates)/scale.index_select(0,candidates)).clamp(-qmax,qmax).to(torch.int16)
            changed_mask=new_codes!=code.index_select(0,candidates)
            idx=candidates.index_select(0,torch.nonzero(changed_mask,as_tuple=False).flatten())
            if idx.numel():
                old=code.index_select(0,idx).clone()
                code[idx]=torch.round(value.index_select(0,idx)/scale.index_select(0,idx)).clamp(-qmax,qmax).to(torch.int16)
                delta=(code.index_select(0,idx).float()-old.float())*scale.index_select(0,idx)
                decoded[idx]+=delta
            else:
                delta=torch.empty(0)
        else:
            idx=torch.empty(0,dtype=torch.long);delta=torch.empty(0)
        encode_ns+=time.perf_counter_ns()-start
        events.append((idx,delta));counts.append(int(idx.numel()));decoded_rows.append(decoded.clone())
    decoded_trace=torch.stack(decoded_rows)
    target=x[calibration_end:]
    rel=float(torch.linalg.vector_norm(decoded_trace-target)/torch.linalg.vector_norm(target).clamp_min(1e-30))
    return {
        "bits":bits,"deadband":deadband,"scale":scale,"events":events,
        "decoded":decoded_trace,"initial":x.new_tensor((torch.round(x[calibration_end-1]/scale).clamp(-qmax,qmax))*scale),
        "event_fraction_median":statistics.median(counts)/x.shape[1],
        "event_fraction_p90":pctl([c/x.shape[1] for c in counts],.9),
        "event_count_median":statistics.median(counts),
        "relative_l2_error":rel,
        "max_abs_error":float((decoded_trace-target).abs().max()),
        "encode_ms":encode_ns/1e6,
    }


def choose(frontier,threshold):
    valid=[x for x in frontier if x["relative_l2_error"]<=threshold]
    if valid:return min(valid,key=lambda x:(x["event_fraction_median"],x["relative_l2_error"]))
    return min(frontier,key=lambda x:x["relative_l2_error"])


def replay_module(module,state):
    W=module.weight.detach().cpu().float().contiguous();bias=None if module.bias is None else module.bias.detach().cpu().float()
    C=W.t().contiguous();initial=state["initial"];decoded=state["decoded"];events=state["events"]
    y0=torch.mv(W,initial)
    if bias is not None:y0=y0+bias
    def full():
        y=y0
        for row in decoded:
            y=torch.mv(W,row)
            if bias is not None:y=y+bias
        return y
    def event():
        y=y0.clone()
        for idx,delta in events:
            if idx.numel():y.add_(torch.matmul(delta,C.index_select(0,idx)))
        return y
    ref=full();got=event();full_ms=median_ms(full);event_ms=median_ms(event)
    out={
        "input_width":int(W.shape[1]),"output_width":int(W.shape[0]),
        "full_ms":full_ms,"event_apply_ms":event_ms,"speedup":full_ms/event_ms,
        "max_abs_error":float((got-ref).abs().max()),
        "route":"event" if event_ms<full_ms else "full",
    }
    del W,C,ref,got;gc.collect();return out


def aggregate(rows):
    full=sum(x["full_ms"] for x in rows);event=sum(x["event_apply_ms"] for x in rows)
    routed=sum(min(x["full_ms"],x["event_apply_ms"]) for x in rows)
    return {
        "modules":len(rows),"full_ms":full,"event_apply_ms":event,"routed_ms":routed,
        "event_speedup":full/event,"routed_speedup":full/routed,
        "event_route_modules":sum(x["route"]=="event" for x in rows),
    }


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(THREADS)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tok=AutoTokenizer.from_pretrained(MODEL_ID)
    model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True)
    enc=tok(text,return_tensors="pt");prompt_len=enc.input_ids.shape[1]
    t=time.perf_counter_ns();generated=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id);gen_ms=(time.perf_counter_ns()-t)/1e6

    captures={};hooks=[];layers=model.model.layers
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    for i,l in enumerate(layers):
        hooks += [
            l.self_attn.q_proj.register_forward_pre_hook(cap(f"L{i}.attn_norm")),
            l.self_attn.o_proj.register_forward_pre_hook(cap(f"L{i}.attn_out")),
            l.mlp.gate_proj.register_forward_pre_hook(cap(f"L{i}.mlp_norm")),
            l.mlp.down_proj.register_forward_pre_hook(cap(f"L{i}.mlp_hidden")),
        ]
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()
    start=max(0,prompt_len-1)

    source_to_modules=defaultdict(list)
    for i,l in enumerate(layers):
        source_to_modules[f"L{i}.attn_norm"] += [("q_proj",l.self_attn.q_proj),("k_proj",l.self_attn.k_proj),("v_proj",l.self_attn.v_proj)]
        source_to_modules[f"L{i}.attn_out"] += [("o_proj",l.self_attn.o_proj)]
        source_to_modules[f"L{i}.mlp_norm"] += [("gate_proj",l.mlp.gate_proj),("up_proj",l.mlp.up_proj)]
        source_to_modules[f"L{i}.mlp_hidden"] += [("down_proj",l.mlp.down_proj)]

    frontiers={};chosen={threshold:{} for threshold in THRESHOLDS};source_summaries=[]
    for name in sorted(source_to_modules):
        x=captures[name][start:];calib=max(3,len(x)//2)
        rows=[replay_hysteretic(x,b,d,calib) for b,d in CONFIGS]
        frontiers[name]=rows
        for threshold in THRESHOLDS:chosen[threshold][name]=choose(rows,threshold)
        source_summaries.append({
            "source":name,"layer":int(name.split('.')[0][1:]),"source_type":name.split('.')[1],
            "width":x.shape[1],
            "frontier":[{k:v for k,v in row.items() if k not in ("scale","events","decoded","initial")} for row in rows],
        })

    threshold_results={}
    for threshold in THRESHOLDS:
        module_rows=[];by_type=defaultdict(list);source_geometry=defaultdict(list)
        for name,modules in source_to_modules.items():
            state=chosen[threshold][name]
            source_type=name.split('.')[1]
            source_geometry[source_type].append(state)
            for module_type,module in modules:
                row=replay_module(module,state)
                row.update({
                    "source":name,"layer":int(name.split('.')[0][1:]),"module_type":module_type,
                    "bits":state["bits"],"deadband":state["deadband"],
                    "event_fraction_median":state["event_fraction_median"],
                    "relative_l2_error":state["relative_l2_error"],
                })
                module_rows.append(row);by_type[module_type].append(row)
        threshold_results[str(threshold)]={
            "all":aggregate(module_rows),
            "by_module_type":{k:aggregate(v) for k,v in sorted(by_type.items())},
            "source_geometry":{
                k:{
                    "sources":len(v),
                    "event_fraction_median":statistics.median(x["event_fraction_median"] for x in v),
                    "relative_l2_error_median":statistics.median(x["relative_l2_error"] for x in v),
                    "bits_median":statistics.median(x["bits"] for x in v),
                    "deadband_median":statistics.median(x["deadband"] for x in v),
                }
                for k,v in sorted(source_geometry.items())
            },
            "fastest_modules":sorted(module_rows,key=lambda x:x["speedup"],reverse=True)[:20],
        }

    result={
        "schema":"smollm2-hysteretic-event-frontier/v1","model":MODEL_ID,
        "parameters":sum(p.numel() for p in model.parameters()),"layers":len(layers),
        "prompt_tokens":prompt_len,"generated_tokens":generated.shape[1]-prompt_len,
        "generation_ms":gen_ms,"generation_tokens_per_second":(generated.shape[1]-prompt_len)/(gen_ms/1000),
        "environment":{"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"torch":torch.__version__,"platform":platform.platform()},
        "configs":[list(x) for x in CONFIGS],"threshold_results":threshold_results,
        "source_frontier_examples":source_summaries[:12],
        "decoded_tail":tok.decode(generated[0,-64:],skip_special_tokens=True),
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps({
        "generation_tokens_per_second":result["generation_tokens_per_second"],
        "thresholds":{
            k:{"all":v["all"],"source_geometry":v["source_geometry"]}
            for k,v in threshold_results.items()
        },
    },indent=2),flush=True)


if __name__=="__main__":main()
