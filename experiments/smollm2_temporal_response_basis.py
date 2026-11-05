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
import gc,json,math,os,platform,statistics,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT="Write a detailed essay about temporal differential neural programs and persistent state computation."
NEW_TOKENS=int(os.environ.get("NEW_TOKENS","160"))
THREADS=int(os.environ.get("TORCH_THREADS","8"))
RANKS=(4,8,16,32,48,64,96,128)
ERROR_THRESHOLDS=(.03,.05,.10,.20,.30)
REPEATS=3


def median_ms(fn,repeats=REPEATS):
    fn();rows=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();fn();rows.append((time.perf_counter_ns()-t)/1e6)
    return float(statistics.median(rows))


def fit_response_basis(W,train_delta,rank):
    response=train_delta@W.T
    _,s,Vh=torch.linalg.svd(response,full_matrices=False)
    r=min(rank,Vh.shape[0]);U=Vh[:r].T.contiguous();A=(W.T@U).contiguous()
    return U,A,s[:r]


def evaluate_basis(W,D,U,A):
    exact=D@W.T;approx=(D@A)@U.T;residual=exact-approx
    exact_energy=float(exact.square().sum());residual_energy=float(residual.square().sum())
    capture=1.0 if exact_energy==0 else 1.0-residual_energy/exact_energy
    per=torch.linalg.vector_norm(residual,dim=1)/torch.linalg.vector_norm(exact,dim=1).clamp_min(1e-30)
    return {
        "capture":capture,
        "relative_error_median":float(torch.median(per)),
        "relative_error_p90":float(torch.quantile(per,.9)),
        "relative_error_max":float(per.max()),
        "exact":exact,"approx":approx,
    }


def replay_times(W,input_states,U,A,split):
    x0=input_states[split-1];eval_x=input_states[split:];deltas=eval_x-input_states[split-1:-1]
    y0=W@x0
    def full_absolute():
        y=y0
        for x in eval_x:y=W@x
        return y
    def full_delta():
        y=y0.clone()
        for d in deltas:y=y+W@d
        return y
    def lowrank():
        y=y0.clone()
        for d in deltas:y=y+U@(A.T@d)
        return y
    exact=full_delta();approx=lowrank()
    return {
        "full_absolute_ms":median_ms(full_absolute),
        "full_delta_ms":median_ms(full_delta),
        "basis_ms":median_ms(lowrank),
        "basis_speedup_vs_absolute":median_ms(full_absolute)/median_ms(lowrank),
        "basis_speedup_vs_full_delta":median_ms(full_delta)/median_ms(lowrank),
        "final_relative_error":float(torch.linalg.vector_norm(exact-approx)/torch.linalg.vector_norm(exact).clamp_min(1e-30)),
    }


def summarize(rows,key):
    values=[x[key] for x in rows]
    return {"median":statistics.median(values),"p90":sorted(values)[min(len(values)-1,math.ceil(.9*len(values))-1)],"min":min(values),"max":max(values)}


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(THREADS)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tok=AutoTokenizer.from_pretrained(MODEL_ID);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt");prompt_len=enc.input_ids.shape[1]
    t=time.perf_counter_ns();generated=model.generate(**enc,max_new_tokens=NEW_TOKENS,min_new_tokens=min(128,NEW_TOKENS),do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id);generation_ms=(time.perf_counter_ns()-t)/1e6
    captures={};hooks=[]
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    modules=[]
    for i,l in enumerate(model.model.layers):
        specs=[("q_proj","attn_norm",l.self_attn.q_proj),("k_proj","attn_norm",l.self_attn.k_proj),("v_proj","attn_norm",l.self_attn.v_proj),("o_proj","attn_out",l.self_attn.o_proj),("gate_proj","mlp_norm",l.mlp.gate_proj),("up_proj","mlp_norm",l.mlp.up_proj),("down_proj","mlp_hidden",l.mlp.down_proj)]
        seen=set()
        for typ,source_type,module in specs:
            source=f"L{i}.{source_type}"
            if source not in seen:
                hooks.append(module.register_forward_pre_hook(cap(source)));seen.add(source)
            modules.append((i,typ,source,module))
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()
    start=max(0,prompt_len-1);rows=[]
    for mi,(layer,typ,source,module) in enumerate(modules):
        X=captures[source][start:].contiguous();D=X[1:]-X[:-1];split=max(8,len(D)//2);train=D[:split];test=D[split:]
        W=module.weight.detach().cpu().float().contiguous()
        rank_rows=[]
        for rank in RANKS:
            if rank>min(train.shape[0],W.shape[0]):continue
            U,A,s=fit_response_basis(W,train,rank);ev=evaluate_basis(W,test,U,A)
            theoretical=(W.shape[0]*W.shape[1])/(rank*(W.shape[0]+W.shape[1]))
            row={
                "rank":rank,"capture":ev["capture"],"relative_error_median":ev["relative_error_median"],
                "relative_error_p90":ev["relative_error_p90"],"relative_error_max":ev["relative_error_max"],
                "storage_ratio":rank*(W.shape[0]+W.shape[1])/(W.shape[0]*W.shape[1]),
                "theoretical_multiply_speedup":theoretical,
            }
            if rank in (16,32,64):row.update(replay_times(W,X,U,A,split+1))
            rank_rows.append(row)
        rows.append({
            "layer":layer,"module_type":typ,"source":source,"input_width":W.shape[1],"output_width":W.shape[0],
            "train_transitions":split,"heldout_transitions":len(test),"ranks":rank_rows,
        })
        del W,X,D,train,test;gc.collect()

    by_type=defaultdict(list)
    for row in rows:by_type[row["module_type"]].append(row)
    summary={}
    for typ,group in sorted(by_type.items()):
        rank_summary={}
        for rank in RANKS:
            selected=[next((r for r in x["ranks"] if r["rank"]==rank),None) for x in group]
            selected=[x for x in selected if x is not None]
            if not selected:continue
            rank_summary[str(rank)]={
                "modules":len(selected),
                "capture":summarize(selected,"capture"),
                "relative_error_median":summarize(selected,"relative_error_median"),
                "theoretical_multiply_speedup":statistics.median(x["theoretical_multiply_speedup"] for x in selected),
                "measured_basis_speedup_vs_absolute_median":statistics.median(x.get("basis_speedup_vs_absolute",0) for x in selected) if rank in (16,32,64) else None,
            }
        summary[typ]=rank_summary

    route_summary={}
    for threshold in ERROR_THRESHOLDS:
        chosen=[]
        for module in rows:
            valid=[r for r in module["ranks"] if r["relative_error_p90"]<=threshold]
            route=max(valid,key=lambda r:r["theoretical_multiply_speedup"]) if valid else None
            chosen.append((module,route))
        applicable=[x for _,x in chosen if x is not None]
        route_summary[str(threshold)]={
            "modules":len(rows),"basis_route_modules":len(applicable),
            "rank_median":statistics.median(x["rank"] for x in applicable) if applicable else None,
            "capture_median":statistics.median(x["capture"] for x in applicable) if applicable else None,
            "theoretical_multiply_speedup_median":statistics.median(x["theoretical_multiply_speedup"] for x in applicable) if applicable else None,
        }
    result={
        "schema":"smollm2-temporal-response-basis/v1","model":MODEL_ID,"parameters":sum(p.numel() for p in model.parameters()),"layers":len(model.model.layers),
        "prompt_tokens":prompt_len,"generated_tokens":generated.shape[1]-prompt_len,"generation_ms":generation_ms,"generation_tokens_per_second":(generated.shape[1]-prompt_len)/(generation_ms/1000),
        "environment":{"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"torch":torch.__version__,"platform":platform.platform()},
        "by_module_type":summary,"route_summary":route_summary,"modules":rows,
        "decoded_tail":tok.decode(generated[0,-96:],skip_special_tokens=True),
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps({"generation_tokens_per_second":result["generation_tokens_per_second"],"route_summary":route_summary,"by_module_type":summary},indent=2),flush=True)


if __name__=="__main__":main()
