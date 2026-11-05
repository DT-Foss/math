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

import gc,json,math,os,platform,statistics,time
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT="Explain why downstream consumer equivalence can compress neural state more strongly than activation reconstruction."
MAX_NEW_TOKENS=48
THREADS=8
REPEATS=2
CONFIGS=((16,16),(32,16))
PENALTIES=(0.0,.5,1.0,2.0,4.0,8.0,16.0)
THRESHOLDS=(.01,.03,.05,.08,.12)


def median_ms(fn,repeats=REPEATS):
    fn();xs=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();fn();xs.append((time.perf_counter_ns()-t)/1e6)
    return float(statistics.median(xs))


def pctl(xs,q):
    xs=sorted(float(x) for x in xs)
    if not xs:return 0.0
    p=(len(xs)-1)*q;lo=int(math.floor(p));hi=int(math.ceil(p))
    return xs[lo] if lo==hi else xs[lo]*(hi-p)+xs[hi]*(p-lo)


def block_responses(X,W,block):
    T,D=X.shape;O=W.shape[0];B=D//block
    xb=X.reshape(T,B,block);wb=W.reshape(O,B,block).transpose(1,0,2)
    return np.einsum("tbi,boi->tbo",xb,wb,optimize=True)


def fit_response_codebook(calibration,W,block,K,seed):
    R=block_responses(calibration,W,block);T,B,O=R.shape;rng=np.random.default_rng(seed)
    cent=np.empty((B,K,O))
    for b in range(B):cent[b]=R[rng.choice(T,K,replace=False),b]
    assign=np.full((T,B),-1,dtype=np.int64)
    for _ in range(10):
        dist=((R[:,:,None,:]-cent[None,:,:,:])**2).sum(-1);new=dist.argmin(-1)
        if np.array_equal(new,assign):break
        assign=new
        for b in range(B):
            for k in range(K):
                mask=assign[:,b]==k
                cent[b,k]=R[mask,b].mean(0) if mask.any() else R[rng.integers(T),b]
    dist=((R[:,:,None,:]-cent[None,:,:,:])**2).sum(-1);nearest=dist.min(-1)
    variance=np.mean((R-R.mean(0,keepdims=True))**2,axis=(0,2))*O
    scale=np.maximum(np.median(nearest,axis=0),variance*1e-6+np.finfo(np.float64).tiny)
    return cent,scale


def encode_eval(X,W,cent,scale,block,penalty,bias):
    R=block_responses(X,W,block);T,B,O=R.shape;codes=[];outputs=[];previous=None
    start=time.perf_counter_ns()
    for row in R:
        cost=((row[:,None,:]-cent)**2).sum(-1)
        if previous is not None and penalty:
            cost+=penalty*scale[:,None];cost[np.arange(B),previous]-=penalty*scale
        current=cost.argmin(1);codes.append(current);outputs.append(cent[np.arange(B),current].sum(0)+bias);previous=current
    assignment_ms=(time.perf_counter_ns()-start)/1e6
    codes=np.stack(codes);outputs=np.stack(outputs);true=X@W.T+bias
    changes=(codes[1:]!=codes[:-1]).sum(1)
    rel=np.linalg.norm(outputs-true)/max(np.linalg.norm(true),np.finfo(np.float64).tiny)
    return {
        "codes":codes,"outputs":outputs,"centroids":cent,"block_size":block,"K":cent.shape[1],"blocks":B,
        "penalty":penalty,"storage_ratio":cent.size/W.size,
        "changed_block_fraction_median":float(np.median(changes)/B),
        "changed_block_fraction_p90":float(np.percentile(changes,90)/B),
        "changed_blocks_median":float(np.median(changes)),
        "output_relative_l2_error":float(rel),"output_max_abs_error":float(np.max(np.abs(outputs-true))),
        "oracle_assignment_ms":assignment_ms,
    }


def choose(rows,threshold):
    valid=[x for x in rows if x["output_relative_l2_error"]<=threshold]
    if valid:return min(valid,key=lambda x:(x["changed_block_fraction_median"]*x["storage_ratio"],x["output_relative_l2_error"]))
    return min(rows,key=lambda x:x["output_relative_l2_error"])


def replay(state,true_outputs):
    cent=torch.from_numpy(state["centroids"]).float();codes=torch.from_numpy(state["codes"]).long();true=torch.from_numpy(true_outputs).float()
    B=cent.shape[0];arange=torch.arange(B);y0=cent[arange,codes[0]].sum(0)
    def full_reference():
        y=y0
        for row in true[1:]:y=row
        return y
    def transition():
        y=y0.clone();previous=codes[0]
        for current in codes[1:]:
            changed=torch.nonzero(current!=previous,as_tuple=False).flatten()
            if changed.numel():
                y.add_(cent[changed,current[changed]].sum(0));y.sub_(cent[changed,previous[changed]].sum(0))
            previous=current
        return y
    ref=cent[arange,codes[-1]].sum(0);got=transition();transition_ms=median_ms(transition)
    # Full baseline is supplied separately from dense W@x measurements.
    return {"transition_ms":transition_ms,"max_internal_error":float((got-ref).abs().max())}


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(THREADS)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tok=AutoTokenizer.from_pretrained(MODEL_ID);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt");prompt_len=enc.input_ids.shape[1]
    t=time.perf_counter_ns();generated=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id);gen_ms=(time.perf_counter_ns()-t)/1e6
    captures={};hooks=[]
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    for i,l in enumerate(model.model.layers):hooks.append(l.mlp.down_proj.register_forward_pre_hook(cap(f"L{i}.mlp_hidden")))
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()
    start=max(0,prompt_len-1);layer_frontiers={};chosen={x:{} for x in THRESHOLDS};dense_times={};true_eval={}
    for i,l in enumerate(model.model.layers):
        name=f"L{i}.mlp_hidden";X=captures[name][start:].numpy().astype(np.float64);split=max(16,len(X)//2);cal=X[:split];ev=X[split:]
        W=l.mlp.down_proj.weight.detach().cpu().numpy().astype(np.float64);bias=np.zeros(W.shape[0]) if l.mlp.down_proj.bias is None else l.mlp.down_proj.bias.detach().cpu().numpy().astype(np.float64)
        true=ev@W.T+bias;true_eval[i]=true
        Wt=torch.from_numpy(W).float();Xt=torch.from_numpy(ev).float();bt=torch.from_numpy(bias).float()
        def dense():
            y=bt
            for row in Xt:y=torch.mv(Wt,row)+bt
            return y
        dense_times[i]=median_ms(dense)
        frontier=[]
        for ci,(block,K) in enumerate(CONFIGS):
            cent,scale=fit_response_codebook(cal,W,block,K,100+i*10+ci)
            for penalty in PENALTIES:frontier.append(encode_eval(ev,W,cent,scale,block,penalty,bias))
        layer_frontiers[str(i)]=[{k:v for k,v in row.items() if k not in ("codes","outputs","centroids")} for row in frontier]
        for threshold in THRESHOLDS:chosen[threshold][i]=choose(frontier,threshold)
        del Wt,Xt;gc.collect()

    results={}
    for threshold in THRESHOLDS:
        rows=[]
        for i,state in chosen[threshold].items():
            timing=replay(state,true_eval[i]);dense_ms=dense_times[i]
            row={k:v for k,v in state.items() if k not in ("codes","outputs","centroids")}
            row.update(timing);row.update({
                "layer":i,"dense_down_ms":dense_ms,"speedup":dense_ms/timing["transition_ms"],
                "route":"quotient" if timing["transition_ms"]<dense_ms else "full",
            });rows.append(row)
        dense=sum(x["dense_down_ms"] for x in rows);transition=sum(x["transition_ms"] for x in rows);routed=sum(min(x["dense_down_ms"],x["transition_ms"]) for x in rows)
        results[str(threshold)]={
            "layers":len(rows),"dense_down_ms":dense,"quotient_transition_ms":transition,"routed_ms":routed,
            "quotient_speedup":dense/transition,"routed_speedup":dense/routed,
            "quotient_route_layers":sum(x["route"]=="quotient" for x in rows),
            "output_relative_l2_error_median":statistics.median(x["output_relative_l2_error"] for x in rows),
            "changed_block_fraction_median":statistics.median(x["changed_block_fraction_median"] for x in rows),
            "storage_ratio_median":statistics.median(x["storage_ratio"] for x in rows),
            "layers_detail":rows,
        }
    result={
        "schema":"smollm2-consumer-quotient-replay/v1","model":MODEL_ID,"parameters":sum(p.numel() for p in model.parameters()),"layers":len(model.model.layers),
        "prompt_tokens":prompt_len,"generated_tokens":generated.shape[1]-prompt_len,"generation_ms":gen_ms,"generation_tokens_per_second":(generated.shape[1]-prompt_len)/(gen_ms/1000),
        "environment":{"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"torch":torch.__version__,"platform":platform.platform()},
        "threshold_results":results,"layer_frontiers":layer_frontiers,
        "runtime_note":"Nearest-response assignment is an offline oracle in this experiment; the target runtime compiles gate/up block codes directly to quotient response codes.",
        "decoded_tail":tok.decode(generated[0,-64:],skip_special_tokens=True),
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps({"generation_tokens_per_second":result["generation_tokens_per_second"],"thresholds":{k:{x:y for x,y in v.items() if x!="layers_detail"} for k,v in results.items()}},indent=2),flush=True)


if __name__=="__main__":main()
