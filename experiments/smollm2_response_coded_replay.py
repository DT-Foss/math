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
import numpy as np
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT="Explain why a response-coded neural model can execute state transitions instead of repeating dense matrix multiplication."
MAX_NEW_TOKENS=48
THREADS=8
REPEATS=1
CODEBOOKS=((8,8),(16,16),(32,16))
PENALTIES=(0.0,.5,1.0,2.0,4.0,8.0,16.0)
THRESHOLDS=(.05,.08,.12,.20)


def ms(fn,repeats=REPEATS):
    fn();rows=[]
    for _ in range(repeats):
        t=time.perf_counter_ns();fn();rows.append((time.perf_counter_ns()-t)/1e6)
    return float(statistics.median(rows))


def pctl(xs,q):
    xs=sorted(float(x) for x in xs)
    if not xs:return 0.0
    p=(len(xs)-1)*q;lo=int(math.floor(p));hi=int(math.ceil(p))
    return xs[lo] if lo==hi else xs[lo]*(hi-p)+xs[hi]*(p-lo)


def fit_codebook(calibration,block_size,K,seed):
    X=calibration.detach().cpu().numpy().astype(np.float64)
    T,D=X.shape;B=D//block_size;blocks=X.reshape(T,B,block_size)
    rng=np.random.default_rng(seed);cent=np.empty((B,K,block_size))
    for b in range(B):cent[b]=blocks[rng.choice(T,K,replace=False),b]
    assign=np.full((T,B),-1,dtype=np.int64)
    for _ in range(15):
        dist=((blocks[:,:,None,:]-cent[None,:,:,:])**2).sum(-1)
        new=dist.argmin(-1)
        if np.array_equal(new,assign):break
        assign=new
        for b in range(B):
            for k in range(K):
                mask=assign[:,b]==k
                cent[b,k]=blocks[mask,b].mean(0) if mask.any() else blocks[rng.integers(T),b]
    dist=((blocks[:,:,None,:]-cent[None,:,:,:])**2).sum(-1)
    nearest=dist.min(-1);variance=np.mean((blocks-blocks.mean(0,keepdims=True))**2,axis=(0,2))*block_size
    scale=np.maximum(np.median(nearest,axis=0),variance*1e-6+np.finfo(np.float64).tiny)
    return cent,scale


def encode_eval(eval_trace,cent,scale,penalty):
    X=eval_trace.detach().cpu().numpy().astype(np.float64);T,D=X.shape;B,K,b=cent.shape
    blocks=X.reshape(T,B,b);codes=[];decoded=[];previous=None
    start=time.perf_counter_ns()
    for row in blocks:
        cost=((row[:,None,:]-cent)**2).sum(-1)
        if previous is not None and penalty:
            cost+=penalty*scale[:,None];cost[np.arange(B),previous]-=penalty*scale
        current=cost.argmin(1);codes.append(current);decoded.append(cent[np.arange(B),current].reshape(D));previous=current
    encode_ms=(time.perf_counter_ns()-start)/1e6
    codes=np.stack(codes);decoded=np.stack(decoded)
    changes=(codes[1:]!=codes[:-1]).sum(1)
    rel=np.linalg.norm(decoded-X)/max(np.linalg.norm(X),np.finfo(np.float64).tiny)
    return {
        "codes":codes,"decoded":decoded,"centroids":cent,
        "block_size":b,"K":K,"blocks":B,"penalty":penalty,
        "response_storage_ratio":K/b,
        "changed_block_fraction_median":float(np.median(changes)/B),
        "changed_block_fraction_p90":float(np.percentile(changes,90)/B),
        "changed_blocks_median":float(np.median(changes)),
        "relative_l2_error":float(rel),"max_abs_error":float(np.max(np.abs(decoded-X))),
        "encode_ms":encode_ms,
    }


def choose(rows,threshold):
    valid=[x for x in rows if x["relative_l2_error"]<=threshold]
    if valid:return min(valid,key=lambda x:(x["changed_block_fraction_median"]*x["response_storage_ratio"],x["relative_l2_error"]))
    return min(rows,key=lambda x:x["relative_l2_error"])


def replay_module(module,state):
    W=module.weight.detach().cpu().float().contiguous();bias=None if module.bias is None else module.bias.detach().cpu().float()
    decoded=torch.from_numpy(state["decoded"]).float();codes=torch.from_numpy(state["codes"]).long()
    cent=torch.from_numpy(state["centroids"]).float();B,K,b=cent.shape;out=W.shape[0]
    Wb=W.reshape(out,B,b).permute(1,0,2).contiguous()
    t=time.perf_counter_ns();responses=torch.einsum("bki,boi->bko",cent,Wb);compile_ms=(time.perf_counter_ns()-t)/1e6
    arange=torch.arange(B);y0=responses[arange,codes[0]].sum(0)
    if bias is not None:y0=y0+bias
    def full():
        y=y0
        for row in decoded[1:]:
            y=torch.mv(W,row)
            if bias is not None:y=y+bias
        return y
    def response():
        y=y0.clone();previous=codes[0]
        for current in codes[1:]:
            changed=torch.nonzero(current!=previous,as_tuple=False).flatten()
            if changed.numel():
                y.add_(responses[changed,current[changed]].sum(0))
                y.sub_(responses[changed,previous[changed]].sum(0))
            previous=current
        return y
    ref=full();got=response();full_ms=ms(full);response_ms=ms(response)
    row={
        "input_width":int(W.shape[1]),"output_width":int(W.shape[0]),
        "full_ms":full_ms,"response_ms":response_ms,"speedup":full_ms/response_ms,
        "compile_ms":compile_ms,"max_abs_error":float((ref-got).abs().max()),
        "route":"response" if response_ms<full_ms else "full",
    }
    del W,Wb,responses,decoded,codes,ref,got;gc.collect();return row


def aggregate(rows):
    full=sum(x["full_ms"] for x in rows);response=sum(x["response_ms"] for x in rows);routed=sum(min(x["full_ms"],x["response_ms"]) for x in rows)
    return {
        "modules":len(rows),"full_ms":full,"response_ms":response,"routed_ms":routed,
        "response_speedup":full/response,"routed_speedup":full/routed,
        "response_route_modules":sum(x["route"]=="response" for x in rows),
        "compile_ms":sum(x["compile_ms"] for x in rows),
    }


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(THREADS)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tok=AutoTokenizer.from_pretrained(MODEL_ID);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt");prompt_len=enc.input_ids.shape[1]
    t=time.perf_counter_ns();generated=model.generate(**enc,max_new_tokens=MAX_NEW_TOKENS,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id);gen_ms=(time.perf_counter_ns()-t)/1e6
    captures={};hooks=[];layers=model.model.layers
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    for i,l in enumerate(layers):
        hooks += [l.self_attn.q_proj.register_forward_pre_hook(cap(f"L{i}.attn_norm")),l.self_attn.o_proj.register_forward_pre_hook(cap(f"L{i}.attn_out")),l.mlp.gate_proj.register_forward_pre_hook(cap(f"L{i}.mlp_norm")),l.mlp.down_proj.register_forward_pre_hook(cap(f"L{i}.mlp_hidden"))]
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()
    start=max(0,prompt_len-1)
    source_to_modules=defaultdict(list)
    for i,l in enumerate(layers):
        source_to_modules[f"L{i}.attn_norm"] += [("q_proj",l.self_attn.q_proj),("k_proj",l.self_attn.k_proj),("v_proj",l.self_attn.v_proj)]
        source_to_modules[f"L{i}.attn_out"] += [("o_proj",l.self_attn.o_proj)]
        source_to_modules[f"L{i}.mlp_norm"] += [("gate_proj",l.mlp.gate_proj),("up_proj",l.mlp.up_proj)]
        source_to_modules[f"L{i}.mlp_hidden"] += [("down_proj",l.mlp.down_proj)]

    chosen={threshold:{} for threshold in THRESHOLDS};examples=[]
    for si,name in enumerate(sorted(source_to_modules)):
        x=captures[name][start:];split=max(16,len(x)//2);cal=x[:split];evaluation=x[split:]
        frontier=[]
        for bi,(block,K) in enumerate(CODEBOOKS):
            cent,scale=fit_codebook(cal,block,K,1000+si*10+bi)
            for penalty in PENALTIES:frontier.append(encode_eval(evaluation,cent,scale,penalty))
        for threshold in THRESHOLDS:chosen[threshold][name]=choose(frontier,threshold)
        if len(examples)<12:
            examples.append({
                "source":name,"source_type":name.split('.')[1],"layer":int(name.split('.')[0][1:]),
                "frontier":[{k:v for k,v in row.items() if k not in ("codes","decoded","centroids")} for row in frontier],
            })

    results={}
    for threshold in THRESHOLDS:
        rows=[];by_type=defaultdict(list);geometry=defaultdict(list)
        for name,modules in source_to_modules.items():
            state=chosen[threshold][name];geometry[name.split('.')[1]].append(state)
            for typ,module in modules:
                row=replay_module(module,state);row.update({
                    "source":name,"layer":int(name.split('.')[0][1:]),"module_type":typ,
                    "block_size":state["block_size"],"K":state["K"],"penalty":state["penalty"],
                    "changed_block_fraction_median":state["changed_block_fraction_median"],
                    "relative_l2_error":state["relative_l2_error"],"response_storage_ratio":state["response_storage_ratio"],
                });rows.append(row);by_type[typ].append(row)
        results[str(threshold)]={
            "all":aggregate(rows),"by_module_type":{k:aggregate(v) for k,v in sorted(by_type.items())},
            "source_geometry":{
                k:{
                    "sources":len(v),"changed_block_fraction_median":statistics.median(x["changed_block_fraction_median"] for x in v),
                    "relative_l2_error_median":statistics.median(x["relative_l2_error"] for x in v),
                    "block_size_median":statistics.median(x["block_size"] for x in v),"codes_median":statistics.median(x["K"] for x in v),
                    "response_storage_ratio_median":statistics.median(x["response_storage_ratio"] for x in v),
                } for k,v in sorted(geometry.items())
            },
            "fastest_modules":sorted(rows,key=lambda x:x["speedup"],reverse=True)[:20],
        }
    result={
        "schema":"smollm2-response-coded-replay/v1","model":MODEL_ID,"parameters":sum(p.numel() for p in model.parameters()),"layers":len(layers),
        "prompt_tokens":prompt_len,"generated_tokens":generated.shape[1]-prompt_len,"generation_ms":gen_ms,"generation_tokens_per_second":(generated.shape[1]-prompt_len)/(gen_ms/1000),
        "environment":{"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"torch":torch.__version__,"platform":platform.platform()},
        "codebooks":[list(x) for x in CODEBOOKS],"penalties":list(PENALTIES),"threshold_results":results,"source_frontier_examples":examples,
        "decoded_tail":tok.decode(generated[0,-64:],skip_special_tokens=True),
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps({"generation_tokens_per_second":result["generation_tokens_per_second"],"thresholds":{k:{"all":v["all"],"source_geometry":v["source_geometry"]} for k,v in results.items()}},indent=2),flush=True)


if __name__=="__main__":main()
