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
import json,math,os,platform,statistics,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
PROMPT="Write a detailed technical essay about persistent state-transition computation, incremental algebra, compressed model layouts, and future computer architectures. Continue for many paragraphs."
NEW_TOKENS=int(os.environ.get("NEW_TOKENS","160"))
THREADS=int(os.environ.get("TORCH_THREADS","8"))
RANKS=(4,8,16,32,64,96,128)


def energy_rank(D,fraction):
    if D.numel()==0:return 0
    gram=D@D.T
    eigen=torch.linalg.eigvalsh(gram.double()).clamp_min(0).flip(0)
    total=eigen.sum()
    if float(total)==0:return 0
    return int(torch.searchsorted(torch.cumsum(eigen,0),fraction*total).item()+1)


def analyze(x):
    D=(x[1:]-x[:-1]).float();T,d=D.shape;split=max(2,T//2);train=D[:split];test=D[split:]
    U,S,Vh=torch.linalg.svd(train,full_matrices=False)
    total=float(test.square().sum())
    capture={}
    for r0 in RANKS:
        r=min(r0,Vh.shape[0])
        projected=test@Vh[:r].T
        capture[str(r0)]=0.0 if total==0 else float(projected.square().sum())/total
    train_rank99=energy_rank(train,.99);full_rank99=energy_rank(D,.99);full_rank999=energy_rank(D,.999)
    # Online novelty: residual energy after projecting each new delta onto all previous deltas.
    basis=[];novel=[]
    for row in D:
        v=row.double()
        if basis:
            B=torch.stack(basis)
            coeff=B@v
            residual=v-coeff@B
        else:residual=v
        denom=float(torch.dot(v,v));ratio=0.0 if denom==0 else float(torch.dot(residual,residual))/denom
        novel.append(ratio)
        norm=torch.linalg.vector_norm(residual)
        if float(norm)>1e-10:basis.append(residual/norm)
    return {
        "width":d,"transitions":T,
        "rank_90":energy_rank(D,.90),"rank_99":full_rank99,"rank_999":full_rank999,
        "rank99_fraction":full_rank99/d,"rank999_fraction":full_rank999/d,
        "train_rank_99":train_rank99,"heldout_energy_capture":capture,
        "online_exact_basis_size":len(basis),
        "online_novel_energy_median":statistics.median(novel),
        "online_novel_energy_last_quarter_median":statistics.median(novel[max(0,3*len(novel)//4):]),
    }


def summarize(rows):
    out={"sources":len(rows)}
    for key in ("width","transitions","rank_90","rank_99","rank_999","rank99_fraction","rank999_fraction","train_rank_99","online_exact_basis_size","online_novel_energy_median","online_novel_energy_last_quarter_median"):
        out[key+"_median"]=statistics.median(x[key] for x in rows)
        out[key+"_p90"]=sorted(x[key] for x in rows)[min(len(rows)-1,math.ceil(.9*len(rows))-1)]
    out["heldout_energy_capture_median"]={r:statistics.median(x["heldout_energy_capture"][r] for x in rows) for r in map(str,RANKS)}
    return out


def main():
    torch.set_grad_enabled(False);torch.set_num_threads(THREADS)
    try:torch.set_num_interop_threads(1)
    except RuntimeError:pass
    tok=AutoTokenizer.from_pretrained(MODEL_ID);model=AutoModelForCausalLM.from_pretrained(MODEL_ID,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt");prompt_len=enc.input_ids.shape[1]
    t=time.perf_counter_ns();generated=model.generate(**enc,max_new_tokens=NEW_TOKENS,min_new_tokens=min(NEW_TOKENS,128),do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id);gen_ms=(time.perf_counter_ns()-t)/1e6
    captures={};hooks=[]
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    for i,l in enumerate(model.model.layers):hooks += [l.self_attn.q_proj.register_forward_pre_hook(cap(f"L{i}.attn_norm")),l.self_attn.o_proj.register_forward_pre_hook(cap(f"L{i}.attn_out")),l.mlp.gate_proj.register_forward_pre_hook(cap(f"L{i}.mlp_norm")),l.mlp.down_proj.register_forward_pre_hook(cap(f"L{i}.mlp_hidden"))]
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()
    start=max(0,prompt_len-1);rows=[];grouped=defaultdict(list)
    for name,x in sorted(captures.items()):
        row=analyze(x[start:]);row.update({"source":name,"layer":int(name.split('.')[0][1:]),"source_type":name.split('.')[1]});rows.append(row);grouped[row["source_type"]].append(row)
    result={
        "schema":"smollm2-temporal-subspace-census/v1","model":MODEL_ID,"parameters":sum(p.numel() for p in model.parameters()),"layers":len(model.model.layers),
        "prompt_tokens":prompt_len,"generated_tokens":generated.shape[1]-prompt_len,"generation_ms":gen_ms,"generation_tokens_per_second":(generated.shape[1]-prompt_len)/(gen_ms/1000),
        "environment":{"cpu_count":os.cpu_count(),"torch_threads":torch.get_num_threads(),"torch":torch.__version__,"platform":platform.platform()},
        "by_source_type":{k:summarize(v) for k,v in sorted(grouped.items())},"sources":rows,
        "decoded_tail":tok.decode(generated[0,-96:],skip_special_tokens=True),
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps({"generated_tokens_per_second":result["generation_tokens_per_second"],"by_source_type":result["by_source_type"]},indent=2),flush=True)


if __name__=="__main__":main()
