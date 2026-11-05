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
import json,os,statistics,time
import torch
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL="HuggingFaceTB/SmolLM2-135M-Instruct";TOKENS=160;RANKS=(8,16,32,64,96,128);THRESHOLDS=(.05,.1,.2,.3,.4,.5)
PROMPT="Write a detailed technical essay about persistent neural state, temporal innovations, and compressed response programs."

def main():
    torch.set_grad_enabled(False);torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(MODEL);model=AutoModelForCausalLM.from_pretrained(MODEL,dtype=torch.float32,attn_implementation="eager").eval()
    text=tok.apply_chat_template([{"role":"user","content":PROMPT}],tokenize=False,add_generation_prompt=True);enc=tok(text,return_tensors="pt");pl=enc.input_ids.shape[1]
    generated=model.generate(**enc,max_new_tokens=TOKENS,min_new_tokens=128,do_sample=False,use_cache=True,pad_token_id=tok.eos_token_id)
    captures={};hooks=[];modules=[]
    def cap(name):
        def hook(_m,args):captures[name]=args[0].detach().cpu().float().squeeze(0).contiguous()
        return hook
    for i,l in enumerate(model.model.layers):
        specs=[("q_proj","attn_norm",l.self_attn.q_proj),("k_proj","attn_norm",l.self_attn.k_proj),("v_proj","attn_norm",l.self_attn.v_proj),("o_proj","attn_out",l.self_attn.o_proj),("gate_proj","mlp_norm",l.mlp.gate_proj),("up_proj","mlp_norm",l.mlp.up_proj),("down_proj","mlp_hidden",l.mlp.down_proj)]
        seen=set()
        for typ,st,m in specs:
            source=f"L{i}.{st}"
            if source not in seen:hooks.append(m.register_forward_pre_hook(cap(source)));seen.add(source)
            modules.append((i,typ,source,m))
    model(generated,attention_mask=torch.ones_like(generated),use_cache=False)
    for h in hooks:h.remove()
    start=pl-1;rows=[]
    for layer,typ,source,m in modules:
        X=captures[source][start:];D=X[1:]-X[:-1];split=len(D)//2;train=D[:split];test=D[split:];W=m.weight.detach().cpu().float()
        response_train=train@W.T;response_test=test@W.T;_,_,Vh=torch.linalg.svd(response_train,full_matrices=False)
        exact_norm=torch.linalg.vector_norm(response_test,dim=1).clamp_min(1e-30)
        ranks=[]
        for rank in RANKS:
            if rank>Vh.shape[0]:continue
            U=Vh[:rank].T;approx=(response_test@U)@U.T
            errors=torch.linalg.vector_norm(response_test-approx,dim=1)/exact_norm
            ranks.append({"rank":rank,"median":float(torch.median(errors)),"p90":float(torch.quantile(errors,.9)),"max":float(errors.max()),"theoretical_speedup":(W.shape[0]*W.shape[1])/(rank*(W.shape[0]+W.shape[1]))})
        rows.append({"layer":layer,"module_type":typ,"source":source,"ranks":ranks})
    summary={}
    for threshold in THRESHOLDS:
        selected=[]
        for row in rows:
            valid=[x for x in row["ranks"] if x["p90"]<=threshold]
            if valid:selected.append((row,max(valid,key=lambda x:x["theoretical_speedup"])))
        by=defaultdict(list)
        for row,rank in selected:by[row["module_type"]].append(rank)
        summary[str(threshold)]={
            "modules":len(rows),"basis_route_modules":len(selected),
            "rank_median":statistics.median(r["rank"] for _,r in selected) if selected else None,
            "theoretical_speedup_median":statistics.median(r["theoretical_speedup"] for _,r in selected) if selected else None,
            "by_module_type":{k:{"modules":len(v),"rank_median":statistics.median(x["rank"] for x in v),"speedup_median":statistics.median(x["theoretical_speedup"] for x in v)} for k,v in sorted(by.items())},
        }
    result={"schema":"smollm2-response-route-census/v1","model":MODEL,"parameters":sum(p.numel() for p in model.parameters()),"generated_tokens":generated.shape[1]-pl,"summary":summary,"modules":rows}
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    print(json.dumps(summary,indent=2),flush=True)
if __name__=="__main__":main()
