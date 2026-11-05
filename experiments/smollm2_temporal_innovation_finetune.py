# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "torch==2.7.1",
#   "transformers==4.57.1",
#   "datasets>=3.6.0",
#   "safetensors>=0.5.3",
#   "huggingface-hub>=0.35.0",
#   "numpy>=2.0,<3",
# ]
# ///
from __future__ import annotations

import json,math,os,platform,random,time
import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from torch import nn
from transformers import AutoModelForCausalLM,AutoTokenizer

MODEL_ID="HuggingFaceTB/SmolLM2-135M-Instruct"
SEED=20261105
SEQ=int(os.environ.get("SEQ","128"))
BATCH=int(os.environ.get("BATCH","2"))
STEPS=int(os.environ.get("STEPS","30"))
TEMPORAL_RANK=int(os.environ.get("TEMPORAL_RANK","32"))
TEMPORAL_WEIGHT=float(os.environ.get("TEMPORAL_WEIGHT","0.20"))
HIDDEN_WEIGHT=float(os.environ.get("HIDDEN_WEIGHT","0.05"))
LR=float(os.environ.get("LR","1e-5"))
BASIS_LR=float(os.environ.get("BASIS_LR","5e-4"))
SELECTED=(0,4,8,12,16,20,24,29)


def set_seed(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def make_chunks(tokenizer):
    ds=load_dataset("wikitext","wikitext-2-raw-v1",split="train")
    texts=[x for x in ds["text"] if x.strip()]
    joined="\n\n".join(texts[:1800])
    ids=tokenizer(joined,return_tensors="pt",add_special_tokens=False).input_ids[0]
    total=(len(ids)//SEQ)*SEQ
    chunks=ids[:total].reshape(-1,SEQ)
    if len(chunks)<80:raise RuntimeError("insufficient token chunks")
    return chunks[:64],chunks[64:80]


def batch_iterator(chunks,batch,steps,seed):
    g=torch.Generator().manual_seed(seed)
    order=torch.randint(0,len(chunks),(steps,batch),generator=g)
    for idx in order:yield chunks.index_select(0,idx)


def delta_matrix(hidden):
    return (hidden[:,1:]-hidden[:,:-1]).reshape(-1,hidden.shape[-1]).float()


def initialize_bases(teacher,chunks,device,rank):
    collected={layer:[] for layer in SELECTED}
    with torch.no_grad():
        for batch in batch_iterator(chunks,2,4,SEED+1):
            out=teacher(batch.to(device),output_hidden_states=True,use_cache=False)
            for layer in SELECTED:
                collected[layer].append(delta_matrix(out.hidden_states[layer+1]).cpu())
    bases={}
    for layer,rows in collected.items():
        D=torch.cat(rows).double()
        _,_,Vh=torch.linalg.svd(D,full_matrices=False)
        bases[str(layer)]=nn.Parameter(Vh[:rank].T.float().to(device))
    return nn.ParameterDict(bases)


def orthobasis(parameter):
    return torch.linalg.qr(parameter.float(),mode="reduced").Q


def temporal_loss(hidden_states,bases):
    losses=[];captures=[]
    for layer in SELECTED:
        D=delta_matrix(hidden_states[layer+1])
        Q=orthobasis(bases[str(layer)])
        coeff=D@Q;recon=coeff@Q.T;residual=D-recon
        energy=D.square().mean().clamp_min(1e-12)
        ratio=residual.square().mean()/energy
        losses.append(ratio);captures.append(1.0-ratio.detach())
    return torch.stack(losses).mean(),torch.stack(captures).mean()


def normalized_mse(a,b):
    denom=b.float().square().mean().clamp_min(1e-8)
    return (a.float()-b.float()).square().mean()/denom


def eval_metrics(model,teacher,bases,chunks,device):
    model.eval();teacher.eval()
    logits_mse=[];top1=[];student_ce=[];teacher_ce=[];hidden_error=[]
    layer_rows={layer:{"teacher":[],"student":[]} for layer in SELECTED}
    with torch.no_grad():
        for batch in batch_iterator(chunks,2,6,SEED+99):
            x=batch.to(device)
            t=teacher(x,output_hidden_states=True,use_cache=False)
            s=model(x,output_hidden_states=True,use_cache=False)
            logits_mse.append(float(normalized_mse(s.logits,t.logits)))
            top1.append(float((s.logits.argmax(-1)==t.logits.argmax(-1)).float().mean()))
            labels=x[:,1:]
            student_ce.append(float(F.cross_entropy(s.logits[:,:-1].float().reshape(-1,s.logits.shape[-1]),labels.reshape(-1))))
            teacher_ce.append(float(F.cross_entropy(t.logits[:,:-1].float().reshape(-1,t.logits.shape[-1]),labels.reshape(-1))))
            for layer in SELECTED:
                hidden_error.append(float(normalized_mse(s.hidden_states[layer+1],t.hidden_states[layer+1])))
                layer_rows[layer]["teacher"].append(delta_matrix(t.hidden_states[layer+1]).cpu())
                layer_rows[layer]["student"].append(delta_matrix(s.hidden_states[layer+1]).cpu())
    layers={}
    for layer in SELECTED:
        Q=orthobasis(bases[str(layer)]).detach().cpu().double()
        row={}
        for kind in ("teacher","student"):
            D=torch.cat(layer_rows[layer][kind]).double()
            projected=D@Q;capture=float(projected.square().sum()/D.square().sum().clamp_min(1e-30))
            singular=torch.linalg.svdvals(D)
            energy=singular.square();cum=torch.cumsum(energy,0)
            rank90=int(torch.searchsorted(cum,.90*energy.sum()).item()+1)
            rank99=int(torch.searchsorted(cum,.99*energy.sum()).item()+1)
            row[kind]={"basis_capture":capture,"rank_90":rank90,"rank_99":rank99}
        layers[str(layer)]=row
    return {
        "normalized_logit_mse":float(np.mean(logits_mse)),
        "top1_agreement":float(np.mean(top1)),
        "student_cross_entropy":float(np.mean(student_ce)),
        "teacher_cross_entropy":float(np.mean(teacher_ce)),
        "hidden_normalized_mse":float(np.mean(hidden_error)),
        "layers":layers,
    }


def main():
    set_seed(SEED)
    if not torch.cuda.is_available():raise RuntimeError("CUDA required")
    device=torch.device("cuda")
    tokenizer=AutoTokenizer.from_pretrained(MODEL_ID)
    train_chunks,eval_chunks=make_chunks(tokenizer)
    teacher=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16).to(device).eval()
    student=AutoModelForCausalLM.from_pretrained(MODEL_ID,torch_dtype=torch.float16).to(device).train()
    for p in teacher.parameters():p.requires_grad_(False)
    bases=initialize_bases(teacher,train_chunks,device,TEMPORAL_RANK)
    optimizer=torch.optim.AdamW([
        {"params":student.parameters(),"lr":LR,"weight_decay":.01},
        {"params":bases.parameters(),"lr":BASIS_LR,"weight_decay":0.0},
    ])
    scaler=torch.amp.GradScaler("cuda")
    before=eval_metrics(student,teacher,bases,eval_chunks,device)
    history=[];started=time.time()
    for step,batch in enumerate(batch_iterator(train_chunks,BATCH,STEPS,SEED+2),1):
        x=batch.to(device);optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            t=teacher(x,output_hidden_states=True,use_cache=False)
        with torch.autocast("cuda",dtype=torch.float16):
            s=student(x,output_hidden_states=True,use_cache=False)
            logit_loss=normalized_mse(s.logits,t.logits)
            labels=x[:,1:]
            ce=F.cross_entropy(s.logits[:,:-1].float().reshape(-1,s.logits.shape[-1]),labels.reshape(-1))
            hidden= torch.stack([normalized_mse(s.hidden_states[l+1],t.hidden_states[l+1]) for l in SELECTED]).mean()
            temp,capture=temporal_loss(s.hidden_states,bases)
            loss=logit_loss+.05*ce+HIDDEN_WEIGHT*hidden+TEMPORAL_WEIGHT*temp
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer);torch.nn.utils.clip_grad_norm_(student.parameters(),1.0)
        scaler.step(optimizer);scaler.update()
        history.append({
            "step":step,"loss":float(loss.detach()),"logit_loss":float(logit_loss.detach()),
            "cross_entropy":float(ce.detach()),"hidden_loss":float(hidden.detach()),
            "temporal_tail":float(temp.detach()),"basis_capture":float(capture),
        })
        if step==1 or step%5==0:print("STEP_JSON="+json.dumps(history[-1]),flush=True)
    after=eval_metrics(student,teacher,bases,eval_chunks,device)
    result={
        "schema":"smollm2-temporal-innovation-finetune/v1","model":MODEL_ID,"parameters":sum(p.numel() for p in student.parameters()),
        "device":torch.cuda.get_device_name(),"rank":TEMPORAL_RANK,"selected_layers":list(SELECTED),"steps":STEPS,
        "temporal_weight":TEMPORAL_WEIGHT,"hidden_weight":HIDDEN_WEIGHT,"model_lr":LR,"basis_lr":BASIS_LR,
        "elapsed_seconds":time.time()-started,"before":before,"after":after,"history":history,
        "environment":{"torch":torch.__version__,"cuda":torch.version.cuda,"gpu_memory_GiB":torch.cuda.get_device_properties(0).total_memory/2**30,"platform":platform.platform()},
    }
    print("RESULT_JSON="+json.dumps(result,separators=(",",":"),sort_keys=True),flush=True)
    compact={
        "device":result["device"],"steps":STEPS,
        "before":{k:v for k,v in before.items() if k!="layers"},
        "after":{k:v for k,v in after.items() if k!="layers"},
        "layer_capture":{
            layer:{"teacher":row["teacher"]["basis_capture"],"student":row["student"]["basis_capture"],"teacher_rank99":row["teacher"]["rank_99"],"student_rank99":row["student"]["rank_99"]}
            for layer,row in after["layers"].items()
        },
    }
    print(json.dumps(compact,indent=2),flush=True)


if __name__=="__main__":main()
