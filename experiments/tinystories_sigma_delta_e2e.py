from __future__ import annotations
import json, time, gc, os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL=os.environ.get('MODEL','roneneldan/TinyStories-1M')
THREADS=int(os.environ.get('THREADS','2'))
torch.set_num_threads(THREADS)

COMPILE_PROMPTS=["Once upon a time", "There was a little girl"]
VALID_PROMPT="A little dog was playing outside"
TEST_PROMPT="One day the sun was shining"
COMPILE_TOKENS=40
VALID_TOKENS=64
TEST_TOKENS=128

CANDIDATES={
 'transformer.h.0.attn.attention.q_proj':3,'transformer.h.0.attn.attention.k_proj':4,
 'transformer.h.0.attn.attention.v_proj':4,'transformer.h.0.attn.attention.out_proj':5,
 'transformer.h.0.mlp.c_proj':8,'transformer.h.1.attn.attention.k_proj':6,
 'transformer.h.1.attn.attention.v_proj':5,'transformer.h.1.attn.attention.out_proj':8,
 'transformer.h.1.mlp.c_proj':6,'transformer.h.2.attn.attention.q_proj':4,
 'transformer.h.2.attn.attention.out_proj':6,'transformer.h.2.mlp.c_proj':7,
 'transformer.h.3.attn.attention.q_proj':5,'transformer.h.3.attn.attention.v_proj':6,
 'transformer.h.3.attn.attention.out_proj':8,'transformer.h.4.attn.attention.q_proj':4,
 'transformer.h.4.attn.attention.v_proj':5,'transformer.h.4.attn.attention.out_proj':6,
 'transformer.h.4.mlp.c_proj':8,'transformer.h.5.attn.attention.q_proj':4,
 'transformer.h.5.attn.attention.k_proj':5,'transformer.h.5.attn.attention.v_proj':5,
 'transformer.h.5.attn.attention.out_proj':6,'transformer.h.5.mlp.c_proj':7,
 'transformer.h.6.attn.attention.q_proj':4,'transformer.h.6.attn.attention.v_proj':6,
 'transformer.h.6.attn.attention.out_proj':7,'transformer.h.7.attn.attention.q_proj':5,
 'transformer.h.7.attn.attention.v_proj':4,'transformer.h.7.attn.attention.out_proj':6,
}

def parent_child(root,name):
    parts=name.split('.'); p=root
    for part in parts[:-1]: p=getattr(p,part)
    return p,parts[-1]

class SigmaDeltaLinear(torch.nn.Module):
    def __init__(self,base,step,bits,name,rebase_interval=32):
        super().__init__(); self.name=name; self.bits=int(bits)
        self.in_features=base.in_features; self.out_features=base.out_features
        self.register_buffer('weight',base.weight.detach().contiguous())
        self.register_buffer('weight_t',base.weight.detach().T.contiguous())
        if base.bias is None: self.bias=None
        else: self.register_buffer('bias',base.bias.detach().contiguous())
        self.register_buffer('step',step.detach().contiguous().clamp_min(1e-8))
        self.mode='full'; self.rebase_interval=int(rebase_interval); self.reset()
    def reset(self):
        self.hat=None; self.last_y=None; self.decode_steps=0
        self.event_fractions=[]; self.event_counts=[]; self.full_rebases=0
    def set_mode(self,mode): self.mode=mode; self.reset()
    def _full(self,x): return torch.nn.functional.linear(x,self.weight,self.bias)
    def forward(self,x):
        rows=x.numel()//x.shape[-1]
        if rows != 1 or self.hat is None:
            y=self._full(x)
            last=x.reshape(-1,self.in_features)[-1].detach()
            self.hat=last.clone(); self.last_y=self._full(last).detach().clone()
            self.decode_steps=0; return y
        cur=x.reshape(self.in_features); innovation=cur-self.hat
        pulse=torch.round(innovation/self.step)*self.step
        idx=torch.nonzero(pulse,as_tuple=False).flatten()
        self.event_counts.append(int(idx.numel())); self.event_fractions.append(float(idx.numel()/self.in_features))
        self.hat=self.hat+pulse; self.decode_steps+=1
        if self.mode=='full':
            y=self._full(self.hat); self.last_y=y.detach().clone()
            return y.view(*x.shape[:-1],self.out_features)
        if self.decode_steps % self.rebase_interval == 0:
            y=self._full(self.hat); self.last_y=y.detach().clone(); self.full_rebases+=1
        else:
            if idx.numel():
                dy=pulse.index_select(0,idx).matmul(self.weight_t.index_select(0,idx))
                self.last_y=self.last_y+dy
            y=self.last_y
        return y.view(*x.shape[:-1],self.out_features)

def wrappers(model): return [m for m in model.modules() if isinstance(m,SigmaDeltaLinear)]
def reset_mode(model,mode):
    for m in wrappers(model): m.set_mode(mode)

def decode(model,tok,prompt,n):
    ids=tok(prompt,return_tensors='pt').input_ids; past=None; produced=[]; times=[]
    with torch.no_grad():
        for _ in range(n):
            x=ids if past is None else torch.tensor([[produced[-1]]],dtype=torch.long)
            st=time.perf_counter_ns(); out=model(input_ids=x,past_key_values=past,use_cache=True)
            times.append((time.perf_counter_ns()-st)/1e6)
            produced.append(int(out.logits[:,-1,:].argmax())); past=out.past_key_values
    return produced,times

def same_on(model,tok,refs,prompts,n):
    reset_mode(model,'full')
    for p in prompts:
        got,_=decode(model,tok,p,n)
        if got!=refs[p]: return False
    return True

def main():
    tok=AutoTokenizer.from_pretrained(MODEL)
    model=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32); model.eval()
    amax={n:None for n in CANDIDATES}; hooks=[]; named=dict(model.named_modules())
    for name in CANDIDATES:
        mod=named[name]
        def hook(mod,args,name=name):
            v=args[0].detach().reshape(-1,args[0].shape[-1]).abs().amax(0).cpu()
            amax[name]=v if amax[name] is None else torch.maximum(amax[name],v)
        hooks.append(mod.register_forward_pre_hook(hook))
    refs={}
    for p in COMPILE_PROMPTS+[VALID_PROMPT,TEST_PROMPT]:
        n=TEST_TOKENS if p==TEST_PROMPT else (VALID_TOKENS if p==VALID_PROMPT else COMPILE_TOKENS)
        refs[p],_=decode(model,tok,p,n)
    for h in hooks:h.remove()

    originals={}; selected={}
    for name,min_static in CANDIDATES.items():
        parent,child=parent_child(model,name); orig=getattr(parent,child); originals[name]=orig
        kept=False
        for bits in range(max(2,min_static-2),9):
            qmax=2**(bits-1)-1; step=(amax[name]/qmax).clamp_min(1e-7)
            setattr(parent,child,SigmaDeltaLinear(orig,step,bits,name,32))
            if same_on(model,tok,refs,COMPILE_PROMPTS,COMPILE_TOKENS):
                selected[name]=bits; kept=True; break
            setattr(parent,child,orig)
        if not kept: setattr(parent,child,orig)
    compile_selected=dict(selected)

    def valid():
        reset_mode(model,'full'); got,_=decode(model,tok,VALID_PROMPT,VALID_TOKENS)
        return got==refs[VALID_PROMPT]
    if not valid():
        for name in list(selected.keys())[::-1]:
            parent,child=parent_child(model,name); setattr(parent,child,originals[name]); selected.pop(name)
            if valid(): break

    base=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);base.eval()
    baseline={}
    for p,n in [(VALID_PROMPT,VALID_TOKENS),(TEST_PROMPT,TEST_TOKENS)]:
        decode(base,tok,p,min(16,n))
        prod,t=decode(base,tok,p,n); med=float(torch.tensor(t[1:]).median())
        baseline[p]={'tokens_equal_reference':prod==refs[p],'median_ms':med,'tok_s':1000/med}
    del base; gc.collect()

    results={}
    for p,n in [(VALID_PROMPT,VALID_TOKENS),(TEST_PROMPT,TEST_TOKENS)]:
        reset_mode(model,'full'); full,tfull=decode(model,tok,p,n)
        reset_mode(model,'delta'); temporal,tstate=decode(model,tok,p,n)
        full_med=float(torch.tensor(tfull[1:]).median()); temp_med=float(torch.tensor(tstate[1:]).median())
        mods=wrappers(model); events=[x for m in mods for x in m.event_fractions]
        results[p]={
            'n':n,'fp_reference_equal_full_sigma':full==refs[p],
            'fp_reference_prefix_full_sigma':next((i for i,(a,b) in enumerate(zip(full,refs[p])) if a!=b),n),
            'temporal_equal_full_sigma':temporal==full,
            'full_sigma_tok_s':1000/full_med,'temporal_sigma_tok_s':1000/temp_med,
            'temporal_speedup_vs_full_sigma':full_med/temp_med,
            'temporal_speedup_vs_fp32':baseline[p]['median_ms']/temp_med,
            'event_fraction_median':float(torch.tensor(events).median()) if events else 0.0,
            'event_fraction_p90':float(torch.quantile(torch.tensor(events),.9)) if events else 0.0,
            'selected_modules':len(selected),'full_rebases':sum(m.full_rebases for m in mods),
        }
    print('RESULT '+json.dumps({'model':MODEL,'threads':THREADS,'compile_selected':compile_selected,
        'validated_selected':selected,'baseline':baseline,'results':results}))
if __name__=='__main__': main()
