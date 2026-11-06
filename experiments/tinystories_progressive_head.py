from __future__ import annotations
import os,time,json,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL=os.environ.get('MODEL','roneneldan/TinyStories-1M');torch.set_num_threads(int(os.environ.get('THREADS','2')))
CAL=('Once there was a curious child who explored forests, rivers, villages, machines, numbers, animals, and stars. The child learned that careful thinking and kindness could solve difficult problems. ')*30
PROMPTS=['Once upon a time','A small dog went to the park','One day the sun was shining'];N=128

def decode(m,tok,prompt,n,head=None,collect=False):
 ids=tok(prompt,return_tensors='pt').input_ids;past=None;p=[];times=[];H=[];stats=[]
 with torch.no_grad():
  for _ in range(n):
   x=ids if past is None else torch.tensor([[p[-1]]]);st=time.perf_counter_ns();o=m.transformer(input_ids=x,past_key_values=past,use_cache=True);h=o.last_hidden_state[:,-1,:].reshape(-1)
   if head is None:nxt=int(torch.mv(m.lm_head.weight,h).argmax());s=None
   else:nxt,s=head.argmax(h)
   times.append((time.perf_counter_ns()-st)/1e6);p.append(nxt);past=o.past_key_values
   if collect:H.append(h.double().cpu())
   if s is not None:stats.append(s)
 return p,times,H,stats

class ProgressiveHead:
 def __init__(self,W,B,stages=(4,8,16,32,48),seed=16):
  self.W=W.contiguous();self.B=B.float().contiguous();self.stages=tuple(stages);self.seed=seed
  self.P=self.W@self.B
  w2=(self.W.double()**2).sum(1);cum=torch.cumsum(self.P.double()**2,1)
  self.rnorm={r:torch.sqrt(torch.clamp(w2-cum[:,r-1],min=0)) for r in self.stages}
 def argmax(self,h):
  hd=h.double();q=self.B.double().T@hd;q2=torch.dot(hd,hd);qc=torch.cumsum(q*q,0)
  V=self.W.shape[0];cand=torch.arange(V);approx=torch.zeros(V,dtype=torch.float64);prev=0;counts=[];best=-float('inf');bestid=-1
  for r in self.stages:
   approx=approx+self.P.index_select(0,cand)[:,prev:r].double()@q[prev:r]
   qres=float(torch.sqrt(torch.clamp(q2-qc[r-1],min=0)))
   ub=approx+self.rnorm[r].index_select(0,cand)*qres+1e-10
   kk=min(self.seed,len(cand));local=torch.topk(ub,kk).indices;ids=cand[local];scores=torch.mv(self.W.index_select(0,ids),h)
   j=int(scores.argmax());v=float(scores[j])
   if v>best:best=v;bestid=int(ids[j])
   keep=ub>best;keep |= cand==bestid
   cand=cand[keep];approx=approx[keep];counts.append(int(len(cand)));prev=r
   if len(cand)<=1:break
  if len(cand)>1:
   scores=torch.mv(self.W.index_select(0,cand),h);j=int(scores.argmax());bestid=int(cand[j])
  return bestid,counts

def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval()
 _,_,H,_=decode(m,tok,CAL,256,collect=True);X=torch.stack(H);_,_,Vh=torch.linalg.svd(X,full_matrices=False);B=Vh.T.contiguous()
 refs={};base={}
 for p in PROMPTS:
  z,t,_,_=decode(m,tok,p,N);med=float(torch.tensor(t[1:]).median());refs[p]=z;base[p]={'tok_s':1000/med,'median_ms':med}
 cases=[]
 for stages in [(4,8,16,32,48),(8,16,32,48),(4,12,24,40,56)]:
  head=ProgressiveHead(m.lm_head.weight,B,stages=stages,seed=16);res=[]
  for p in PROMPTS:
   z,t,_,ss=decode(m,tok,p,N,head=head);med=float(torch.tensor(t[1:]).median());maxlen=max(len(x) for x in ss);stagecounts=[]
   for i in range(maxlen):
    v=[x[i] for x in ss if len(x)>i];stagecounts.append({'median':float(torch.tensor(v,dtype=torch.float32).median()),'p90':float(torch.quantile(torch.tensor(v,dtype=torch.float32),.9))})
   res.append({'prompt':p,'tokens_equal':z==refs[p],'tok_s':1000/med,'speedup':base[p]['median_ms']/med,'stage_candidates':stagecounts})
  cases.append({'stages':stages,'results':res})
 print('RESULT '+json.dumps({'model':MODEL,'baseline':base,'cases':cases}))
if __name__=='__main__':main()
