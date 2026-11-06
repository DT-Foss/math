from __future__ import annotations
import os,time,json,tempfile,subprocess,ctypes
import numpy as np, torch
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL=os.environ.get('MODEL','roneneldan/TinyStories-1M');torch.set_num_threads(int(os.environ.get('THREADS','2')))
CAL=('Once there was a curious child who explored forests, rivers, villages, machines, numbers, animals, and stars. The child learned that careful thinking and kindness could solve difficult problems. ')*30
PROMPTS=['Once upon a time','A small dog went to the park','One day the sun was shining'];N=128
C=r'''
#include <stdint.h>
#include <stdlib.h>
#include <math.h>
#include <float.h>
int prune_vocab(const float*h,const float*W,const double*B,const double*P,const double*R,const double*WN,
 int V,int D,const int*stages,int S,int seed,int*counts,int*out_ids){
 double*q=(double*)calloc((size_t)D,sizeof(double)); int*cand=(int*)malloc((size_t)V*sizeof(int));
 double*approx=(double*)calloc((size_t)V,sizeof(double)); double*ub=(double*)malloc((size_t)V*sizeof(double));
 double hnorm2=0.0;for(int i=0;i<D;i++){hnorm2+=(double)h[i]*h[i];for(int j=0;j<D;j++)q[j]+=(double)h[i]*B[(size_t)i*D+j];}
 double hnorm=sqrt(hnorm2); for(int i=0;i<V;i++)cand[i]=i;int nc=V,prev=0;double best=-DBL_MAX;
 const double u=5.9604644775390625e-8; const double gamma=(D*u)/(1.0-D*u);
 for(int s=0;s<S;s++){
  int r=stages[s];double qtail=0.0;for(int j=r;j<D;j++)qtail+=q[j]*q[j];qtail=sqrt(qtail);
  int topn=0;double topv[32];int topid[32];
  for(int z=0;z<nc;z++){int id=cand[z];double a=approx[z];for(int j=prev;j<r;j++)a+=P[(size_t)id*D+j]*q[j];approx[z]=a;
   double v=a+R[(size_t)s*V+id]*qtail+gamma*WN[id]*hnorm+1e-9;ub[z]=v;
   if(topn<seed){int p=topn++;while(p>0&&topv[p-1]<v){topv[p]=topv[p-1];topid[p]=topid[p-1];p--;}topv[p]=v;topid[p]=id;}
   else if(v>topv[topn-1]){int p=topn-1;while(p>0&&topv[p-1]<v){topv[p]=topv[p-1];topid[p]=topid[p-1];p--;}topv[p]=v;topid[p]=id;}
  }
  for(int z=0;z<topn;z++){int id=topid[z];double sc=0.0;for(int j=0;j<D;j++)sc+=(double)W[(size_t)id*D+j]*h[j];double lb=sc-gamma*WN[id]*hnorm-1e-9;if(lb>best)best=lb;}
  int w=0;for(int z=0;z<nc;z++)if(ub[z]>=best){cand[w]=cand[z];approx[w]=approx[z];w++;}nc=w;counts[s]=nc;prev=r;if(nc<=1)break;
 }
 for(int i=0;i<nc;i++)out_ids[i]=cand[i];free(q);free(cand);free(approx);free(ub);return nc;
}
''';
def build():
 d=tempfile.mkdtemp();src=os.path.join(d,'p.c');so=os.path.join(d,'p.so');open(src,'w').write(C);subprocess.run(['gcc','-O3','-march=native','-shared','-fPIC',src,'-lm','-o',so],check=True);return ctypes.CDLL(so)
def decode(m,tok,prompt,n,head=None,collect=False):
 ids=tok(prompt,return_tensors='pt').input_ids;past=None;p=[];ts=[];H=[];cs=[]
 with torch.no_grad():
  for _ in range(n):
   x=ids if past is None else torch.tensor([[p[-1]]]);st=time.perf_counter_ns();o=m.transformer(input_ids=x,past_key_values=past,use_cache=True);h=o.last_hidden_state[:,-1,:].reshape(-1)
   if head is None:nxt=int(torch.mv(m.lm_head.weight,h).argmax());c=None
   else:nxt,c=head.argmax(h)
   ts.append((time.perf_counter_ns()-st)/1e6);p.append(nxt);past=o.past_key_values
   if collect:H.append(h.double().cpu())
   if c is not None:cs.append(c)
 return p,ts,H,cs
class NativeHead:
 def __init__(self,W,B,stages=(4,12,24,40,56),seed=16):
  self.W=W.detach().cpu().contiguous();self.Wn=np.ascontiguousarray(self.W.numpy());self.B=np.ascontiguousarray(B.numpy().astype(np.float64));Wd=self.Wn.astype(np.float64);self.P=np.ascontiguousarray(Wd@self.B)
  wn=np.linalg.norm(Wd,axis=1);cum=np.cumsum(self.P*self.P,axis=1);self.R=np.ascontiguousarray(np.stack([np.sqrt(np.maximum(wn*wn-cum[:,r-1],0)) for r in stages]));self.WN=np.ascontiguousarray(wn);self.stages=np.array(stages,np.int32);self.seed=seed;self.V,self.D=self.Wn.shape;self.S=len(stages);self.lib=build()
  PF=ctypes.POINTER(ctypes.c_float);PD=ctypes.POINTER(ctypes.c_double);PI=ctypes.POINTER(ctypes.c_int);self.lib.prune_vocab.argtypes=[PF,PF,PD,PD,PD,PD,ctypes.c_int,ctypes.c_int,PI,ctypes.c_int,ctypes.c_int,PI,PI];self.counts=np.zeros(self.S,np.int32);self.ids=np.zeros(self.V,np.int32)
 def argmax(self,h):
  hn=np.ascontiguousarray(h.detach().cpu().numpy().astype(np.float32));self.counts.fill(0);PF=ctypes.POINTER(ctypes.c_float);PD=ctypes.POINTER(ctypes.c_double);PI=ctypes.POINTER(ctypes.c_int)
  nc=self.lib.prune_vocab(hn.ctypes.data_as(PF),self.Wn.ctypes.data_as(PF),self.B.ctypes.data_as(PD),self.P.ctypes.data_as(PD),self.R.ctypes.data_as(PD),self.WN.ctypes.data_as(PD),self.V,self.D,self.stages.ctypes.data_as(PI),self.S,self.seed,self.counts.ctypes.data_as(PI),self.ids.ctypes.data_as(PI))
  ids=torch.from_numpy(self.ids[:nc].copy()).long();scores=torch.mv(self.W.index_select(0,ids),h.cpu());return int(ids[int(scores.argmax())]),self.counts.copy().tolist()
def main():
 tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval();_,_,H,_=decode(m,tok,CAL,256,collect=True);_,_,Vh=torch.linalg.svd(torch.stack(H),full_matrices=False);B=Vh.T.contiguous()
 refs={};base={}
 for p in PROMPTS:
  z,t,_,_=decode(m,tok,p,N);med=float(torch.tensor(t[1:]).median());refs[p]=z;base[p]={'tok_s':1000/med,'median_ms':med}
 runs=[]
 for stages in [(4,12,24,40,56),(8,16,32,48,56)]:
  head=NativeHead(m.lm_head.weight,B,stages=stages,seed=16);per=[]
  for p in PROMPTS:
   z,t,_,cs=decode(m,tok,p,N,head=head);med=float(torch.tensor(t[1:]).median());arr=np.array(cs[1:],np.float64);per.append({'prompt':p,'tokens_equal':z==refs[p],'tok_s':1000/med,'speedup':base[p]['median_ms']/med,'candidate_median':np.median(arr,axis=0).tolist(),'candidate_p90':np.percentile(arr,90,axis=0).tolist()})
  runs.append({'stages':stages,'results':per})
 print('RESULT '+json.dumps({'baseline':base,'runs':runs}))
if __name__=='__main__':main()
