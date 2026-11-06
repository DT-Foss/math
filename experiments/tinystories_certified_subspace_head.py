from __future__ import annotations
import os,time,json,torch
from transformers import AutoTokenizer,AutoModelForCausalLM
MODEL=os.environ.get('MODEL','roneneldan/TinyStories-1M')
THREADS=int(os.environ.get('THREADS','2'));torch.set_num_threads(THREADS)
CAL_PROMPT=('Once there was a curious child who explored forests, rivers, villages, machines, numbers, animals, and stars. '
            'The child learned that careful thinking and kindness could solve difficult problems. ')*30
TEST_PROMPTS=['Once upon a time','A small dog went to the park','One day the sun was shining']
N=128

def hidden_decode(model,tok,prompt,n,head=None,collect=False):
    ids=tok(prompt,return_tensors='pt').input_ids;past=None;prod=[];times=[];hidden=[];cand=[]
    with torch.no_grad():
        for _ in range(n):
            x=ids if past is None else torch.tensor([[prod[-1]]],dtype=torch.long)
            st=time.perf_counter_ns();o=model.transformer(input_ids=x,past_key_values=past,use_cache=True)
            h=o.last_hidden_state[:,-1,:].reshape(-1)
            if head is None:
                logits=torch.nn.functional.linear(h,model.lm_head.weight);nxt=int(logits.argmax());c=model.lm_head.weight.shape[0]
            else:nxt,c=head.argmax(h)
            times.append((time.perf_counter_ns()-st)/1e6);prod.append(nxt);cand.append(c);past=o.past_key_values
            if collect:hidden.append(h.detach().cpu().double())
    return prod,times,hidden,cand

class CertifiedSubspaceHead:
    def __init__(self,W,B,seed=32):
        self.W=W.detach().contiguous();self.B=B.detach().float().contiguous();self.seed=int(seed)
        with torch.no_grad():
            self.P=self.W@self.B
            recon=self.P@self.B.T
            self.rnorm=torch.linalg.vector_norm(self.W-recon,dim=1).double()
    def argmax(self,h):
        hd=h.double();B=self.B.double();qp=B.T@hd
        h2=torch.dot(hd,hd);q2=torch.dot(qp,qp);qr=float(torch.sqrt(torch.clamp(h2-q2,min=0)))
        approx=(self.P.double()@qp)
        ub=approx+self.rnorm*qr+1e-10
        k=min(self.seed,len(ub));seed_idx=torch.topk(ub,k).indices
        seed_scores=torch.mv(self.W.index_select(0,seed_idx),h)
        best=float(seed_scores.max())
        candidates=torch.nonzero(ub>best,as_tuple=False).flatten()
        candidates=torch.unique(torch.cat([seed_idx,candidates]))
        scores=torch.mv(self.W.index_select(0,candidates),h)
        j=int(scores.argmax());return int(candidates[j]),int(candidates.numel())

def main():
    tok=AutoTokenizer.from_pretrained(MODEL);m=AutoModelForCausalLM.from_pretrained(MODEL,torch_dtype=torch.float32);m.eval()
    _,_,H,_=hidden_decode(m,tok,CAL_PROMPT,256,collect=True)
    X=torch.stack(H);_,_,Vh=torch.linalg.svd(X,full_matrices=False)
    refs={};base={}
    for p in TEST_PROMPTS:
        prod,t,_,_=hidden_decode(m,tok,p,N)
        med=float(torch.tensor(t[1:]).median());refs[p]=prod;base[p]={'median_ms':med,'tok_s':1000/med}
    runs=[]
    for rank in [4,8,12,16,24,32,48]:
        B=Vh[:rank].T.contiguous();head=CertifiedSubspaceHead(m.lm_head.weight,B,seed=32);per=[]
        for p in TEST_PROMPTS:
            prod,t,_,cand=hidden_decode(m,tok,p,N,head=head)
            med=float(torch.tensor(t[1:]).median());ct=torch.tensor(cand[1:],dtype=torch.float32)
            per.append({'prompt':p,'tokens_equal':prod==refs[p],'median_ms':med,'tok_s':1000/med,
                        'speedup_vs_full':base[p]['median_ms']/med,'candidate_median':float(ct.median()),
                        'candidate_p90':float(torch.quantile(ct,.9)),'candidate_max':int(ct.max())})
        runs.append({'rank':rank,'projection_values':int(head.P.numel()),'per_prompt':per})
    print('RESULT '+json.dumps({'model':MODEL,'vocab':m.lm_head.weight.shape[0],'hidden':m.lm_head.weight.shape[1],
        'calibration_states':len(H),'baseline':base,'runs':runs}))
if __name__=='__main__':main()
