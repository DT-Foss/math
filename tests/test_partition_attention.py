import numpy as np
from wirkung_math.partition_attention import CompiledPartitionAttention


def dense(base,signatures,values,alpha):
    scores=base+signatures@alpha
    m=np.max(scores);w=np.exp(scores-m)
    return (w@values)/w.sum(),float(m+np.log(w.sum()))


def test_compiled_partition_attention_exact():
    rng=np.random.default_rng(220)
    n,r,dv=512,3,16
    codebook=rng.integers(-2,3,size=(40,r))
    signatures=codebook[rng.integers(0,len(codebook),size=n)]
    base=rng.normal(size=n);values=rng.normal(size=(n,dv));alpha=rng.normal(scale=.2,size=r)
    p=CompiledPartitionAttention.compile(base,signatures,values)
    got,logz=p.query(alpha);ref,logref=dense(base,signatures,values,alpha)
    np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-12)
    assert abs(logz-logref)<1e-12
    assert p.occupied_groups<=40


def test_partition_append_exact():
    rng=np.random.default_rng(221)
    n,r,dv=100,2,8
    signatures=rng.integers(-1,2,size=(n,r));base=rng.normal(size=n);values=rng.normal(size=(n,dv))
    p=CompiledPartitionAttention.compile(base,signatures,values)
    sig=np.array([1,-1]);score=3.0;value=rng.normal(size=dv)
    p.append(score,sig,value)
    signatures=np.vstack([signatures,sig]);base=np.append(base,score);values=np.vstack([values,value])
    alpha=np.array([.2,-.1])
    got,logz=p.query(alpha);ref,logref=dense(base,signatures,values,alpha)
    np.testing.assert_allclose(got,ref,rtol=1e-12,atol=1e-12)
    assert abs(logz-logref)<1e-12
