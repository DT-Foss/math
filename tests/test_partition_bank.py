import numpy as np
from wirkung_math.partition_attention import PartitionAttentionBank


def dense(base,codes,values,support,alpha):
    scores=base+codes[:,support]@alpha
    m=np.max(scores);w=np.exp(scores-m)
    return (w@values)/w.sum()


def test_partition_bank_from_key_codes_and_append():
    rng=np.random.default_rng(222)
    n,width,dv=300,8,12
    codes=rng.integers(-2,3,size=(n,width));base=rng.normal(size=n);values=rng.normal(size=(n,dv))
    bank=PartitionAttentionBank.compile_supports(base,codes,values,[(1,4),(0,2,7)])
    alpha=np.array([.2,-.1])
    got,_=bank.query((1,4),alpha)
    np.testing.assert_allclose(got,dense(base,codes,values,(1,4),alpha),rtol=1e-12,atol=1e-12)

    code=rng.integers(-2,3,size=width);score=2.0;value=rng.normal(size=dv)
    bank.append(score,code,value)
    codes=np.vstack([codes,code]);base=np.append(base,score);values=np.vstack([values,value])
    got,_=bank.query((1,4),alpha)
    np.testing.assert_allclose(got,dense(base,codes,values,(1,4),alpha),rtol=1e-12,atol=1e-12)
