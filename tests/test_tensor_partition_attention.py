import numpy as np
from wirkung_math.tensor_train import TensorTrain
from wirkung_math.tensor_partition_attention import TensorTrainPartitionAttention


def test_tensor_train_exact_reconstruction_and_contraction():
    rng=np.random.default_rng(260)
    T=rng.normal(size=(3,4,2))
    tt=TensorTrain.from_tensor(T)
    np.testing.assert_allclose(tt.reconstruct(),T,rtol=1e-11,atol=1e-11)
    vectors=[rng.normal(size=3),rng.normal(size=4),rng.normal(size=2)]
    got=tt.contract_prefix(vectors)
    ref=np.einsum("ijk,i,j,k->",T,*vectors)
    assert abs(got-ref)<1e-10


def test_tensor_train_partition_attention_exact():
    rng=np.random.default_rng(261)
    n,dv=300,7
    levels=(np.array([-1.,0.,1.]),np.array([-2.,0.,2.]),np.array([-.5,.5]))
    codes=np.column_stack([rng.integers(0,len(level),size=n) for level in levels])
    base=rng.normal(size=n);values=rng.normal(size=(n,dv));alpha=rng.normal(scale=.2,size=3)
    state=TensorTrainPartitionAttention.compile(base,codes,levels,values,rtol=0.0)
    got,logz=state.query(alpha)
    signatures=np.column_stack([levels[j][codes[:,j]] for j in range(3)])
    scores=base+signatures@alpha;m=np.max(scores);w=np.exp(scores-m)
    ref=(w@values)/w.sum();logref=m+np.log(w.sum())
    np.testing.assert_allclose(got,ref,rtol=1e-10,atol=1e-10)
    assert abs(logz-logref)<1e-10
