import numpy as np
from wirkung_math.lazy_gram import LazyGramState

def test_lazy_gram_stream_solve_logdet_and_block():
    rng=np.random.default_rng(140)
    d=32
    z=rng.normal(size=(80,d)); G=z.T@z+10*np.eye(d)
    s=LazyGramState.from_gram(G)
    current=G.copy()
    for _ in range(3):
        old=rng.normal(scale=.1,size=(2,d))
        new=old+rng.normal(scale=.03,size=(2,d))
        s.replace_rows(old,new)
        current+=new.T@new-old.T@old
    rhs=rng.normal(size=(d,3))
    np.testing.assert_allclose(s.solve(rhs),np.linalg.solve(current,rhs),rtol=1e-10,atol=1e-10)
    sg,lg=s.slogdet; sr,lr=np.linalg.slogdet(current)
    assert sg==sr and abs(lg-lr)<1e-10
    I=np.array([1,5,11]);J=np.array([2,9,20,25])
    np.testing.assert_allclose(s.block(I,J),current[np.ix_(I,J)],rtol=1e-11,atol=1e-11)
