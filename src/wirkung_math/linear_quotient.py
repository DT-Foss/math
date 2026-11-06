from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _row_space_basis(matrix:np.ndarray,tol:float|None=None)->np.ndarray:
    M=np.asarray(matrix,dtype=np.float64)
    if M.ndim!=2:
        raise ValueError("matrix must be two-dimensional")
    n=M.shape[1]
    if M.size==0 or M.shape[0]==0:
        return np.empty((n,0),dtype=np.float64)
    _,s,Vh=np.linalg.svd(M,full_matrices=False)
    if len(s)==0 or s[0]==0.0:
        rank=0
    else:
        threshold=(
            float(tol) if tol is not None
            else np.finfo(np.float64).eps*max(M.shape)*float(s[0])
        )
        rank=int(np.count_nonzero(s>threshold))
    return Vh[:rank].T.copy()


@dataclass(frozen=True)
class LinearConsumerQuotient:
    consumer:np.ndarray
    basis:np.ndarray
    reduced_consumer:np.ndarray
    reconstruction_residual:float

    @classmethod
    def fit(
        cls,consumer:np.ndarray,*,tol:float|None=None
    )->"LinearConsumerQuotient":
        C=np.asarray(consumer,dtype=np.float64)
        if C.ndim!=2:
            raise ValueError("consumer must be two-dimensional")
        Q=_row_space_basis(C,tol)
        Cr=C@Q
        residual=float(np.linalg.norm(C-Cr@Q.T,ord="fro"))
        return cls(C,Q,Cr,residual)

    @property
    def dimension(self)->int:
        return int(self.basis.shape[1])

    @property
    def input_dimension(self)->int:
        return int(self.basis.shape[0])

    def project_state(self,state:np.ndarray)->np.ndarray:
        x=np.asarray(state,dtype=np.float64)
        if x.shape!=(self.input_dimension,):
            raise ValueError("state shape mismatch")
        return self.basis.T@x

    def consumer_from_state(self,state:np.ndarray)->np.ndarray:
        z=np.asarray(state,dtype=np.float64)
        if z.shape!=(self.dimension,):
            raise ValueError("quotient-state shape mismatch")
        return self.reduced_consumer@z

    def compose_upstream(self,upstream:np.ndarray)->np.ndarray:
        A=np.asarray(upstream,dtype=np.float64)
        if A.ndim!=2 or A.shape[0]!=self.input_dimension:
            raise ValueError("upstream shape mismatch")
        return self.basis.T@A


@dataclass(frozen=True)
class ObservableLinearQuotient:
    A:np.ndarray
    B:np.ndarray
    C:np.ndarray
    basis:np.ndarray
    reduced_A:np.ndarray
    reduced_B:np.ndarray
    reduced_C:np.ndarray
    invariance_residual:float
    output_residual:float
    tolerance:float
    discovery_method:str
    closure_iterations:int
    generated_basis_columns:int

    @classmethod
    def fit(
        cls,A:np.ndarray,B:np.ndarray,C:np.ndarray,*,tol:float|None=None
    )->"ObservableLinearQuotient":
        A=np.asarray(A,dtype=np.float64)
        B=np.asarray(B,dtype=np.float64)
        C=np.asarray(C,dtype=np.float64)
        if A.ndim!=2 or A.shape[0]!=A.shape[1]:
            raise ValueError("A must be square")
        n=A.shape[0]
        if B.ndim!=2 or B.shape[0]!=n:
            raise ValueError("B shape mismatch")
        if C.ndim!=2 or C.shape[1]!=n:
            raise ValueError("C shape mismatch")

        # Smallest A^T-invariant subspace containing row(C).
        # This is the observable row space, discovered without materializing
        # the n-block observability matrix.
        Q=_row_space_basis(C,tol)
        generated=int(Q.shape[1])
        iterations=0
        for _ in range(max(1,n)):
            iterations+=1
            if Q.shape[1]==0:
                break
            image=A.T@Q
            generated+=int(image.shape[1])
            residual=image-Q@(Q.T@image)
            if residual.size==0:
                break
            U,s,_=np.linalg.svd(residual,full_matrices=False)
            scale=max(1.0,float(np.linalg.norm(image,ord="fro")))
            threshold=(
                float(tol) if tol is not None
                else 1024*np.finfo(np.float64).eps*max(1,n)*scale
            )
            add=int(np.count_nonzero(s>threshold))
            if add==0:
                break
            Q=np.concatenate([Q,U[:,:add]],axis=1)
            Q,_=np.linalg.qr(Q,mode="reduced")
            if Q.shape[1]>=n:
                break

        Aq=Q.T@A@Q
        Bq=Q.T@B
        Cq=C@Q
        lhs=Q.T@A
        rhs=Aq@Q.T
        inv=float(np.linalg.norm(lhs-rhs,ord="fro"))
        out=float(np.linalg.norm(C-Cq@Q.T,ord="fro"))
        scale=max(
            1.0,
            float(np.linalg.norm(lhs,ord="fro")),
            float(np.linalg.norm(C,ord="fro")),
        )
        tolerance=float(
            float(tol) if tol is not None
            else 256*np.finfo(np.float64).eps*max(1,n)*scale
        )
        return cls(
            A,B,C,Q,Aq,Bq,Cq,inv,out,tolerance,
            "krylov_closure",int(iterations),int(generated),
        )

    @property
    def dimension(self)->int:
        return int(self.basis.shape[1])

    @property
    def state_dimension(self)->int:
        return int(self.basis.shape[0])

    @property
    def is_exact(self)->bool:
        return bool(
            self.invariance_residual<=self.tolerance
            and self.output_residual<=self.tolerance
        )

    def project_state(self,state:np.ndarray)->np.ndarray:
        x=np.asarray(state,dtype=np.float64)
        if x.shape!=(self.state_dimension,):
            raise ValueError("state shape mismatch")
        return self.basis.T@x

    def step(self,state:np.ndarray,input_value:np.ndarray)->np.ndarray:
        z=np.asarray(state,dtype=np.float64)
        u=np.asarray(input_value,dtype=np.float64)
        if z.shape!=(self.dimension,):
            raise ValueError("quotient-state shape mismatch")
        if u.shape!=(self.reduced_B.shape[1],):
            raise ValueError("input shape mismatch")
        return self.reduced_A@z+self.reduced_B@u

    def output(self,state:np.ndarray)->np.ndarray:
        z=np.asarray(state,dtype=np.float64)
        if z.shape!=(self.dimension,):
            raise ValueError("quotient-state shape mismatch")
        return self.reduced_C@z
