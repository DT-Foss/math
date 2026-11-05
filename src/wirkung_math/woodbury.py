from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class GeneralLowRankUpdate:
    """General square update ΔA = U V^T; no symmetry assumption."""
    U: np.ndarray
    V: np.ndarray

    def __post_init__(self):
        if self.U.ndim!=2 or self.V.ndim!=2 or self.U.shape!=self.V.shape:
            raise ValueError("U and V must have the same n×r shape")

    @property
    def rank_bound(self) -> int:
        return int(self.U.shape[1])

    def materialize(self) -> np.ndarray:
        return self.U@self.V.T


@dataclass
class GeneralWoodburyCache:
    """Shared solve/logdet state for A + U V^T."""
    A_inv: np.ndarray
    update: GeneralLowRankUpdate
    AU: np.ndarray
    middle: np.ndarray

    @classmethod
    def prepare(cls,A_inv: np.ndarray,update: GeneralLowRankUpdate) -> "GeneralWoodburyCache":
        A_inv=np.asarray(A_inv,dtype=np.float64)
        AU=A_inv@update.U
        middle=np.eye(update.rank_bound)+update.V.T@AU
        return cls(A_inv,update,AU,middle)

    def apply_inverse(self,rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim==1
        B=rhs[:,None] if vector else rhs
        z=self.A_inv@B
        out=z-self.AU@np.linalg.solve(self.middle,self.update.V.T@z)
        return out[:,0] if vector else out

    def logdet_update(self,base_sign: float,base_logabsdet: float) -> tuple[float,float]:
        sign,logabs=np.linalg.slogdet(self.middle)
        return float(base_sign*sign),float(base_logabsdet+logabs)


@dataclass
class SolverWoodburyCache:
    """
    Woodbury on top of any cached base solver; no explicit A^-1 is required.

    solve_base(B) must return A^-1 B.
    """
    solve_base: object
    update: GeneralLowRankUpdate
    AU: np.ndarray
    middle: np.ndarray

    @classmethod
    def prepare(cls,solve_base,update: GeneralLowRankUpdate) -> "SolverWoodburyCache":
        AU=np.asarray(solve_base(update.U),dtype=np.float64)
        middle=np.eye(update.rank_bound)+update.V.T@AU
        return cls(solve_base,update,AU,middle)

    def apply_inverse(self,rhs: np.ndarray) -> np.ndarray:
        rhs=np.asarray(rhs,dtype=np.float64)
        vector=rhs.ndim==1
        B=rhs[:,None] if vector else rhs
        z=np.asarray(self.solve_base(B),dtype=np.float64)
        out=z-self.AU@np.linalg.solve(self.middle,self.update.V.T@z)
        return out[:,0] if vector else out

    def logdet_update(self,base_sign: float,base_logabsdet: float) -> tuple[float,float]:
        sign,logabs=np.linalg.slogdet(self.middle)
        return float(base_sign*sign),float(base_logabsdet+logabs)
