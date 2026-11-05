from __future__ import annotations
import numpy as np
from .symmetric import SymmetricLowRankDelta


def laplacian_edge_delta(
    n: int, i: int, j: int, delta_weight: float
) -> SymmetricLowRankDelta:
    """Exact graph-Laplacian update for changing one undirected edge weight."""
    if i==j or min(i,j)<0 or max(i,j)>=n:
        raise ValueError("invalid edge")
    b=np.zeros((n,1),dtype=np.float64)
    b[i,0]=1.0; b[j,0]=-1.0
    return SymmetricLowRankDelta(b,np.array([[float(delta_weight)]]))


def laplacian_edges_delta(
    n: int, edges: list[tuple[int,int,float]]
) -> SymmetricLowRankDelta:
    """Batch e edge changes as a rank-at-most-e symmetric update."""
    U=np.zeros((n,len(edges)),dtype=np.float64)
    C=np.zeros((len(edges),len(edges)),dtype=np.float64)
    for k,(i,j,dw) in enumerate(edges):
        if i==j or min(i,j)<0 or max(i,j)>=n:
            raise ValueError("invalid edge")
        U[i,k]=1.0; U[j,k]=-1.0; C[k,k]=float(dw)
    return SymmetricLowRankDelta(U,C)
