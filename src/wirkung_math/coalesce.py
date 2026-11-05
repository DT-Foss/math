from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


@dataclass
class RowReplacementLog:
    """
    Coalesce repeated edits to the same row.

    The first observed old value and latest new value are sufficient for any
    downstream replacement algebra. Intermediate writes disappear.
    """
    _old: dict[int,np.ndarray]=field(default_factory=dict)
    _new: dict[int,np.ndarray]=field(default_factory=dict)

    def add(self, index: int, old: np.ndarray, new: np.ndarray) -> None:
        i=int(index)
        old=np.asarray(old).copy(); new=np.asarray(new).copy()
        if i not in self._old:
            self._old[i]=old
        self._new[i]=new
        if np.array_equal(self._old[i],self._new[i]):
            del self._old[i]; del self._new[i]

    @property
    def support(self) -> int:
        return len(self._old)

    def arrays(self) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
        idx=np.array(sorted(self._old),dtype=np.int64)
        if not len(idx):
            return idx,np.empty((0,)),np.empty((0,))
        old=np.stack([self._old[int(i)] for i in idx])
        new=np.stack([self._new[int(i)] for i in idx])
        return idx,old,new
