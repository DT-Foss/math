from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _f64(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float64)


@dataclass
class SufficientStatisticsState:
    """FP64 additive statistics for a row set X."""
    count: int
    sum: np.ndarray
    sumsq: np.ndarray
    gram: np.ndarray

    @classmethod
    def from_rows(cls, x: np.ndarray) -> "SufficientStatisticsState":
        x=_f64(x)
        return cls(
            int(x.shape[0]),
            x.sum(axis=0),
            np.square(x).sum(axis=0),
            x.T @ x,
        )

    def replace_rows(self, old: np.ndarray, new: np.ndarray) -> None:
        old=_f64(old); new=_f64(new)
        if old.shape != new.shape or old.shape[1] != self.sum.shape[0]:
            raise ValueError("replacement shape mismatch")
        self.sum += new.sum(axis=0) - old.sum(axis=0)
        self.sumsq += np.square(new).sum(axis=0) - np.square(old).sum(axis=0)
        self.gram += new.T @ new - old.T @ old

    def insert_rows(self, rows: np.ndarray) -> None:
        rows=_f64(rows)
        if rows.ndim != 2 or rows.shape[1] != self.sum.shape[0]:
            raise ValueError("insert shape mismatch")
        self.count += int(rows.shape[0])
        self.sum += rows.sum(axis=0)
        self.sumsq += np.square(rows).sum(axis=0)
        self.gram += rows.T @ rows

    def delete_rows(self, rows: np.ndarray) -> None:
        rows=_f64(rows)
        if rows.ndim != 2 or rows.shape[1] != self.sum.shape[0]:
            raise ValueError("delete shape mismatch")
        if rows.shape[0] > self.count:
            raise ValueError("cannot delete more rows than state contains")
        self.count -= int(rows.shape[0])
        self.sum -= rows.sum(axis=0)
        self.sumsq -= np.square(rows).sum(axis=0)
        self.gram -= rows.T @ rows

    @property
    def mean(self) -> np.ndarray:
        return self.sum / self.count

    @property
    def variance(self) -> np.ndarray:
        mu=self.mean
        return self.sumsq / self.count - mu*mu

    @property
    def covariance(self) -> np.ndarray:
        mu=self.mean
        return self.gram / self.count - np.outer(mu,mu)


@dataclass
class NormalEquationsState:
    """Cached X^T X, X^T Y and Y^T Y for exact row replacement/add/remove."""
    count: int
    gram: np.ndarray
    cross: np.ndarray
    target_gram: np.ndarray

    @classmethod
    def from_xy(cls, x: np.ndarray, y: np.ndarray) -> "NormalEquationsState":
        x=_f64(x); y=_f64(y)
        if y.ndim == 1:
            y=y[:,None]
        if x.shape[0] != y.shape[0]:
            raise ValueError("row mismatch")
        return cls(int(x.shape[0]), x.T@x, x.T@y, y.T@y)

    def replace_rows(
        self, old_x: np.ndarray, new_x: np.ndarray,
        old_y: np.ndarray, new_y: np.ndarray,
    ) -> None:
        old_x=_f64(old_x); new_x=_f64(new_x)
        old_y=_f64(old_y); new_y=_f64(new_y)
        if old_y.ndim == 1: old_y=old_y[:,None]
        if new_y.ndim == 1: new_y=new_y[:,None]
        if old_x.shape != new_x.shape or old_y.shape != new_y.shape:
            raise ValueError("replacement shape mismatch")
        self.gram += new_x.T@new_x - old_x.T@old_x
        self.cross += new_x.T@new_y - old_x.T@old_y
        self.target_gram += new_y.T@new_y - old_y.T@old_y

    def solve(self, ridge: float=0.0) -> np.ndarray:
        a=self.gram.copy()
        if ridge:
            a.flat[::a.shape[0]+1] += ridge
        return np.linalg.solve(a,self.cross)
