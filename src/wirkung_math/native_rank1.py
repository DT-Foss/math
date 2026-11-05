from __future__ import annotations
from dataclasses import dataclass
import ctypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import platform
import sys
import numpy as np

_C_SOURCE = r"""
#include <stddef.h>

void rank1_stats_update_f64(
    const double *oldx,
    const double *newx,
    int d,
    double *sum_state,
    double *sumsq_state,
    double *gram_state
){
    double diff[256];
    double plus[256];
    if (d > 256) return;
    for(int i=0;i<d;i++){
        const double di = newx[i]-oldx[i];
        const double pi = newx[i]+oldx[i];
        diff[i]=di;
        plus[i]=pi;
        sum_state[i]+=di;
        sumsq_state[i]+=di*pi;
    }
    for(int i=0;i<d;i++){
        const double di=diff[i];
        const double pi=plus[i];
        #pragma GCC ivdep
        for(int j=0;j<d;j++){
            gram_state[(size_t)i*(size_t)d + (size_t)j] +=
                0.5*(di*plus[j] + pi*diff[j]);
        }
    }
}

void rank1_stats_update_packed_f64(
    const double *oldx,
    const double *newx,
    int d,
    double *sum_state,
    double *sumsq_state,
    double *gram_upper
){
    double diff[256];
    double plus[256];
    if (d > 256) return;
    for(int i=0;i<d;i++){
        const double di = newx[i]-oldx[i];
        const double pi = newx[i]+oldx[i];
        diff[i]=di;
        plus[i]=pi;
        sum_state[i]+=di;
        sumsq_state[i]+=di*pi;
    }
    for(int i=0;i<d;i++){
        const double di=diff[i];
        const double pi=plus[i];
        const size_t base=(size_t)i*(size_t)d - ((size_t)i*(size_t)(i-1))/2u;
        for(int j=i;j<d;j++){
            gram_upper[base+(size_t)(j-i)] +=
                0.5*(di*plus[j] + pi*diff[j]);
        }
    }
}
"""


def _compiler() -> str:
    for name in (os.environ.get("CC"),"clang","gcc","cc"):
        if name:
            path=shutil.which(name)
            if path:
                return path
    raise RuntimeError("No C compiler found")


def build_rank1_stats_kernel(directory: str|os.PathLike[str]|None=None) -> str:
    out=Path(directory or tempfile.mkdtemp(prefix="wirkung-rank1-"))
    out.mkdir(parents=True,exist_ok=True)
    src=out/"rank1_stats.c"
    lib=out/("rank1_stats.dylib" if sys.platform=="darwin" else ("rank1_stats.dll" if os.name=="nt" else "rank1_stats.so"))
    src.write_text(_C_SOURCE)
    arch=["-mcpu=native"] if sys.platform=="darwin" and platform.machine().lower() in {"arm64","aarch64"} else ["-march=native"]
    cmd=[_compiler(),"-O3",*arch,"-shared","-fPIC",str(src),"-o",str(lib)]
    subprocess.run(cmd,check=True)
    return str(lib)


@dataclass
class NativeRank1StatsState:
    sum:np.ndarray
    sumsq:np.ndarray
    gram:np.ndarray
    library_path:str|None=None

    def __post_init__(self):
        self.sum=np.ascontiguousarray(self.sum,dtype=np.float64)
        self.sumsq=np.ascontiguousarray(self.sumsq,dtype=np.float64)
        self.gram=np.ascontiguousarray(self.gram,dtype=np.float64)
        d=len(self.sum)
        if self.sumsq.shape!=(d,) or self.gram.shape!=(d,d):
            raise ValueError("state shape mismatch")
        path=self.library_path or build_rank1_stats_kernel()
        self._lib=ctypes.CDLL(path)
        self._fn=self._lib.rank1_stats_update_f64
        self._fn.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_int,ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p]
        self._d=d
        self._sum_ptr=ctypes.c_void_p(self.sum.ctypes.data)
        self._sumsq_ptr=ctypes.c_void_p(self.sumsq.ctypes.data)
        self._gram_ptr=ctypes.c_void_p(self.gram.ctypes.data)

    def update(self,old_row:np.ndarray,new_row:np.ndarray)->None:
        old=np.ascontiguousarray(old_row,dtype=np.float64)
        new=np.ascontiguousarray(new_row,dtype=np.float64)
        if old.shape!=(self._d,) or new.shape!=(self._d,):
            raise ValueError("row shape mismatch")
        self._fn(
            ctypes.c_void_p(old.ctypes.data),ctypes.c_void_p(new.ctypes.data),self._d,
            self._sum_ptr,self._sumsq_ptr,self._gram_ptr,
        )
