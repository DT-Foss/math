"""Portable native C++ build helpers for Wirkung Remath."""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Iterable


def shared_library_suffix() -> str:
    if sys.platform == "darwin":
        return ".dylib"
    if os.name == "nt":
        return ".dll"
    return ".so"


def find_cxx() -> str:
    candidates=[]
    if os.environ.get("CXX"):
        candidates.append(os.environ["CXX"])
    candidates += ["clang++", "g++", "c++"]
    for candidate in candidates:
        path=shutil.which(candidate)
        if path:
            return path
    raise RuntimeError("No C++17 compiler found. Set CXX or install clang++/g++.")


def _native_arch_flag(cxx: str) -> list[str]:
    override=os.environ.get("REMATH_NATIVE_ARCH_FLAGS")
    if override is not None:
        return override.split() if override.strip() else []
    machine=platform.machine().lower()
    if sys.platform == "darwin" and machine in {"arm64", "aarch64"}:
        return ["-mcpu=native"]
    if os.name != "nt":
        return ["-march=native"]
    return []


def compile_shared(
    source: str | os.PathLike[str],
    output: str | os.PathLike[str],
    *,
    include_dirs: Iterable[str | os.PathLike[str]] = (),
    use_blas: bool = False,
    fast_math: bool = False,
) -> list[str]:
    source=str(source); output=str(output)
    cxx=find_cxx()
    cmd=[cxx, "-O3", "-std=c++17"]
    if fast_math:
        cmd.append("-ffast-math")
    cmd += _native_arch_flag(cxx)
    if os.name != "nt":
        cmd += ["-shared", "-fPIC"]
    for inc in include_dirs:
        cmd.append(f"-I{inc}")
    cmd += [source, "-o", output]
    if use_blas:
        if sys.platform == "darwin":
            cmd += ["-DACCELERATE_NEW_LAPACK", "-framework", "Accelerate"]
        elif os.name != "nt":
            cmd += ["-lblas"]
        else:
            raise RuntimeError("Native BLAS synthesis is not configured for Windows")
    subprocess.run(cmd, check=True)
    return cmd


def compiled_library_path(directory: str | os.PathLike[str], stem: str) -> str:
    return str(Path(directory) / f"{stem}{shared_library_suffix()}")
