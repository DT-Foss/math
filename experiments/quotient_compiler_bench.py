#!/usr/bin/env python3
from __future__ import annotations
import json,os,time
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from wirkung_math.quotient_benchmark import (
    make_observable_case,benchmark_observable_case,
    make_linear_chain_case,benchmark_linear_chain_case,
)


def main():
    start=time.perf_counter()
    observable=make_observable_case(
        state_dim=4096,observable_dim=8,input_dim=4,
        steps=64,seed=20261106,
    )
    observable_build=time.perf_counter()-start
    start=time.perf_counter()
    obs=benchmark_observable_case(observable,repeats=3)
    obs["case_build_seconds"]=observable_build
    obs["discovery_and_benchmark_seconds"]=time.perf_counter()-start

    chain=make_linear_chain_case(
        input_dim=1024,intermediate_dim=4096,output_dim=256,
        quotient_dim=8,calls=128,seed=20261107,
    )
    start=time.perf_counter()
    linear=benchmark_linear_chain_case(chain,repeats=5)
    linear["discovery_and_benchmark_seconds"]=time.perf_counter()-start

    result={
        "schema":"consumer-relative-quotient-benchmark/v1",
        "source":"David Tom Foss automated Quotient Compiler benchmark",
        "observable_state":obs,
        "linear_consumer":linear,
    }
    print("RESULT "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
