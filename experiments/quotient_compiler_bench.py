#!/usr/bin/env python3
from __future__ import annotations
import json,os,time,statistics
import numpy as np
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from wirkung_math.quotient_benchmark import (
    make_observable_case,benchmark_observable_case,
    make_linear_chain_case,benchmark_linear_chain_case,
    make_finite_state_case,benchmark_finite_state_case,
)
from wirkung_math.consumer_ir import ProgramNode,ConsumerProgram
from wirkung_math.linear_graph_quotient import compile_linear_graph_quotient


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

    finite=make_finite_state_case(
        physical_states=65536,behavior_classes=16,actions=4,
        steps=100000,seed=20261108,
    )
    start=time.perf_counter()
    discrete=benchmark_finite_state_case(finite,repeats=3)
    discrete["discovery_and_benchmark_seconds"]=time.perf_counter()-start

    rng=np.random.default_rng(20261109)
    input_dim,hidden_dim,out_dim,joint_rank,batch=1024,4096,256,12,128
    A=rng.normal(scale=1/np.sqrt(input_dim),size=(hidden_dim,input_dim))
    Q,_=np.linalg.qr(rng.normal(size=(hidden_dim,joint_rank)))
    C1=rng.normal(size=(out_dim,joint_rank))@Q.T
    C2=rng.normal(size=(out_dim,joint_rank))@Q.T
    program=ConsumerProgram(
        (
            ProgramNode("x","source",metadata={"dimension":input_dim}),
            ProgramNode("h","linear",("x",),{"matrix":A}),
            ProgramNode("y1","linear",("h",),{"matrix":C1}),
            ProgramNode("y2","linear",("h",),{"matrix":C2}),
        ),
        frozenset({"x"}),frozenset({"y1","y2"}),
    )
    start=time.perf_counter()
    compiled=compile_linear_graph_quotient(program)
    compile_seconds=time.perf_counter()-start
    X=rng.normal(size=(batch,input_dim))
    def full_dag():
        H=X@A.T
        return H@C1.T,H@C2.T
    def reduced_dag():
        return compiled.run_batch({"x":X})
    full_ref=full_dag();reduced_ref=reduced_dag()
    dag_error=max(
        float(np.max(np.abs(full_ref[0]-reduced_ref["y1"]))),
        float(np.max(np.abs(full_ref[1]-reduced_ref["y2"]))),
    )
    def median_ms(fn,repeats=5):
        samples=[]
        for _ in range(repeats):
            st=time.perf_counter_ns();fn()
            samples.append((time.perf_counter_ns()-st)/1e6)
        return float(statistics.median(samples))
    full_dag();reduced_dag()
    full_dag_ms=median_ms(full_dag)
    reduced_dag_ms=median_ms(reduced_dag)
    branching={
        "input_dimension":input_dim,
        "dense_hidden_dimension":hidden_dim,
        "discovered_hidden_dimension":compiled.node_dimensions["h"],
        "requested_consumers":2,
        "consumer_output_dimension_each":out_dim,
        "batch":batch,
        "compile_seconds":compile_seconds,
        "full_ms":full_dag_ms,
        "quotient_ms":reduced_dag_ms,
        "speedup":full_dag_ms/reduced_dag_ms,
        "max_abs_error":dag_error,
    }

    result={
        "schema":"consumer-relative-quotient-benchmark/v1",
        "source":"David Tom Foss automated Quotient Compiler benchmark",
        "observable_state":obs,
        "linear_consumer":linear,
        "finite_consumer_state":discrete,
        "branching_linear_dag":branching,
    }
    print("RESULT "+json.dumps(result,sort_keys=True))


if __name__=="__main__":
    main()
