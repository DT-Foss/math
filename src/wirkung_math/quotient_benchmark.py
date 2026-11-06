from __future__ import annotations
from dataclasses import dataclass
import statistics
import time
import numpy as np
from .linear_quotient import LinearConsumerQuotient,ObservableLinearQuotient
from .finite_quotient import FiniteStateConsumerQuotient


@dataclass(frozen=True)
class ObservableBenchmarkCase:
    A:np.ndarray
    B:np.ndarray
    C:np.ndarray
    x0:np.ndarray
    inputs:np.ndarray
    planted_dimension:int


@dataclass(frozen=True)
class LinearChainBenchmarkCase:
    upstream:np.ndarray
    consumer:np.ndarray
    inputs:np.ndarray
    planted_dimension:int


def make_observable_case(
    *,state_dim:int,observable_dim:int,input_dim:int,steps:int,seed:int=0
)->ObservableBenchmarkCase:
    if not (0<observable_dim<=state_dim):
        raise ValueError("invalid observable dimension")
    if input_dim<=0 or steps<=0:
        raise ValueError("input_dim and steps must be positive")
    rng=np.random.default_rng(seed)
    Q,_=np.linalg.qr(rng.normal(size=(state_dim,observable_dim)))
    # A stable cyclic observable block. e0^T A^k visits every coordinate.
    P=np.zeros((observable_dim,observable_dim),dtype=np.float64)
    for i in range(observable_dim):
        P[i,(i+1)%observable_dim]=1.0
    Aobs=.91*P
    outside=.37
    A=outside*np.eye(state_dim)+Q@(Aobs-outside*np.eye(observable_dim))@Q.T
    B=rng.normal(scale=.2,size=(state_dim,input_dim))
    C=np.zeros((1,observable_dim),dtype=np.float64)
    C[0,0]=1.0
    C=C@Q.T
    x0=rng.normal(size=state_dim)
    inputs=rng.normal(scale=.1,size=(steps,input_dim))
    return ObservableBenchmarkCase(A,B,C,x0,inputs,observable_dim)


def replay_observable_case(case:ObservableBenchmarkCase)->dict:
    q=ObservableLinearQuotient.fit(case.A,case.B,case.C)
    x=case.x0.copy()
    z=q.project_state(x)
    max_error=0.0
    for u in case.inputs:
        full=case.C@x
        reduced=q.output(z)
        max_error=max(max_error,float(np.max(np.abs(full-reduced))))
        x=case.A@x+case.B@u
        z=q.step(z,u)
    full=case.C@x
    reduced=q.output(z)
    max_error=max(max_error,float(np.max(np.abs(full-reduced))))
    return {
        "full_dimension":int(case.A.shape[0]),
        "discovered_dimension":q.dimension,
        "planted_dimension":int(case.planted_dimension),
        "max_abs_error":max_error,
        "invariance_residual":q.invariance_residual,
        "output_residual":q.output_residual,
        "proof_passed":q.is_exact,
        "state_compression":(
            float(case.A.shape[0]/q.dimension)
            if q.dimension else float("inf")
        ),
    }


def make_linear_chain_case(
    *,input_dim:int,intermediate_dim:int,output_dim:int,
    quotient_dim:int,calls:int,seed:int=0,
)->LinearChainBenchmarkCase:
    if not (0<quotient_dim<=min(intermediate_dim,output_dim)):
        raise ValueError("invalid quotient dimension")
    rng=np.random.default_rng(seed)
    upstream=rng.normal(
        scale=1/np.sqrt(input_dim),size=(intermediate_dim,input_dim)
    )
    Q,_=np.linalg.qr(rng.normal(size=(intermediate_dim,quotient_dim)))
    V=rng.normal(size=(output_dim,quotient_dim))
    consumer=V@Q.T
    inputs=rng.normal(size=(calls,input_dim))
    return LinearChainBenchmarkCase(
        upstream,consumer,inputs,quotient_dim
    )


def replay_linear_chain_case(case:LinearChainBenchmarkCase)->dict:
    q=LinearConsumerQuotient.fit(case.consumer)
    composed=q.compose_upstream(case.upstream)
    full=case.inputs@case.upstream.T@case.consumer.T
    reduced=(case.inputs@composed.T)@q.reduced_consumer.T
    error=float(np.max(np.abs(full-reduced))) if full.size else 0.0
    full_multiplies=(
        case.upstream.size+case.consumer.size
    )
    reduced_multiplies=(
        composed.size+q.reduced_consumer.size
    )
    return {
        "input_dimension":int(case.upstream.shape[1]),
        "dense_intermediate_dimension":int(case.upstream.shape[0]),
        "output_dimension":int(case.consumer.shape[0]),
        "discovered_dimension":q.dimension,
        "planted_dimension":int(case.planted_dimension),
        "max_abs_error":error,
        "reconstruction_residual":q.reconstruction_residual,
        "full_multiplies_per_call":int(full_multiplies),
        "reduced_multiplies_per_call":int(reduced_multiplies),
        "multiply_reduction":float(
            full_multiplies/max(1,reduced_multiplies)
        ),
    }


def _median_ms(fn,repeats:int)->float:
    samples=[]
    for _ in range(repeats):
        start=time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns()-start)/1e6)
    return float(statistics.median(samples))


def benchmark_observable_case(
    case:ObservableBenchmarkCase,*,repeats:int=5
)->dict:
    correctness=replay_observable_case(case)
    q=ObservableLinearQuotient.fit(case.A,case.B,case.C)

    def full_run():
        x=case.x0.copy()
        checksum=0.0
        for u in case.inputs:
            x=case.A@x+case.B@u
            checksum+=float((case.C@x)[0])
        return checksum

    def reduced_run():
        z=q.project_state(case.x0)
        checksum=0.0
        for u in case.inputs:
            z=q.step(z,u)
            checksum+=float(q.output(z)[0])
        return checksum

    # Warm both BLAS paths before timing.
    full_run(); reduced_run()
    full_ms=_median_ms(full_run,repeats)
    reduced_ms=_median_ms(reduced_run,repeats)
    return {
        **correctness,
        "steps":int(len(case.inputs)),
        "full_ms":full_ms,
        "quotient_ms":reduced_ms,
        "speedup":full_ms/reduced_ms,
    }


def benchmark_linear_chain_case(
    case:LinearChainBenchmarkCase,*,repeats:int=7
)->dict:
    correctness=replay_linear_chain_case(case)
    q=LinearConsumerQuotient.fit(case.consumer)
    composed=q.compose_upstream(case.upstream)

    def full_run():
        return case.inputs@case.upstream.T@case.consumer.T

    def reduced_run():
        return (
            case.inputs@composed.T
        )@q.reduced_consumer.T

    full_run(); reduced_run()
    full_ms=_median_ms(full_run,repeats)
    reduced_ms=_median_ms(reduced_run,repeats)
    return {
        **correctness,
        "calls":int(len(case.inputs)),
        "full_ms":full_ms,
        "quotient_ms":reduced_ms,
        "speedup":full_ms/reduced_ms,
    }


@dataclass(frozen=True)
class FiniteStateBenchmarkCase:
    transitions:np.ndarray
    outputs:np.ndarray
    initial_state:int
    actions:np.ndarray
    planted_classes:int


def make_finite_state_case(
    *,physical_states:int,behavior_classes:int,actions:int,
    steps:int,seed:int=0,
)->FiniteStateBenchmarkCase:
    if physical_states<=0 or behavior_classes<=0 or actions<=0 or steps<=0:
        raise ValueError("finite-state dimensions must be positive")
    if physical_states%behavior_classes:
        raise ValueError("physical_states must be divisible by behavior_classes")
    copies=physical_states//behavior_classes
    rng=np.random.default_rng(seed)
    class_transition=np.empty((behavior_classes,actions),dtype=np.int64)
    for a in range(actions):
        # Each action is a nonzero rotation of one hidden behavior cycle.
        class_transition[:,a]=(
            np.arange(behavior_classes)+(a+1)
        )%behavior_classes
    # Current output is deliberately much coarser than the true quotient.
    # Future behavior under the cycle separates all hidden classes.
    class_output=(np.arange(behavior_classes)==0).astype(np.int64)
    T=np.empty((physical_states,actions),dtype=np.int64)
    outputs=np.empty(physical_states,dtype=np.int64)
    for cls in range(behavior_classes):
        for copy in range(copies):
            state=cls*copies+copy
            outputs[state]=class_output[cls]
            for a in range(actions):
                nxt=int(class_transition[cls,a])
                # Physical copies evolve differently while remaining in the
                # same future consumer-equivalence class.
                nxt_copy=(copy*(a+2)+a+1)%copies
                T[state,a]=nxt*copies+nxt_copy
    action_trace=rng.integers(0,actions,size=steps,dtype=np.int64)
    initial=int(rng.integers(0,physical_states))
    return FiniteStateBenchmarkCase(
        T,outputs,initial,action_trace,int(behavior_classes)
    )


def replay_finite_state_case(case:FiniteStateBenchmarkCase)->dict:
    q=FiniteStateConsumerQuotient.fit(case.transitions,case.outputs)
    physical=int(case.initial_state)
    reduced=int(q.class_of_state[physical])
    equal=True
    for action0 in case.actions:
        action=int(action0)
        full_output=case.outputs[physical]
        reduced_output=q.output_for_class(reduced)
        if _freeze_benchmark_value(full_output)!=_freeze_benchmark_value(reduced_output):
            equal=False
            break
        physical=int(case.transitions[physical,action])
        reduced=q.step_class(reduced,action)
    if equal:
        equal=(
            _freeze_benchmark_value(case.outputs[physical])
            ==_freeze_benchmark_value(q.output_for_class(reduced))
        )
    full_bytes=int(case.transitions.nbytes+case.outputs.nbytes)
    reduced_bytes=int(
        q.reduced_transition.nbytes
        +q.representative_state.nbytes
        +q.class_of_state[:q.class_count].nbytes
    )
    return {
        "physical_states":q.state_count,
        "discovered_classes":q.class_count,
        "planted_classes":int(case.planted_classes),
        "state_compression":q.state_compression,
        "refinement_rounds":q.refinement_rounds,
        "proof_passed":bool(q.proof_passed),
        "trajectory_equal":bool(equal),
        "full_table_bytes":full_bytes,
        "reduced_table_bytes":reduced_bytes,
        "table_byte_reduction":float(full_bytes/max(1,reduced_bytes)),
        "steps":int(len(case.actions)),
    }


def _freeze_benchmark_value(value):
    if isinstance(value,np.ndarray):
        return tuple(_freeze_benchmark_value(x) for x in value.tolist())
    if isinstance(value,(list,tuple)):
        return tuple(_freeze_benchmark_value(x) for x in value)
    if isinstance(value,np.generic):
        return value.item()
    return value


def benchmark_finite_state_case(
    case:FiniteStateBenchmarkCase,*,repeats:int=5
)->dict:
    correctness=replay_finite_state_case(case)
    q=FiniteStateConsumerQuotient.fit(case.transitions,case.outputs)

    def full_run():
        state=int(case.initial_state)
        checksum=0
        for action0 in case.actions:
            checksum+=int(case.outputs[state])
            state=int(case.transitions[state,int(action0)])
        checksum+=int(case.outputs[state])
        return checksum

    reduced_transition=q.reduced_transition
    reduced_outputs=np.asarray(q.reduced_outputs)

    def reduced_run():
        state=int(q.class_of_state[int(case.initial_state)])
        checksum=0
        for action0 in case.actions:
            checksum+=int(reduced_outputs[state])
            state=int(reduced_transition[state,int(action0)])
        checksum+=int(reduced_outputs[state])
        return checksum

    full_run(); reduced_run()
    full_ms=_median_ms(full_run,repeats)
    reduced_ms=_median_ms(reduced_run,repeats)
    return {
        **correctness,
        "full_ms":full_ms,
        "quotient_ms":reduced_ms,
        "speedup":full_ms/reduced_ms,
    }
