import numpy as np

from wirkung_math.consumer_ir import (
    EvidenceLevel, CandidateRoute, CompilationPolicy, ProgramNode, ConsumerProgram
)
from wirkung_math.quotient_compiler import QuotientCompiler


def program():
    return ConsumerProgram(
        nodes=(
            ProgramNode("input","source"),
            ProgramNode("middle","linear",("input",)),
            ProgramNode("output","consumer",("middle",)),
        ),
        changed=frozenset({"input"}),
        requested=frozenset({"output"}),
    )


def test_exact_compiler_rejects_cheaper_validated_route():
    routes=(
        CandidateRoute("full","start","goal",0,10,0,100,EvidenceLevel.PROVEN),
        CandidateRoute("trace-fast","start","goal",0,1,0,10,EvidenceLevel.VALIDATED),
    )
    result=QuotientCompiler().compile(program(),routes,start="start",goal="goal",updates=1)
    assert result.selected_route_names==("full",)
    assert tuple(r.name for r in result.rejected)==("trace-fast",)


def test_relaxed_compiler_can_choose_validated_route():
    routes=(
        CandidateRoute("full","start","goal",0,10,0,100,EvidenceLevel.PROVEN),
        CandidateRoute("trace-fast","start","goal",0,1,0,10,EvidenceLevel.VALIDATED),
    )
    compiler=QuotientCompiler(CompilationPolicy(exact_only=False,allow_validated=True))
    result=compiler.compile(program(),routes,start="start",goal="goal",updates=1)
    assert result.selected_route_names==("trace-fast",)


def test_compiler_discovers_linear_consumer_quotient_and_removes_dense_intermediate():
    rng=np.random.default_rng(410)
    A=rng.normal(size=(64,128))
    # 32 requested outputs but only a 3-dimensional row space.
    U=rng.normal(size=(3,64))
    V=rng.normal(size=(32,3))
    C=V@U
    compiler=QuotientCompiler()
    result=compiler.compile_linear_chain(
        name="chain",program=program(),upstream=A,consumer=C,
        full_ms=4.0,quotient_ms=.3,updates=1,
    )
    assert result.selected_route_names==("chain:consumer_quotient",)
    selected=result.selected[0]
    assert selected.metadata["quotient_dimension"]==3
    assert selected.metadata["dense_intermediate_dimension"]==64
    assert selected.evidence==EvidenceLevel.PROVEN
