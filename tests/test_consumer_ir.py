import pytest

from wirkung_math.consumer_ir import (
    EvidenceLevel, ProgramNode, ConsumerProgram, CandidateRoute, CompilationPolicy
)


def test_consumer_program_slices_change_and_demand_intersection():
    p=ConsumerProgram(
        nodes=(
            ProgramNode("x","source"),
            ProgramNode("dead","linear",("x",)),
            ProgramNode("a","linear",("x",)),
            ProgramNode("b","elementwise",("a",)),
            ProgramNode("y","consumer",("b",)),
            ProgramNode("other","consumer",("dead",)),
        ),
        changed=frozenset({"x"}),
        requested=frozenset({"y"}),
    )
    assert p.required_nodes()==frozenset({"x","a","b","y"})


def test_consumer_program_rejects_missing_dependency():
    with pytest.raises(ValueError,match="missing dependency"):
        ConsumerProgram(
            nodes=(ProgramNode("a","linear",("missing",)),),
            changed=frozenset({"a"}),
            requested=frozenset({"a"}),
        )


def test_exact_policy_allows_only_proven_routes():
    proven=CandidateRoute("p","x","y",0,1,0,8,EvidenceLevel.PROVEN)
    validated=CandidateRoute("v","x","y",0,0.5,0,8,EvidenceLevel.VALIDATED)
    heuristic=CandidateRoute("h","x","y",0,0.1,0,8,EvidenceLevel.HEURISTIC)
    exact=CompilationPolicy()
    assert exact.filter((proven,validated,heuristic))==(proven,)
    relaxed=CompilationPolicy(exact_only=False,allow_validated=True)
    assert relaxed.filter((proven,validated,heuristic))==(proven,validated)


def test_quotient_compiler_is_public_runtime_api():
    import wirkung_math as wm
    assert wm.QuotientCompiler is not None
    assert wm.ObservableLinearQuotient is not None
    assert wm.discover_linear_graph_quotients is not None
    assert wm.discover_empirical_quotient is not None
    assert wm.LinearOrderCertificate is not None
    assert wm.FiniteStateConsumerQuotient is not None
    assert wm.consumer_program_from_remath_trace is not None
    assert wm.compile_linear_graph_quotient is not None
    assert wm.CompiledLinearGraphQuotient is not None
