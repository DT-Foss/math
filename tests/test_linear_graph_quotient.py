import numpy as np

from wirkung_math.consumer_ir import ProgramNode,ConsumerProgram
from wirkung_math.linear_graph_quotient import discover_linear_graph_quotients


def test_backward_quotient_propagation_finds_minimal_chain_states():
    rng=np.random.default_rng(420)
    M1=rng.normal(size=(64,100))
    U=rng.normal(size=(2,64));V=rng.normal(size=(3,2));M2=V@U
    p=ConsumerProgram(
        nodes=(
            ProgramNode("x","source",metadata={"dimension":100}),
            ProgramNode("h","linear",("x",),{"matrix":M1}),
            ProgramNode("y","linear",("h",),{"matrix":M2}),
        ),
        changed=frozenset({"x"}),requested=frozenset({"y"}),
    )
    q={x.node:x for x in discover_linear_graph_quotients(p)}
    assert q["y"].required_dimension==3
    assert q["h"].required_dimension==2
    assert q["x"].required_dimension<=2


def test_branching_consumers_stack_requirements_and_requested_subset_changes_quotient():
    rng=np.random.default_rng(421)
    A=rng.normal(size=(8,20))
    C1=rng.normal(size=(2,8))
    # Make C2 add exactly two independent directions beyond C1.
    extra=rng.normal(size=(2,8))
    C2=np.vstack([C1[:1],extra])
    nodes=(
        ProgramNode("x","source",metadata={"dimension":20}),
        ProgramNode("h","linear",("x",),{"matrix":A}),
        ProgramNode("y1","linear",("h",),{"matrix":C1}),
        ProgramNode("y2","linear",("h",),{"matrix":C2}),
    )
    both=ConsumerProgram(nodes, frozenset({"x"}), frozenset({"y1","y2"}))
    one=ConsumerProgram(nodes, frozenset({"x"}), frozenset({"y1"}))
    qb={x.node:x for x in discover_linear_graph_quotients(both)}
    qo={x.node:x for x in discover_linear_graph_quotients(one)}
    assert qb["h"].required_dimension==4
    assert qo["h"].required_dimension==2
    assert qb["h"].required_dimension > qo["h"].required_dimension


def test_discovered_requirement_reconstructs_all_requested_outputs():
    rng=np.random.default_rng(422)
    A=rng.normal(size=(10,16))
    C1=rng.normal(size=(2,10));C2=rng.normal(size=(3,10))
    p=ConsumerProgram(
        (
            ProgramNode("x","source",metadata={"dimension":16}),
            ProgramNode("h","linear",("x",),{"matrix":A}),
            ProgramNode("y1","linear",("h",),{"matrix":C1}),
            ProgramNode("y2","linear",("h",),{"matrix":C2}),
        ),
        frozenset({"x"}),frozenset({"y1","y2"}),
    )
    q={x.node:x for x in discover_linear_graph_quotients(p)}
    x=rng.normal(size=16);h=A@x
    z=q["h"].basis.T@h
    reconstructed=q["h"].reduced_consumer@z
    expected=np.concatenate([C1@h,C2@h])
    np.testing.assert_allclose(reconstructed,expected,rtol=1e-11,atol=1e-11)
