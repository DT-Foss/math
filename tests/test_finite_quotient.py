import numpy as np

from wirkung_math.finite_quotient import FiniteStateConsumerQuotient


def test_finite_state_quotient_minimizes_redundant_code_states_exactly():
    # Four physical copies of each of three consumer states.
    classes=3
    copies=4
    n=classes*copies
    actions=2
    outputs=np.array([c for c in range(classes) for _ in range(copies)])
    T=np.empty((n,actions),dtype=np.int64)
    class_transition=np.array([[1,2],[2,0],[0,1]])
    for c in range(classes):
        for k in range(copies):
            s=c*copies+k
            # Different physical representatives inside the correct next class.
            for a in range(actions):
                nc=class_transition[c,a]
                T[s,a]=nc*copies+((k+a+1)%copies)
    q=FiniteStateConsumerQuotient.fit(T,outputs)
    assert q.class_count==3
    assert q.state_compression==4.0
    assert q.proof_passed
    for s in range(n):
        for a in range(actions):
            assert q.step_class(q.class_of_state[s],a)==q.class_of_state[T[s,a]]
        assert q.output_for_class(q.class_of_state[s])==outputs[s]


def test_same_current_output_is_split_when_future_consumer_behavior_differs():
    # States 0 and 1 emit the same current output, but action 0 reaches
    # different future-output classes. They must not be quotiented together.
    T=np.array([
        [2],
        [3],
        [2],
        [3],
    ])
    outputs=np.array([0,0,1,2])
    q=FiniteStateConsumerQuotient.fit(T,outputs)
    assert q.class_of_state[0] != q.class_of_state[1]
    assert q.class_count==4


def test_vector_consumer_signatures_are_supported():
    T=np.array([[1],[0],[3],[2]])
    outputs=np.array([[1,0],[1,0],[0,1],[0,1]])
    q=FiniteStateConsumerQuotient.fit(T,outputs)
    assert q.class_count==2
    assert q.proof_passed
