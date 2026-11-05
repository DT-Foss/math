from wirkung_math.delta_ir import DeltaDescriptor,Structure


def test_representation_switching():
    d=DeltaDescriptor((1000,64),frozenset({Structure.SPARSE_ROWS,Structure.LOW_RANK}),changed_rows=8,rank_bound=4)
    after=d.through_elementwise()
    assert Structure.SPARSE_ROWS in after.structures
    assert Structure.LOW_RANK not in after.structures
    pair=d.through_pairwise_symmetric(8)
    assert pair.rank_bound==16
    assert Structure.ROW_COLUMN in pair.structures
