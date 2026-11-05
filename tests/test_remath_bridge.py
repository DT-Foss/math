from wirkung_math.remath_bridge import classify_remath_formula,annotate_remath_trace


def test_remath_attention_maps_to_dynamic_bilinear():
    h=classify_remath_formula({"canonical_name":"Scaled Dot-Product Attention","domain":"attention"})
    assert h.family=="normalized_bilinear"
    assert h.rewrite=="dynamic_normalized_bilinear"
    assert h.consumer_sensitive


def test_remath_activation_switches_structure():
    h=classify_remath_formula({"canonical_name":"SILU Activation","domain":"activation"})
    assert "low_rank" in h.destroys
    assert "sparse_rows" in h.preserves


def test_remath_matrix_contraction_propagates_factors():
    h=classify_remath_formula({"canonical_name":"Matrix Contraction (GEMM/GEMV)","domain":"linear_algebra"})
    assert h.rewrite=="factor_propagation"


def test_trace_annotation_is_dependency_light():
    trace={"nodes":[
        {"id":"n1","name":"softmax","category":"mathematical_operation",
         "formula":{"canonical_name":"Numerically Stable Softmax","domain":"attention"}},
        {"id":"n2","name":"norm","category":"mathematical_operation",
         "formula":{"canonical_name":"Root Mean Square Normalization (RMSNorm)","domain":"normalization"}},
    ]}
    a=annotate_remath_trace(trace)
    assert a[0]["rewrite"]=="mass_gated_normalized_state"
    assert a[1]["rewrite"]=="structured_norm_rewrite"
