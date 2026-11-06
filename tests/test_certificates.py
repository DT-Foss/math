import numpy as np

from wirkung_math.certificates import (
    LinearOrderCertificate, QuantizationCellCertificate, ThresholdCertificate
)


def test_linear_argmax_certificate_proves_small_delta_and_rejects_large_delta():
    W=np.array([[2.,0.],[0.,2.],[-1.,-1.]])
    x=np.array([2.,0.])
    logits=W@x
    cert=LinearOrderCertificate(W)
    assert cert.certify_argmax(logits,0,np.array([.01,.01]))
    assert not cert.certify_argmax(logits,0,np.array([-2.,2.]))


def test_linear_topk_certificate_preserves_selected_set():
    W=np.array([[3.,0.],[2.,0.],[0.,2.],[-1.,0.]])
    x=np.array([2.,0.])
    logits=W@x
    cert=LinearOrderCertificate(W)
    selected=np.array([0,1])
    assert cert.certify_topk(logits,selected,np.array([.01,.01]))
    assert not cert.certify_topk(logits,selected,np.array([-2.,2.]))


def test_quantization_cell_certificate_uses_distance_to_boundary():
    cert=QuantizationCellCertificate(np.array([1.0,2.0]))
    value=np.array([2.1,3.8])
    assert cert.stable(value,np.array([.2,.5]))
    assert not cert.stable(value,np.array([.45,.5]))


def test_threshold_certificate_is_symmetric():
    cert=ThresholdCertificate(0.0)
    assert cert.stable(2.0,.5)
    assert cert.stable(-2.0,.5)
    assert not cert.stable(.2,.3)
