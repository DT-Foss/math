from __future__ import annotations
import numpy as np

def sparse_requested_linear_delta(
    operator_block,
    requested_outputs: np.ndarray,
    changed_inputs: np.ndarray,
    delta_values: np.ndarray,
) -> np.ndarray:
    """
    Two-sided sparse linear response.

    For y=A x, changed input set U and requested output set K:
        delta y_K = A[K,U] delta x_U.

    operator_block(K,U) may generate, fetch or compute only the requested
    submatrix; A never needs to be materialized globally.
    """
    K=np.asarray(requested_outputs,dtype=np.int64)
    U=np.asarray(changed_inputs,dtype=np.int64)
    dx=np.asarray(delta_values)
    block=np.asarray(operator_block(K,U))
    if block.shape != (len(K),len(U)):
        raise ValueError("operator block shape mismatch")
    if dx.shape[0] != len(U):
        raise ValueError("delta length mismatch")
    return block@dx
