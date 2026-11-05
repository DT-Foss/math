from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .lowrank import FactorDelta


@dataclass(frozen=True)
class ScaledBasePlusFactor:
    row_scale: np.ndarray
    base: np.ndarray
    factor: FactorDelta
    offset: np.ndarray | None = None

    def materialize(self) -> np.ndarray:
        y = self.row_scale[:, None] * self.base + self.factor.materialize()
        if self.offset is not None:
            y = y + self.offset
        return y


def rmsnorm(X: np.ndarray, gamma: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    rho = np.sqrt(np.mean(X * X, axis=1) + eps)
    return (X / rho[:, None]) * gamma


def rmsnorm_lowrank_update(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, gamma: np.ndarray, eps: float = 1e-6
) -> ScaledBasePlusFactor:
    """Exact structured RMSNorm(X + ABᵀ) rewrite."""
    d = X.shape[1]
    rho2 = np.mean(X * X, axis=1) + eps
    XB = X @ B
    gram = B.T @ B
    cross = 2.0 * np.sum(XB * A, axis=1) / d
    quad = np.einsum("ni,ij,nj->n", A, gram, A) / d
    rho = np.sqrt(rho2)
    rho_new = np.sqrt(rho2 + cross + quad)
    base = (X / rho[:, None]) * gamma
    left = A / rho_new[:, None]
    right = gamma[:, None] * B
    return ScaledBasePlusFactor(rho / rho_new, base, FactorDelta(left, right))


def layernorm(X: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    mu = X.mean(axis=1, keepdims=True)
    xc = X - mu
    sigma = np.sqrt(np.mean(xc * xc, axis=1) + eps)
    return (xc / sigma[:, None]) * gamma + beta


def layernorm_lowrank_update(
    X: np.ndarray, A: np.ndarray, B: np.ndarray, gamma: np.ndarray, beta: np.ndarray, eps: float = 1e-5
) -> ScaledBasePlusFactor:
    """Exact structured LayerNorm(X + ABᵀ) rewrite."""
    d = X.shape[1]
    mu = X.mean(axis=1, keepdims=True)
    xc = X - mu
    Bc = B - B.mean(axis=0, keepdims=True)
    sigma2 = np.mean(xc * xc, axis=1) + eps
    XBc = xc @ Bc
    gram = Bc.T @ Bc
    cross = 2.0 * np.sum(XBc * A, axis=1) / d
    quad = np.einsum("ni,ij,nj->n", A, gram, A) / d
    sigma = np.sqrt(sigma2)
    sigma_new = np.sqrt(sigma2 + cross + quad)
    centered_base = (xc / sigma[:, None]) * gamma
    left = A / sigma_new[:, None]
    right = gamma[:, None] * Bc
    return ScaledBasePlusFactor(sigma / sigma_new, centered_base, FactorDelta(left, right), offset=beta)
