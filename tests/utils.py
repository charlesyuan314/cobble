import numpy as np
from scipy.linalg import sqrtm


def qsp_unitary_dilation(A: np.ndarray) -> np.ndarray:
    """
    Construct the canonical unitary dilation used in Quantum Signal Processing (QSP)
    for a Hermitian matrix A (‖A‖ ≤ 1).

    U = [[A, √(I - A²)],
         [√(I - A²), -A]]
    """
    if not np.allclose(A, A.conj().T, atol=1e-10):
        raise ValueError("Input matrix A must be Hermitian.")

    inner = np.eye(A.shape[0], dtype=complex) - A @ A
    sqrt_term = sqrtm(inner)
    return np.vstack([np.hstack([A, sqrt_term]), np.hstack([sqrt_term, -A])])  # type: ignore


def matrix_polyval(coeffs, A):
    """
    Evaluate polynomial with coefficients [a0, a1, ..., an] on matrix A,
    using matrix powers (not elementwise).
    """
    A = np.asarray(A)
    n = len(coeffs)
    result = np.zeros_like(A, dtype=A.dtype)
    # Start from highest degree (Horner's method)
    for a in reversed(coeffs):
        result = a * np.eye(A.shape[0], dtype=A.dtype) + A @ result
    return result
