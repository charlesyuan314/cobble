from cobble.expr import Expr, Sum
from numpy.polynomial.chebyshev import cheb2poly


def T_n(n: int, X: Expr):
    coeffs = cheb2poly([0.0] * n + [1.0]).tolist()
    return (
        "chebyshev",
        Sum([(coeff, X**i) for i, coeff in enumerate(coeffs)]),
    )
