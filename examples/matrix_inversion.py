from cobble.expr import Const, Expr, Poly
from cobble.polynomial import Polynomial
from numpy.polynomial.chebyshev import cheb2poly


def matrix_inversion(X: Expr):
    """Matrix inversion.

    Implements Eq. 5 of https://arxiv.org/pdf/2507.15537."""

    n = 7
    a = 1.0 / 6.0

    def T_n(n: int, X: Expr):
        return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0]).tolist()))

    def L_n(n: int, X: Expr):
        return (T_n(n, X) + ((1 - a) / (1 + a)) * T_n(n - 1, X)) / Const(2**n - 1)

    P = Const(1.0) / X - L_n(
        n, (2 * X**2 - Const(1 + a**2)) / Const(1 - a**2)
    ) / (X * L_n(n, Const(-((1 + a**2)) / (1 - a**2))))
    P = P.optimize()
    assert isinstance(P, Poly)

    return ("matrix-inversion", P)
