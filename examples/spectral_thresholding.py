import math

from cobble.expr import Const, Expr, Poly, Sum
from cobble.polynomial import Polynomial
from numpy.polynomial.chebyshev import cheb2poly


def spectral_thresholding(X: Expr):
    """Matrix sign function.

    Implements Ex. 5A of https://www.cfm.brown.edu/people/dobrush/am34/Mathematica/ch5/chebyshev.html
    and Eq. 53 of https://arxiv.org/pdf/2105.02859."""

    def T_n(n, X):
        return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0])))

    n = 10
    P = Const(4 / math.pi) * Sum(
        [(1.0 / (2 * i + 1), T_n(2 * i + 1, X)) for i in range(n)]
    )
    P = P.optimize()
    assert isinstance(P, Poly)

    return ("spectral-thresholding", P)
