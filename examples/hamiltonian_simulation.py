from cobble.expr import Const, Expr, Poly, Sum
from cobble.polynomial import Polynomial
from numpy.polynomial.chebyshev import cheb2poly
from scipy.special import jv


def hamiltonian_simulation(X: Expr, real_phase: bool = False):
    def T_n(n: int, X: Expr):
        return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0]).tolist()))

    n = 7
    t = 7.0
    cos = Const(jv(0, t)) + 2 * Sum(
        [((-1) ** k * jv(2 * k, t), T_n(2 * k, X)) for k in range(1, n + 1)]
    )
    sin = 2 * Sum(
        [((-1) ** k * jv(2 * k + 1, t), T_n(2 * k + 1, X)) for k in range(n + 1)]
    )
    P = cos + Const(1 if real_phase else 1j) * sin
    P = P.optimize()
    assert isinstance(P, Poly if real_phase else Sum)

    return ("hamiltonian-simulation", P)
