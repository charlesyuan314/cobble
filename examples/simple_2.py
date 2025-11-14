from cobble.expr import Const, Dagger, Expr, Poly
from cobble.polynomial import Polynomial


def simple_2(A: Expr):
    I = Const(1.0)

    prog = I - Dagger(A) * A
    targ = Poly(A, Polynomial([1.0, 0.0, -1.0]))
    return ("I - A^dagger * A", prog, targ)
