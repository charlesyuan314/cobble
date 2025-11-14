from cobble.expr import Expr, Poly
from cobble.polynomial import Polynomial


def regression_example(A: Expr, B: Expr):
    sum_ab = A - B
    f = (A - B) + 0.5 * (A - B) ** 2
    g = (A - B) - 0.5 * (A - B) ** 2
    prog = f * g
    targ = Poly(sum_ab, Polynomial([0.0, 0.0, 1.0, 0.0, -0.25]))
    return ("regression-example", prog, targ)
