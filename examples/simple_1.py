from cobble.expr import Expr


def simple_1(A: Expr):
    prog = 0.5 * A + 0.5 * A
    targ = A
    return ("(A + A) * 0.5", prog, targ)
