from cobble.expr import Expr


def simulation_example(X: Expr, Y: Expr):
    kron = Expr.kron

    A = kron(X, X) + kron(Y, Y)
    B = kron(X, X) - kron(Y, Y)
    H = 1.0 * A + 0.3 * B
    targ = 1.3 * kron(X, X) + 0.7 * kron(Y, Y)
    return ("simulation-example", H, targ)
