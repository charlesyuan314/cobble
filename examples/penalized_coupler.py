from cobble.expr import Basic, Const, Expr


def penalized_coupler():
    kron = Expr.kron
    Z0 = kron(Basic("Z"), Const(1.0))
    Z1 = kron(Const(1.0), Basic("Z"))
    ZZ = kron(Basic("Z"), Basic("Z"))
    I = kron(Const(1.0), Const(1.0))
    H_tot = 2 * Z0 + 2 * Z1 + 1.2 * ZZ - (Z0 - Z1 + I)
    targ = 1.2 * ZZ + Z0 + 3 * Z1 - I
    return ("penalized-coupler", H_tot, targ)
