from cobble.expr import Basic, Const, Expr


def penalized_coupler():
    """Physics-inspired Hamiltonian plus simple constraint penalty.

    Z₀ = Z⊗I,  Z₁ = I⊗Z,  ZZ = Z⊗Z,
    H_phys = 2·Z₀ + 2·Z₁ + 1.2·ZZ,
    H_pen  = -Z₀ - Z₁ + const,
    H_tot  = H_phys + H_pen

    Returns: (name, unoptimized_form, target_form) for this example.
    """

    kron = Expr.kron
    Z0 = kron(Basic("Z"), Const(1.0))
    Z1 = kron(Const(1.0), Basic("Z"))
    ZZ = kron(Basic("Z"), Basic("Z"))
    I = kron(Const(1.0), Const(1.0))
    H_tot = 2 * Z0 + 2 * Z1 + 1.2 * ZZ - (Z0 - Z1 + I)
    targ = 1.2 * ZZ + Z0 + 3 * Z1 - I
    return ("penalized-coupler", H_tot, targ)
