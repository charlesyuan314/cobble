from cobble.expr import Expr, Poly
from cobble.polynomial import Polynomial


def laplacian_filter(Hx: Expr, Hy: Expr):
    kron = Expr.kron

    f = [1.0, 0.5]  # 1 + 0.5x
    g = [-2.9, 3.4]
    p = [1.1, 0.4]
    q = [-5.8, -0.3]
    term_fp = kron(Poly(Hx, Polynomial(f)), Poly(Hy, Polynomial(p)))
    term_fq = kron(Poly(Hx, Polynomial(f)), Poly(Hy, Polynomial(q)))
    term_gp = kron(Poly(Hx, Polynomial(g)), Poly(Hy, Polynomial(p)))
    term_gq = kron(Poly(Hx, Polynomial(g)), Poly(Hy, Polynomial(q)))
    prog = term_fp + term_gp + term_fq + term_gq
    targ = kron(Poly(Hx, Polynomial([-1.9, 3.9])), Poly(Hy, Polynomial([-4.7, 0.1])))
    return ("laplacian-filter", prog, targ)
