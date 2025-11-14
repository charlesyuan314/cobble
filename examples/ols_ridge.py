from cobble.expr import Const, Dagger, Expr, Poly, Sum
from cobble.polynomial import Polynomial


def ols_ridge(X: Expr):
    def build_p(d: int, lam: float, mu: float) -> Polynomial:
        n = d + 1
        # T1(x) = (1 - (1 - x/mu)^{n})/(mu x)
        P = [0.0] * (n + 1)
        c = 1.0
        for k in range(0, n + 1):
            P[k] = c * ((-1.0 / mu) ** k)
            if k < n:
                c *= (n - k) / (k + 1)
        A = Polynomial([1.0]).add(Polynomial(P).scale(-1.0)).coeffs
        T1 = [(1.0 / mu) * A[k + 1] for k in range(0, len(A) - 1)]
        # T2(x) = (1 - (-(x/lam))^{n})/(1 + x/lam)
        N = [1.0] + [0.0] * (n - 1) + [-((1.0 / lam) ** n)]
        Q = [N[0]] + [0.0] * (n - 1)
        a = 1.0 / lam
        for k in range(1, n):
            Q[k] = N[k] - a * Q[k - 1]
        return Polynomial(T1).add(Polynomial(Q))

    d = 8
    theta = 0.5
    lam = 0.5
    mu = 2.0
    I = Const(1.0)
    H = Dagger(X) * X
    prog = theta * X * 1 / mu * Sum.of(
        (I - H / mu) ** k for k in range(d + 1)
    ) * Dagger(X) + (1 - theta) * X * 1 / lam * Sum.of(
        (-H / lam) ** k for k in range(d + 1)
    ) * Dagger(
        X
    )
    targ = X * Poly(H, build_p(d, lam, mu)) * Dagger(X)
    return ("ols-ridge", prog, targ)
