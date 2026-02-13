from cobble.expr import Const, Expr, Poly, Sum
from cobble.polynomial import Polynomial
from numpy.polynomial.chebyshev import cheb2poly
from scipy.special import jv


def hamiltonian_simulation(H: Expr, real_phase: bool = False):
    """

    Given a block encoding for a Hamiltonian H, this method approximates
    the time evolution of H using the Jacobi-Anger expansion for a hardcoded 
    evolution time of t=7. The expansion of e^(-iHt) is truncated after 7 terms.
    
    e^(-iHt) = J_0(t) 
               + 2 • ∑_{even k > 0}^{k=7} (-1)^(k/2)     • J_k(t) • T_k(H) 
               + 2i• ∑_{odd k > 0}^{k=7}  (-1)^((k-1)/2) • J_k(t) • T_k(H) 

    For more information see Eq 32 of
    https://quantum-journal.org/papers/q-2019-07-12-163/pdf/

    Args:
        H: A block encoding of the Hamiltonian we intend to simulate, given
            as an object of type ``Expr``
        real_phase: boolean flag 
    Returns:
        P: An expression of tpye Expr, which encodes a Polynomial of the input H that
        approximates e^(-iAt) 
    """
    def T_n(n: int, X: Expr):
        """
        Helper function that returns the n'th degree Chebyshev polynomial of the 
        first kind, for some block encoded matrix X. 

        The n-degree Chebyshev polynomial can be defined by the recurrence relation
            T_0(x) = 1
            T_1(x) = x
            T_{n+1}(x) = 2x•T_n(x) - T_{n-1}(x)
        
        Args:
            n: the degree of the desired Chebyshev polynomial 
            H: A block encoding of some matrix X 
        
        Returns:
            P: An object of type Polynomial encoding T_n(X)
        """
        return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0]).tolist()))

    n = 7
    t = 7.0
    cos = Const(jv(0, t)) + 2 * Sum(
        [((-1) ** k * jv(2 * k, t), T_n(2 * k, H)) for k in range(1, n + 1)]
    )
    sin = 2 * Sum(
        [((-1) ** k * jv(2 * k + 1, t), T_n(2 * k + 1, H)) for k in range(n + 1)]
    )
    P = cos + Const(1 if real_phase else 1j) * sin
    P = P.optimize()
    assert isinstance(P, Poly if real_phase else Sum)

    return ("hamiltonian-simulation", P)
