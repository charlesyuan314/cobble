import math

from cobble.expr import (
    Basic,
    Condition,
    Const,
    Dagger,
    Division,
    Expr,
    If,
    Poly,
    Prod,
    QSP_TOL,
    Sum,
    Tensor,
    UnresolvedDivisionError,
)
from cobble.polynomial import Polynomial, TOL
from cobble.qtype import BitType, TensorType, TypeCheckError
import pytest


class TestBase:
    """Test Base matrix constructor."""

    def test_base_subnormalization_default(self):
        """Base should have default subnormalization=1.0."""
        A = Basic("A")
        assert A.subnormalization() == 1.0

    def test_base_subnormalization_custom(self):
        """Base with custom subnormalization value."""
        A = Basic("A", subnormalization_=2.5)
        assert A.subnormalization() == 2.5

    def test_base_queries(self):
        """Base should have queries=1."""
        A = Basic("A")
        assert A.queries() == 1

    def test_base_ancilla_qubits_default(self):
        """Base should have default ancilla_qubits=0."""
        A = Basic("A")
        assert A.ancilla_qubits() == 0

    def test_base_ancilla_qubits_custom(self):
        """Base with custom ancilla count."""
        A = Basic("A", ancilla_qubits_=3)
        assert A.ancilla_qubits() == 3

    def test_base_optimize(self):
        """Base should normalize to itself."""
        A = Basic("A")
        assert A.optimize() == A

    def test_base_hermitian(self):
        """Base Hermitian property."""
        A = Basic("A", hermitian=True)
        B = Basic("B", hermitian=False, hermitian_block_encoding=False)
        assert A.is_hermitian() is True
        assert A.is_hermitian_block_encoding() is True
        assert B.is_hermitian() is False
        assert B.is_hermitian_block_encoding() is False
        with pytest.raises(ValueError):
            _ = Basic("C", hermitian=False, hermitian_block_encoding=True)
        C = Basic("C", hermitian=True, hermitian_block_encoding=False)
        assert C.is_hermitian_block_encoding() is False


class TestConst:
    """Test constant (identity scalar) constructor."""

    def test_const_subnormalization_positive(self):
        """Const subnormalization should be absolute value."""
        C = Const(3.5)
        assert C.subnormalization() == 3.5

    def test_const_subnormalization_complex(self):
        """Const subnormalization should be absolute value of complex."""
        C = Const(3.5 + 1.0j)
        assert C.subnormalization() == abs(3.5 + 1.0j)

    def test_const_subnormalization_negative(self):
        """Const subnormalization should be absolute value of negative."""
        C = Const(-2.0)
        assert C.subnormalization() == 2.0

    def test_const_subnormalization_zero(self):
        """Zero constant has subnormalization=0."""
        C = Const(0.0)
        assert C.subnormalization() == 0.0

    def test_const_queries(self):
        """Const should have queries=0 (no oracle calls)."""
        C = Const(5.0)
        assert C.queries() == 0

    def test_const_ancilla_qubits(self):
        """Const should have ancilla_qubits=0."""
        C = Const(5.0)
        assert C.ancilla_qubits() == 0

    def test_const_optimize(self):
        """Const should normalize to itself."""
        C = Const(3.0)
        assert C.optimize() == C

    def test_const_hermitian(self):
        """Const is always Hermitian."""
        C = Const(3.0)
        assert C.is_hermitian() is True


class TestDagger:
    """Test adjoint (dagger) constructor."""

    def test_dagger_subnormalization(self):
        """Dagger preserves subnormalization."""
        A = Basic("A", subnormalization_=2.0)
        D = Dagger(A)
        assert D.subnormalization() == 2.0

    def test_dagger_queries(self):
        """Dagger preserves queries."""
        A = Basic("A")
        D = Dagger(A)
        assert D.queries() == 1

    def test_dagger_ancilla_qubits(self):
        """Dagger preserves ancilla_qubits."""
        A = Basic("A", ancilla_qubits_=2)
        D = Dagger(A)
        assert D.ancilla_qubits() == 2

    def test_dagger_ancilla_qubits_edge_cases(self):
        """Test ancilla_qubits() for Dagger edge cases."""
        A = Basic("A", ancilla_qubits_=0)
        D1 = Dagger(A)
        assert D1.ancilla_qubits() == 0

        B = Basic("B", ancilla_qubits_=3)
        S = Sum([(1.0, A), (1.0, B)])
        D2 = Dagger(S)
        assert D2.ancilla_qubits() == max(0, 3) + math.ceil(math.log2(2))

        C = Basic("C", ancilla_qubits_=5)
        P = Prod([B, C])
        D3 = Dagger(P)
        assert D3.ancilla_qubits() == max(3, 5) + math.ceil(math.log2(2))

        T = Tensor([A, B])
        D4 = Dagger(T)
        assert D4.ancilla_qubits() == 0 + 3

    def test_dagger_normalize_hermitian_base(self):
        """Dagger of Hermitian base normalizes to base."""
        A = Basic("A", hermitian=True)
        D = Dagger(A).optimize()
        assert D == A

    def test_dagger_normalize_const(self):
        """Dagger of const normalizes to const."""
        C = Const(3.0)
        D = Dagger(C).optimize()
        assert D == C

    def test_dagger_normalize_double(self):
        """Double dagger normalizes to original."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        D = Dagger(Dagger(A)).optimize()
        assert D == A

    def test_dagger_normalize_sum(self):
        """Dagger distributes over sum."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        B = Basic("B", hermitian=False, hermitian_block_encoding=False)
        S = Sum([(1.0, A), (2.0, B)])
        D = Dagger(S).optimize()
        expected = Sum([(1.0, Dagger(A)), (2.0, Dagger(B))]).optimize()
        assert Expr.structural_eq(D, expected)

    def test_dagger_normalize_prod(self):
        """Dagger reverses product."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        B = Basic("B", hermitian=False, hermitian_block_encoding=False)
        P = Prod([A, B])
        D = Dagger(P).optimize()
        expected = Prod([Dagger(B), Dagger(A)]).optimize()
        assert Expr.structural_eq(D, expected)

    def test_dagger_normalize_tensor(self):
        """Dagger distributes over tensor."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        B = Basic("B", hermitian=False, hermitian_block_encoding=False)
        T = Tensor([A, B])
        D = Dagger(T).optimize()
        expected = Tensor([Dagger(A), Dagger(B)]).optimize()
        assert Expr.structural_eq(D, expected)

    def test_dagger_normalize_poly_hermitian(self):
        """Dagger of poly with Hermitian expr normalizes to poly."""
        A = Basic("A", hermitian=True)
        P = Poly(A, Polynomial([1.0, 2.0]))
        D = Dagger(P).optimize()
        assert D == P


class TestSum:
    """Test linear combination (sum) constructor."""

    def test_sum_subnormalization_simple(self):
        """Sum subnormalization = sum of |coeffs| * term subnormalizations."""
        A = Basic("A", subnormalization_=1.0)
        B = Basic("B", subnormalization_=2.0)
        S = Sum([(1.0, A), (1.0, B)])
        assert S.subnormalization() == abs(1.0) * 1.0 + abs(1.0) * 2.0

    def test_sum_subnormalization_negative_coeff(self):
        """Sum subnormalization uses absolute value of coefficients."""
        A = Basic("A", subnormalization_=1.0)
        S = Sum([(1.0, A), (-1.0, A)])
        assert S.subnormalization() == abs(1.0) * 1.0 + abs(-1.0) * 1.0

    def test_sum_queries_simple(self):
        """Sum queries = sum of term queries."""
        A = Basic("A")
        B = Basic("B")
        S = Sum([(1.0, A), (1.0, B)])
        assert S.queries() == 1 + 1

    def test_sum_ancilla_qubits_simple(self):
        """Sum n = max(term n) + log2(num_terms)."""
        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=2)
        S = Sum([(1.0, A), (1.0, B)])
        assert S.ancilla_qubits() == max(1, 2) + math.ceil(math.log2(2))

    def test_sum_ancilla_qubits_single_term(self):
        """Sum with single term has no LCU overhead."""
        A = Basic("A", ancilla_qubits_=2)
        S = Sum([(1.0, A)])
        assert S.ancilla_qubits() == 2 + 0

    def test_sum_ancilla_qubits_edge_cases(self):
        """Test ancilla_qubits() for various Sum edge cases."""

        A = Basic("A", ancilla_qubits_=3)
        B = Basic("B", ancilla_qubits_=3)
        S2 = Sum([(1.0, A), (1.0, B)])
        assert S2.ancilla_qubits() == max(3, 3) + math.ceil(math.log2(2))

        C = Basic("C", ancilla_qubits_=5)
        S3 = Sum([(1.0, A), (1.0, C)])
        assert S3.ancilla_qubits() == max(3, 5) + math.ceil(math.log2(2))

        D = Basic("D", ancilla_qubits_=0)
        S4 = Sum([(1.0, A), (1.0, D)])
        assert S4.ancilla_qubits() == max(3, 0) + math.ceil(math.log2(2))

        E = Basic("E", ancilla_qubits_=0)
        S5 = Sum([(1.0, D), (1.0, E)])
        assert S5.ancilla_qubits() == max(0, 0) + math.ceil(math.log2(2))

        F = Basic("F", ancilla_qubits_=2)
        G = Basic("G", ancilla_qubits_=4)
        S6 = Sum([(1.0, A), (1.0, B), (1.0, F), (1.0, G)])
        assert S6.ancilla_qubits() == max(3, 3, 2, 4) + math.ceil(math.log2(4))

        S7 = Sum([(1.0, A), (1.0, B), (1.0, F)])
        assert S7.ancilla_qubits() == max(3, 3, 2) + math.ceil(math.log2(3))

        H = Basic("H", ancilla_qubits_=1)
        S8 = Sum([(1.0, A), (1.0, B), (1.0, F), (1.0, G), (1.0, H)])
        assert S8.ancilla_qubits() == max(3, 3, 2, 4, 1) + math.ceil(math.log2(5))

    def test_sum_normalize_flatten(self):
        """Nested sums are flattened."""
        A = Basic("A")
        B = Basic("B")
        inner = Sum([(1.0, A), (1.0, B)])
        outer = Sum([(1.0, inner), (1.0, A)])
        normalized = outer.optimize()
        expected = Sum([(2.0, A), (1.0, B)]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_sum_normalize_combine_like_terms(self):
        """Like terms are combined."""
        A = Basic("A")
        S = Sum([(1.0, A), (2.0, A), (0.5, A)])
        normalized = S.optimize()
        expected = Sum([(3.5, A)])
        assert Expr.structural_eq(normalized, expected)

    def test_sum_normalize_drop_zeros(self):
        """Zero coefficients are dropped."""
        A = Basic("A")
        B = Basic("B")
        S = Sum([(0.0, A), (1.0, B)])
        normalized = S.optimize()
        assert Expr.structural_eq(normalized, B)

    def test_sum_normalize_cancel_to_zero(self):
        """Canceling terms normalize to Const(0)."""
        A = Basic("A")
        S = Sum([(1.0, A), (-1.0, A)])
        normalized = S.optimize()
        assert Expr.structural_eq(normalized, Const(0.0))

    def test_sum_normalize_single_term(self):
        """Sum with single coefficient=1 term normalizes to term."""
        A = Basic("A")
        S = Sum([(1.0, A)])
        normalized = S.optimize()
        assert normalized == A

    def test_sum_normalize_poly_combination(self):
        """Polynomials with same base are combined."""
        A = Basic("A")
        P1 = Poly(A, Polynomial([1.0, 1.0]))
        P2 = Poly(A, Polynomial([0.0, 0.0, 1.0]))
        S = Sum([(1.0, P1), (1.0, P2)])
        normalized = S.optimize()
        expected = Poly(A, Polynomial([1.0, 1.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_sum_normalize_poly_combination2(self):
        """Polynomials with same base are combined."""
        A = Basic("A")
        P1 = Poly(A, Polynomial([1.0, 1.0]))
        P2 = Poly(A, Polynomial([0.0, 0.0, 1.0]))
        S = Sum([(1.0, P1), (1.0, P2)])
        normalized = S.optimize()
        expected = Poly(A, Polynomial([1.0, 1.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_lcu_cost(self):
        """Test LCU cost for sum of terms."""
        A = Basic("A", subnormalization_=1.0)
        B = Basic("B", subnormalization_=2.0)
        C = Basic("C", subnormalization_=3.0)
        expr = Sum([(0.5, A), (0.3, B), (0.2, C)])
        assert abs(expr.subnormalization() - 1.7) < TOL
        assert expr.queries() == 3


class TestProd:
    """Test product constructor."""

    def test_prod_subnormalization_simple(self):
        """Prod subnormalization = product of factor subnormalizations."""
        A = Basic("A", subnormalization_=2.0)
        B = Basic("B", subnormalization_=3.0)
        P = Prod([A, B])
        assert P.subnormalization() == 2.0 * 3.0

    def test_prod_queries_simple(self):
        """Prod queries = sum of factor queries values."""
        A = Basic("A")
        B = Basic("B")
        P = Prod([A, B])
        assert P.queries() == 1 + 1

    def test_prod_ancilla_qubits_simple(self):
        """Prod n = max of factor n values + ceil(log2(N))."""
        A = Basic("A", ancilla_qubits_=2)
        B = Basic("B", ancilla_qubits_=3)
        P = Prod([A, B])
        assert P.ancilla_qubits() == max(2, 3) + math.ceil(math.log2(2))

    def test_prod_ancilla_qubits_edge_cases(self):
        """Test ancilla_qubits() for various Prod edge cases."""
        A = Basic("A", ancilla_qubits_=5)
        P1 = Prod([A])
        assert P1.ancilla_qubits() == 5
        assert P1.circuit().ancilla_qubits == 5

        B = Basic("B", ancilla_qubits_=5)
        P2 = Prod([A, B])
        assert P2.ancilla_qubits() == max(5, 5) + math.ceil(math.log2(2))

        C = Basic("C", ancilla_qubits_=8)
        P3 = Prod([A, C])
        assert P3.ancilla_qubits() == max(5, 8) + math.ceil(math.log2(2))

        D = Basic("D", ancilla_qubits_=0)
        P4 = Prod([A, D])
        assert P4.ancilla_qubits() == max(5, 0) + math.ceil(math.log2(2))

        E = Basic("E", ancilla_qubits_=0)
        P5 = Prod([D, E])
        assert P5.ancilla_qubits() == max(0, 0) + math.ceil(math.log2(2))

        F = Basic("F", ancilla_qubits_=3)
        P6 = Prod([A, B, F])
        assert P6.ancilla_qubits() == max(5, 5, 3) + math.ceil(math.log2(3))

        G = Basic("G", ancilla_qubits_=10)
        P7 = Prod([A, B, F, G])
        assert P7.ancilla_qubits() == max(5, 5, 3, 10) + math.ceil(math.log2(4))

        H = Basic("H", ancilla_qubits_=2)
        I = Basic("I", ancilla_qubits_=2)
        J = Basic("J", ancilla_qubits_=2)
        P8 = Prod([H, I, J])
        assert P8.ancilla_qubits() == max(2, 2, 2) + math.ceil(math.log2(3))

        K = Basic("K", ancilla_qubits_=1)
        L = Basic("L", ancilla_qubits_=1)
        P9 = Prod([H, I, J, K, L])
        assert P9.ancilla_qubits() == max(2, 2, 2, 1, 1) + math.ceil(math.log2(5))

        P10 = Prod([H, I, J, K, L, H, I, J])
        assert P10.ancilla_qubits() == max(2, 2, 2, 1, 1, 2, 2, 2) + math.ceil(
            math.log2(8)
        )

    def test_prod_normalize_flatten(self):
        """Nested products are flattened."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        inner = Prod([A, B])
        outer = Prod([inner, C])
        normalized = outer.optimize()
        expected = Prod([A, B, C])
        assert Expr.structural_eq(normalized, expected)

    def test_prod_normalize_identity(self):
        """Const(1) identity is removed from product."""
        A = Basic("A")
        B = Basic("B")
        I = Const(1.0)

        P1 = Prod([A, I, I])
        assert P1.optimize() == A

        P2 = Prod([A, I, B, I])
        assert Expr.structural_eq(P2.optimize(), Prod([A, B]))

    def test_prod_normalize_zero(self):
        """Product with zero normalizes to zero."""
        A = Basic("A")
        Z = Const(0.0)
        P = Prod([A, Z])
        normalized = P.optimize()
        assert Expr.structural_eq(normalized, Const(0.0))

    def test_prod_normalize_repeated_to_poly(self):
        """Repeated factors A*A*A normalize to poly(A, x^3)."""
        A = Basic("A")
        P = Prod([A, A, A])
        normalized = P.optimize()
        expected = Poly(A, Polynomial([0.0, 0.0, 0.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_prod_normalize_poly_fusion(self):
        """poly(A, f) * poly(A, g) normalizes to poly(A, f*g)."""
        A = Basic("A")
        P1 = Poly(A, Polynomial([1.0, 1.0]))
        P2 = Poly(A, Polynomial([1.0, -1.0]))
        P = Prod([P1, P2])
        normalized = P.optimize()
        expected = Poly(A, Polynomial([1.0, 0.0, -1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_prod_normalize_if_same_condition(self):
        """Products of Ifs with same condition are combined."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        D = Basic("D")
        I1 = If(Condition("x"), A, B)
        I2 = If(Condition("x"), C, D)
        P = Prod([I1, I2])
        normalized = P.optimize()
        expected = If(Condition("x"), Prod([A, C]), Prod([B, D])).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_nested_product_cost(self):
        """Test cost of nested products."""
        A = Basic("A", subnormalization_=2.0)
        expr = Prod([A, A, A, A])
        normalized = expr.optimize()
        expected_subnormalization = 16.0 / (1 - QSP_TOL)
        assert abs(normalized.subnormalization() - expected_subnormalization) < TOL
        assert normalized.queries() == 4

    def test_prod_hermitian(self):
        """Product is Hermitian if all factors are Hermitian and commute."""
        A = Basic("A", hermitian=True)
        B = Basic("B", hermitian=True)
        P = Prod([A, B], hermitian=True)
        assert P.is_hermitian() is True

        P = Prod([A, B], hermitian=False)
        assert P.is_hermitian() is False

    def test_prod_hermitian_identical(self):
        """Product is Hermitian if all factors are Hermitian and identical."""
        A = Basic("A", hermitian=True)
        P = Prod([A, A])
        assert P.is_hermitian() is True

    def test_prod_hermitian_identical_non_commutative(self):
        """Product is Hermitian if all factors are Hermitian and identical."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        P = Prod([A, A])
        assert P.is_hermitian() is False

    def test_prod_hermitian_block_encoding(self):
        """Product is Hermitian block encoding if all factors are Hermitian block encoding."""
        A = Basic("A", hermitian_block_encoding=True)
        B = Basic("B", hermitian_block_encoding=True)
        P = Prod([A, B], hermitian=True)
        assert P.is_hermitian_block_encoding() is True

        B = Basic("B", hermitian_block_encoding=False)
        P = Prod([A, B], hermitian=True)
        assert P.is_hermitian() is True
        assert P.is_hermitian_block_encoding() is False


class TestTensor:
    """Test tensor product constructor."""

    def test_tensor_subnormalization_simple(self):
        """Tensor subnormalization = product of factor subnormalizations."""
        A = Basic("A", subnormalization_=2.0)
        B = Basic("B", subnormalization_=3.0)
        T = Tensor([A, B])
        assert T.subnormalization() == 2.0 * 3.0

    def test_tensor_queries_simple(self):
        """Tensor queries = sum of factor queries values."""
        A = Basic("A")
        B = Basic("B")
        T = Tensor([A, B])
        assert T.queries() == 1 + 1

    def test_tensor_ancilla_qubits_simple(self):
        """Tensor n = sum of factor n values."""
        A = Basic("A", ancilla_qubits_=2)
        B = Basic("B", ancilla_qubits_=3)
        T = Tensor([A, B])
        assert T.ancilla_qubits() == 2 + 3

    def test_tensor_ancilla_qubits_edge_cases(self):
        """Test ancilla_qubits() for various Tensor edge cases."""
        A = Basic("A", ancilla_qubits_=5)
        T1 = Tensor([A])
        assert T1.ancilla_qubits() == 5

        B = Basic("B", ancilla_qubits_=5)
        T2 = Tensor([A, B])
        assert T2.ancilla_qubits() == 5 + 5

        C = Basic("C", ancilla_qubits_=3)
        T3 = Tensor([A, C])
        assert T3.ancilla_qubits() == 5 + 3

        D = Basic("D", ancilla_qubits_=0)
        T4 = Tensor([A, D])
        assert T4.ancilla_qubits() == 5 + 0

        E = Basic("E", ancilla_qubits_=0)
        T5 = Tensor([D, E])
        assert T5.ancilla_qubits() == 0 + 0

        F = Basic("F", ancilla_qubits_=2)
        T6 = Tensor([A, B, F])
        assert T6.ancilla_qubits() == 5 + 5 + 2

        G = Basic("G", ancilla_qubits_=1)
        T7 = Tensor([A, B, F, G])
        assert T7.ancilla_qubits() == 5 + 5 + 2 + 1

        H = Basic("H", ancilla_qubits_=0)
        I = Basic("I", ancilla_qubits_=3)
        J = Basic("J", ancilla_qubits_=1)
        T8 = Tensor([A, H, I, J])
        assert T8.ancilla_qubits() == 5 + 0 + 3 + 1

    def test_tensor_normalize_flatten(self):
        """Nested tensors are flattened."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        inner = Tensor([A, B])
        outer = Tensor([inner, C])
        normalized = outer.optimize()
        expected = Tensor([A, B, C])
        assert Expr.structural_eq(normalized, expected)

    def test_tensor_normalize_identity_removal(self):
        """Const(1) is removed from tensor."""
        A = Basic("A")
        I1 = Const(1.0)
        T = Tensor([A, I1])
        normalized = T.optimize()
        assert normalized == A

    def test_tensor_normalize_const_product(self):
        """Multiple Const factors are multiplied together."""
        A = Basic("A")
        C1 = Const(2.0)
        C2 = Const(3.0)
        T = Tensor([C1, A, C2])
        normalized = T.optimize()
        expected = Tensor([Const(6.0), A])
        assert Expr.structural_eq(normalized, expected)

    def test_bilinear_factorization_simple(self):
        """Test factorization of (1+A)⊗(1+B) expansion."""
        A = Basic("A")
        B = Basic("B")
        I = Const(1.0)

        expr = Sum.of(
            Tensor([I, I]),
            Tensor([I, B]),
            Tensor([A, I]),
            Tensor([A, B]),
        )
        normalized = expr.optimize()
        expected = Tensor([Sum.of(I, A), Sum.of(I, B)])
        assert Expr.structural_eq(
            normalized, expected
        ), f"Expected (1+A)⊗(1+B), got {normalized}"

    def test_bilinear_factorization_with_negatives(self):
        """Test factorization of (1-A)⊗(1-B) expansion."""
        A = Basic("A")
        B = Basic("B")
        I = Const(1.0)

        expr = Tensor([I, I]) - Tensor([I, B]) - Tensor([A, I]) + Tensor([A, B])
        normalized = expr.optimize()
        expected = Tensor([Sum.of(I, -A), Sum.of(I, -B)])
        assert Expr.structural_eq(normalized, expected)

    def test_tensor_cost_multiplication(self):
        """Test that tensor multiplies subnormalization and adds g."""
        A = Basic("A", subnormalization_=2.0)
        B = Basic("B", subnormalization_=3.0)
        expr = Tensor([A, B])
        assert expr.subnormalization() == 2.0 * 3.0
        assert expr.queries() == 1 + 1


class TestIf:
    """Test conditional (if-then-else) constructor."""

    def test_if_queries_sum(self):
        """If queries = sum of branch queries values."""
        A = Basic("A")
        B = Basic("B")
        I = If(Condition("x"), A, B)
        assert I.queries() == 1 + 1

    def test_if_ancilla_qubits_max(self):
        """If n = max of branch n values."""
        A = Basic("A", ancilla_qubits_=2)
        B = Basic("B", ancilla_qubits_=3)
        I = If(Condition("x"), A, B)
        assert I.ancilla_qubits() == max(2, 3)

    def test_if_ancilla_qubits_edge_cases(self):
        """Test ancilla_qubits() for various If edge cases."""
        A = Basic("A", ancilla_qubits_=5)
        B = Basic("B", ancilla_qubits_=5)
        I1 = If(Condition("x"), A, B)
        assert I1.ancilla_qubits() == max(5, 5)

        C = Basic("C", ancilla_qubits_=8)
        I2 = If(Condition("x"), A, C)
        assert I2.ancilla_qubits() == max(5, 8)

        D = Basic("D", ancilla_qubits_=0)
        I3 = If(Condition("x"), A, D)
        assert I3.ancilla_qubits() == max(5, 0)

        E = Basic("E", ancilla_qubits_=0)
        I4 = If(Condition("x"), D, E)
        assert I4.ancilla_qubits() == max(0, 0)

        F = Basic("F", ancilla_qubits_=100)
        G = Basic("G", ancilla_qubits_=1)
        I5 = If(Condition("x"), F, G)
        assert I5.ancilla_qubits() == max(100, 1)

        I6 = If(Condition("x"), A, A)
        assert I6.ancilla_qubits() == max(5, 5)

    def test_if_normalize_same_branches(self):
        """If with identical branches normalizes to I ⊗ branch."""
        A = Basic("A")
        I = If(Condition("x"), A, A)
        normalized = I.optimize()
        assert normalized == Const(1.0).kron(A)

    def test_if_normalize_not_condition(self):
        """If with 'not x' swaps branches."""
        A = Basic("A")
        B = Basic("B")
        I = If(Condition("x", active=False), A, B)
        normalized = I.optimize()
        expected = If(Condition("x", active=True), B, A).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_if_normalize_factor_prod_left(self):
        """If with common left product factor is factored."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        I = If(Condition("x"), Prod([A, B]), Prod([A, C]))
        normalized = I.optimize()
        expected = Prod([A, If(Condition("x"), B, C)]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_if_normalize_factor_prod_right(self):
        """If with common right product factor is factored."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        I = If(Condition("x"), Prod([A, C]), Prod([B, C]))
        normalized = I.optimize()
        expected = Prod([If(Condition("x"), A, B), C]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_if_with_prod_branches(self):
        """If with product branches computes costs correctly."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")

        I = If(Condition("x"), Prod([A, B]), Prod([A, C]))
        normalized = I.optimize()
        expected = Prod([A, If(Condition("x"), B, C)]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_if_normalize_factor_tensor_left(self):
        """If with common left tensor factor is factored."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        I = If(Condition("x"), Tensor([A, B]), Tensor([A, C]))
        normalized = I.optimize()
        expected = Tensor([A, If(Condition("x"), B, C)]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_if_normalize_factor_tensor_right(self):
        """If with common right tensor factor is factored."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        I = If(Condition("x"), Tensor([A, C]), Tensor([B, C]))
        normalized = I.optimize()
        expected = Tensor([If(Condition("x"), A, B), C]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_conditional_cost(self):
        """Test that conditional takes max subnormalization and sum queries."""
        A = Basic("A", subnormalization_=5.0)
        B = Basic("B", subnormalization_=3.0)
        I = If(Condition("x"), A, B)
        assert I.subnormalization() == max(5.0, 3.0)
        assert I.queries() == 1 + 1


class TestPoly:
    """Test polynomial evaluation constructor."""

    def test_poly_subnormalization_simple(self):
        """Poly subnormalization = sup |f(x)| on [-a, a] where a = expr.subnormalization()."""
        A = Basic("A", subnormalization_=1.0)
        P = Poly(A, Polynomial([0.0, 2.0]))
        expected_subnormalization = 2.0 / (1 - QSP_TOL)
        assert abs(P.subnormalization() - expected_subnormalization) < TOL

    def test_poly_subnormalization_quadratic(self):
        """Poly with quadratic should evaluate sup correctly."""
        A = Basic("A", subnormalization_=2.0)
        P = Poly(A, Polynomial([0.0, 0.0, 1.0]))
        expected_subnormalization = 4.0 / (1 - QSP_TOL)
        assert abs(P.subnormalization() - expected_subnormalization) < TOL

    def test_poly_subnormalization_sup_norm(self):
        """Poly should evaluate sup norm correctly for QSP (interval [-1, 1])."""
        A = Basic("A", subnormalization_=2.0)
        P = Poly(A, Polynomial([1.0, 0.0, -1.0]))
        expected_subnormalization = 3.0 / (1 - QSP_TOL)
        assert abs(P.subnormalization() - expected_subnormalization) < TOL

    def test_poly_queries_simple(self):
        """Poly queries depends on parity: fixed parity = queries(expr), mixed parity is more."""
        A = Basic("A")
        P1 = Poly(A, Polynomial([1.0, 1.0, 1.0]))
        assert P1.queries() == 3
        P2 = Poly(A, Polynomial([1.0, 1.0, 1.0]))
        assert P2.queries() == 3

    def test_poly_ancilla_qubits_simple(self):
        """Poly n depends on parity: fixed parity = 1 + expr.ancilla_qubits(), mixed parity = 2 + expr.ancilla_qubits()."""
        A = Basic("A", ancilla_qubits_=0)
        B = Basic("B", ancilla_qubits_=3)
        P1 = Poly(A, Polynomial([1.0, 1.0]))
        P2 = Poly(B, Polynomial([1.0, 1.0]))
        assert P1.ancilla_qubits() == 2 + 0
        assert P2.ancilla_qubits() == 2 + 3

        P1_odd = Poly(A, Polynomial([0.0, 1.0]))
        P2_odd = Poly(B, Polynomial([0.0, 1.0]))
        assert P1_odd.ancilla_qubits() == 1 + 0
        assert P2_odd.ancilla_qubits() == 1 + 3

    def test_poly_ancilla_qubits_edge_cases(self):
        """Test ancilla_qubits() for Poly edge cases."""
        A = Basic("A", ancilla_qubits_=0)
        P1 = Poly(A, Polynomial([1.0]))
        assert P1.ancilla_qubits() == 1

        B = Basic("B", ancilla_qubits_=10)
        P2 = Poly(B, Polynomial([1.0, 1.0]))
        assert P2.ancilla_qubits() == 2 + 10

        P2_odd = Poly(B, Polynomial([0.0, 1.0]))
        assert P2_odd.ancilla_qubits() == 1 + 10

        C = Basic("C", ancilla_qubits_=2)
        S = Sum([(1.0, A), (1.0, C)])
        P3 = Poly(S, Polynomial([1.0, 1.0]))
        assert P3.ancilla_qubits() == 2 + max(0, 2) + math.ceil(math.log2(2))

        D = Basic("D", ancilla_qubits_=5)
        Pr = Prod([C, D])
        P4 = Poly(Pr, Polynomial([1.0, 1.0]))
        assert P4.ancilla_qubits() == 2 + max(2, 5) + math.ceil(math.log2(2))

        T = Tensor([A, C])
        P5 = Poly(T, Polynomial([1.0, 1.0]))
        assert P5.ancilla_qubits() == 2 + 2

    def test_poly_normalize_nested(self):
        """Nested poly composes: poly(poly(A, f), g) => poly(A, g∘f)."""
        A = Basic("A")
        inner = Poly(A, Polynomial([0.0, 2.0]))
        outer = Poly(inner, Polynomial([1.0, 1.0]))
        normalized = outer.optimize()
        expected = Poly(A, Polynomial([1.0, 2.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_poly_normalize_constant(self):
        """Poly with constant polynomial normalizes to Const."""
        A = Basic("A")
        P = Poly(A, Polynomial([5.0]))
        normalized = P.optimize()
        assert Expr.structural_eq(normalized, Const(5.0))

    def test_poly_normalize_identity_base(self):
        """Poly on Const(1) evaluates polynomial at x=1."""
        P = Poly(Const(1), Polynomial([2.0, 3.0]))
        normalized = P.optimize()
        assert Expr.structural_eq(normalized, Const(5.0))

    def test_poly_horner(self):
        """Poly with Horner's method evaluation."""
        A = Basic("A")
        P = Poly(A, Polynomial([1.0, 2.0, 3.0]))
        assert Expr.structural_eq(
            P.horner(), ((Const(1.0) * A) + Const(2.0)) * A + Const(3.0)
        )


class TestDivision:
    """Test division expression constructor and optimization."""

    def test_division_basic_same_expr(self):
        """A / A => Const(1.0)."""
        A = Basic("A")
        div = Division(A, A)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, Const(1.0))

    def test_division_constants(self):
        """Const(c1) / Const(c2) => Const(c1/c2)."""
        c1 = Const(6.0)
        c2 = Const(2.0)
        div = Division(c1, c2)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, Const(3.0))

    def test_division_by_constant(self):
        """X / Const(c) => (1/c) * X."""
        A = Basic("A")
        c = Const(2.0)
        div = Division(A, c)
        normalized = div.optimize()
        expected = Sum([(0.5, A)])
        assert Expr.structural_eq(normalized, expected)

    def test_division_by_zero(self):
        """Division by zero raises ValueError."""
        A = Basic("A")
        zero = Const(0.0)
        div = Division(A, zero)
        with pytest.raises(ValueError, match="Division by zero"):
            div.optimize()

    def test_division_double_reciprocal(self):
        """Const(1.0) / (Const(1.0) / A) => A."""
        A = Basic("A")
        one = Const(1.0)
        inner = Division(one, A)
        outer = Division(one, inner)
        normalized = outer.optimize()
        assert Expr.structural_eq(normalized, A)

    def test_division_poly_clean_division(self):
        """Poly(A, x^3) / Poly(A, x^2) => Poly(A, x)."""
        A = Basic("A")
        num = Poly(A, Polynomial([0.0, 0.0, 0.0, 1.0]))
        den = Poly(A, Polynomial([0.0, 0.0, 1.0]))
        div = Division(num, den)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, A)

    def test_division_poly_with_remainder(self):
        """Polynomial division with remainder stays unoptimized."""
        A = Basic("A")
        num = Poly(A, Polynomial([1.0, 1.0]))
        den = Poly(A, Polynomial([0.0, 1.0]))
        div = Division(num, den)
        normalized = div.optimize()
        assert isinstance(normalized, Division)

        num = Poly(A, Polynomial([1.0, 0.0, 1.0]))
        den = Poly(A, Polynomial([1.0, 1.0]))
        div = Division(num, den)
        normalized = div.optimize()
        assert isinstance(normalized, Division)

    def test_division_constant_polynomials(self):
        """Division of constant polynomials."""
        A = Basic("A")
        B = Basic("B")
        num = Poly(A, Polynomial([4.0]))
        den = Poly(B, Polynomial([2.0]))
        div = Division(num, den)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, Const(2.0))

    def test_division_product_cancellation(self):
        """(A * B) / A => B."""
        A = Basic("A")
        B = Basic("B")
        num = Prod([A, B])
        div = Division(num, A)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, B)

    def test_division_product_cancellation_reverse(self):
        """A / (A * B) => 1 / B."""
        A = Basic("A")
        B = Basic("B")
        den = Prod([A, B])
        div = Division(A, den)
        normalized = div.optimize()
        expected = Division(Const(1.0), B)
        assert Expr.structural_eq(normalized, expected)

    def test_division_product_multiple_cancellation(self):
        """(A * B * C) / (A * C) => B."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        num = Prod([A, B, C])
        den = Prod([A, C])
        div = Division(num, den)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, B)

    def test_division_sum_factorization(self):
        """(2A + 2B) / Const(2) => A + B."""
        A = Basic("A")
        B = Basic("B")
        num = Sum([(2.0, A), (2.0, B)])
        den = Const(2.0)
        div = Division(num, den)
        normalized = div.optimize()
        expected = Sum([(1.0, A), (1.0, B)])
        assert Expr.structural_eq(normalized, expected)

    def test_division_tensor_factorization(self):
        """(A ⊗ B) / (C ⊗ D) where A=C, B=D => Const(1) ⊗ Const(1) => Const(1)."""
        A = Basic("A")
        B = Basic("B")
        num = Tensor([A, B])
        den = Tensor([A, B])
        div = Division(num, den)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, Const(1.0))

    def test_division_tensor_same(self):
        """(A ⊗ B) / (A ⊗ B) => Const(1)."""
        A = Basic("A")
        B = Basic("B")
        num = Tensor([A, B])
        den = Tensor([A, B])
        div = Division(num, den)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, Const(1.0))

    def test_division_tensor_different_lengths(self):
        """Tensor division with different lengths stays unoptimized."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        num = Tensor([A, B])
        den = Tensor([C])
        div = Division(num, den)
        normalized = div.optimize()
        assert isinstance(normalized, Division)

    def test_division_subnormalization(self):
        """Division subnormalization is numerator / denominator."""
        A = Basic("A", subnormalization_=4.0)
        B = Basic("B", subnormalization_=2.0)
        div = Division(A, B)
        assert div.subnormalization() == A.subnormalization() / B.subnormalization()

    def test_division_unresolved_queries(self):
        """Unresolved division raises UnresolvedDivisionError for queries."""
        A = Basic("A")
        B = Basic("B")
        div = Division(A, B)
        with pytest.raises(UnresolvedDivisionError):
            div.queries()

    def test_division_unresolved_parallel_queries(self):
        """Unresolved division raises UnresolvedDivisionError for parallel queries."""
        A = Basic("A")
        B = Basic("B")
        div = Division(A, B)
        with pytest.raises(UnresolvedDivisionError):
            div.parallel_queries()

    def test_division_unresolved_ancilla(self):
        """Unresolved division raises UnresolvedDivisionError for ancilla."""
        A = Basic("A")
        B = Basic("B")
        div = Division(A, B)
        with pytest.raises(UnresolvedDivisionError):
            div.ancilla_qubits()

    def test_division_unresolved_circuit(self):
        """Unresolved division raises UnresolvedDivisionError for circuit."""
        A = Basic("A")
        B = Basic("B")
        div = Division(A, B)
        with pytest.raises(UnresolvedDivisionError):
            div.circuit()

    def test_division_unresolved_hermitian_block_encoding(self):
        """Unresolved division raises UnresolvedDivisionError for is_hermitian_block_encoding."""
        A = Basic("A")
        B = Basic("B")
        div = Division(A, B)
        with pytest.raises(UnresolvedDivisionError):
            div.is_hermitian_block_encoding()

    def test_division_is_hermitian(self):
        """Division is Hermitian if both operands are Hermitian."""
        A = Basic("A", hermitian=True)
        B = Basic("B", hermitian=True)
        C = Basic("C", hermitian=False, hermitian_block_encoding=False)
        div1 = Division(A, B)
        div2 = Division(A, C)
        assert div1.is_hermitian() is True
        assert div2.is_hermitian() is False

    def test_division_qtype_matching(self):
        """Division requires matching types."""
        A = Basic("A", qtype_=BitType(1))
        B = Basic("B", qtype_=BitType(1))
        C = Basic("C", qtype_=BitType(2))
        div1 = Division(A, B)
        div2 = Division(A, C)
        assert div1.qtype() == BitType(1)
        with pytest.raises(TypeCheckError):
            div2.qtype()

    def test_division_str(self):
        """Division string representation."""
        A = Basic("A")
        B = Basic("B")
        div = Division(A, B)
        assert str(div) == "(A / B)"

    def test_division_operator(self):
        """Test __truediv__ operator."""
        A = Basic("A")
        B = Basic("B")
        div = A / B
        assert isinstance(div, Division)
        assert div.numerator == A
        assert div.denominator == B

    def test_division_poly_powers(self):
        """Test polynomial power divisions: x^5 / x^2 = x^3."""
        A = Basic("A")
        num = Poly(A, Polynomial([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]))
        den = Poly(A, Polynomial([0.0, 0.0, 1.0]))
        div = Division(num, den)
        normalized = div.optimize()
        expected = Poly(A, Polynomial([0.0, 0.0, 0.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_division_poly_quadratic_by_linear(self):
        """Test (2 + 3x + x^2) / (1 + x) = (2 + x)."""
        A = Basic("A")
        num = Poly(A, Polynomial([2.0, 3.0, 1.0]))
        den = Poly(A, Polynomial([1.0, 1.0]))
        div = Division(num, den)
        normalized = div.optimize()
        expected = Poly(A, Polynomial([2.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_division_nested_products(self):
        """Test (A*B*C) / (B*C) = A."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        num = Prod([A, B, C])
        den = Prod([B, C])
        div = Division(num, den)
        normalized = div.optimize()
        assert Expr.structural_eq(normalized, A)

    def test_division_with_sum_numerator(self):
        """Test (A + B) / Const(2) = 0.5*A + 0.5*B."""
        A = Basic("A")
        B = Basic("B")
        num = Sum([(1.0, A), (1.0, B)])
        den = Const(2.0)
        div = Division(num, den)
        normalized = div.optimize()
        expected = Sum([(0.5, A), (0.5, B)])
        assert Expr.structural_eq(normalized, expected)

    def test_division_dagger(self):
        """Test Dagger(A/B) = (1/Dagger(B)) * Dagger(A) due to order reversal."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        B = Basic("B", hermitian=False, hermitian_block_encoding=False)
        div = Division(A, B)
        dag = Dagger(div)
        normalized = dag.optimize()
        expected = Prod([Division(Const(1.0), Dagger(B)), Dagger(A)])
        assert Expr.structural_eq(normalized, expected)

    def test_division_hermitian_dagger_simplifies(self):
        """Test Dagger(A/B) = (1/B) * A = A/B when both are Hermitian."""
        A = Basic("A", hermitian=True)
        B = Basic("B", hermitian=True)
        div = Division(A, B)
        dag = Dagger(div)
        normalized = dag.optimize()
        expected = Prod([Division(Const(1.0), B), A])
        assert Expr.structural_eq(normalized, expected)

    def test_division_odd_polynomial_simple(self):
        """Test simpler odd polynomial division: (x^5 - x^3) / x^2 = x^3 - x."""
        A = Basic("A")
        num_poly = Poly(A, Polynomial([0.0, 0.0, 0.0, -1.0, 0.0, 1.0]))
        den_poly = Poly(A, Polynomial([0.0, 0.0, 1.0]))

        div = Division(num_poly, den_poly)
        result = div.optimize()

        expected = Poly(A, Polynomial([0.0, -1.0, 0.0, 1.0]))
        assert Expr.structural_eq(result, expected)

        assert isinstance(result, Poly)
        assert result.p.parity() == Polynomial.Parity.ODD


class TestNormalize:
    """Test normalization behavior including idempotence, complex scenarios, and edge cases."""

    def test_normalize_idempotent_sum(self):
        """Normalizing twice should give the same result."""
        A = Basic("A")
        B = Basic("B")
        expr = Sum([(0.5, A), (0.5, A), (1.0, B)])
        norm1 = expr.optimize()
        norm2 = norm1.optimize()
        assert Expr.structural_eq(norm1, norm2)

    def test_normalize_idempotent_prod(self):
        """Normalizing a product twice should give the same result."""
        A = Basic("A")
        expr = Prod([A] * 5)
        norm1 = expr.optimize()
        norm2 = norm1.optimize()
        assert Expr.structural_eq(norm1, norm2)

    def test_normalize_idempotent_complex(self):
        """Normalizing complex expressions twice should give the same result."""
        A = Basic("A")
        B = Basic("B")
        expr = Sum([(1.0, Prod([A, B])), (1.0, Prod([A, B])), (-1.0, Tensor([A, B]))])
        norm1 = expr.optimize()
        norm2 = norm1.optimize()
        assert Expr.structural_eq(norm1, norm2)

    def test_normalize_adjoint_product(self):
        """Test A†A normalization to poly(A, x^2)."""
        A = Basic("A")
        expr = Prod([Dagger(A), A])
        normalized = expr.optimize()
        expected = Poly(A, Polynomial([0.0, 0.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_normalize_i_minus_a_squared(self):
        """Test I - A^2 normalization to poly(A, 1 - x^2)."""
        A = Basic("A")
        expr = Sum([(1.0, Const(1.0)), (-1.0, Prod([A, A]))])
        normalized = expr.optimize()
        expected = Poly(A, Polynomial([1.0, 0.0, -1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_normalize_tensor_flatten(self):
        """Test tensor flattening."""
        A = Basic("A")
        B = Basic("B")
        T_nested = Tensor([Tensor([A, B]), B])
        normalized = T_nested.optimize()
        expected = Tensor([A, B, B])
        assert Expr.structural_eq(normalized, expected)

    def test_normalize_sum_factorization_tensor(self):
        """Test factorization: (A⊗B) + (A⊗C) => A⊗(B+C)."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        expr = Sum([(1.0, Tensor([A, B])), (1.0, Tensor([A, C]))])
        normalized = expr.optimize()
        expected = Tensor([A, Sum([(1.0, B), (1.0, C)])]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_normalize_sum_factorization_prod(self):
        """Test factorization: (A*B) + (A*C) => A*(B+C)."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        expr = Sum([(1.0, Prod([A, B])), (1.0, Prod([A, C]))])
        normalized = expr.optimize()
        expected = Prod([A, Sum([(1.0, B), (1.0, C)])]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_normalize_power_to_poly(self):
        """Test (A+B)^3 normalizes to poly(A+B, x^3)."""
        A = Basic("A")
        B = Basic("B")
        sum_ab = Sum([(1.0, A), (1.0, B)])
        expr = Prod([sum_ab, sum_ab, sum_ab])
        normalized = expr.optimize()
        expected = Poly(sum_ab, Polynomial([0.0, 0.0, 0.0, 1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_normalize_polynomial_difference_squares(self):
        """Test (1+x)(1-x) = 1-x^2 via poly fusion."""
        A = Basic("A")
        P1 = Poly(A, Polynomial([1.0, 1.0]))
        P2 = Poly(A, Polynomial([1.0, -1.0]))
        expr = Prod([P1, P2])
        normalized = expr.optimize()
        expected = Poly(A, Polynomial([1.0, 0.0, -1.0]))
        assert Expr.structural_eq(normalized, expected)

    def test_empty_sum(self):
        """Sum with all zero terms normalizes to Const(0)."""
        A = Basic("A")
        S = Sum([(0.0, A), (0.0, A)])
        normalized = S.optimize()
        assert Expr.structural_eq(normalized, Const(0.0))

    def test_single_element_prod(self):
        """Product with single element normalizes to element."""
        A = Basic("A")
        P = Prod([A])
        normalized = P.optimize()
        assert normalized == A

    def test_single_element_tensor(self):
        """Tensor with single element normalizes to element."""
        A = Basic("A")
        T = Tensor([A])
        normalized = T.optimize()
        assert normalized == A

    def test_zero_degree_polynomial(self):
        """Constant polynomial normalizes to Const."""
        A = Basic("A")
        P = Poly(A, Polynomial([3.14159]))
        normalized = P.optimize()
        assert Expr.structural_eq(normalized, Const(3.14159))

    def test_very_small_coefficients(self):
        """Coefficients below TOL are treated as zero."""
        A = Basic("A")
        S = Sum([(1e-20, A), (1.0, A)])
        normalized = S.optimize()
        assert Expr.structural_eq(normalized, A)

    def test_structural_eq_with_tolerance(self):
        """Structural equality respects tolerance for floats."""
        A = Basic("A")
        S1 = Sum([(1.0 + 1e-17, A)])
        S2 = Sum([(1.0, A)])
        assert Expr.structural_eq(S1, S2)


class TestOperators:
    """Test operator overloads and utility functions for Expr."""

    def test_total_cost(self):
        """Test total_cost = subnormalization * queries."""
        A = Basic("A", subnormalization_=2.5)
        B = Basic("B", subnormalization_=3.0)
        expr = Prod([A, B])
        assert expr.total_cost() == A.subnormalization() * B.subnormalization() * (
            A.queries() + B.queries()
        )

    def test_add_operator(self):
        """Test A + B creates Sum([(1.0, A), (1.0, B)])."""
        A = Basic("A")
        B = Basic("B")
        result = A + B
        assert Expr.structural_eq(result, Sum([(1.0, A), (1.0, B)]))

    def test_add_operator_chain(self):
        """Test chaining addition: A + B + C."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result = A + B + C
        normalized = result.optimize()
        expected = Sum([(1.0, A), (1.0, B), (1.0, C)]).optimize()
        assert Expr.structural_eq(normalized, expected)

    def test_add_operator_with_const(self):
        """Test addition with constants."""
        A = Basic("A")
        C = Const(2.0)
        result = A + C
        assert Expr.structural_eq(result, Sum([(1.0, A), (1.0, C)]))

    def test_sub_operator(self):
        """Test A - B creates Sum([(1.0, A), (-1.0, B)])."""
        A = Basic("A")
        B = Basic("B")
        result = A - B
        assert Expr.structural_eq(result, Sum([(1.0, A), (-1.0, B)]))

    def test_sub_operator_normalizes(self):
        """Test A - A normalizes to zero."""
        A = Basic("A")
        result = (A - A).optimize()
        assert Expr.structural_eq(result, Const(0.0))

    def test_sub_operator_chain(self):
        """Test chaining subtraction: A - B - C."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result = A - B - C
        assert Expr.structural_eq(result, Sum([(1.0, A), (-1.0, B), (-1.0, C)]))

    def test_neg_operator(self):
        """Test -A creates Sum([(-1.0, A)])."""
        A = Basic("A")
        result = -A
        assert Expr.structural_eq(result, Sum([(-1.0, A)]))

    def test_neg_operator_double(self):
        """Test -(-A) normalizes to A."""
        A = Basic("A")
        result = (-(-A)).optimize()
        assert Expr.structural_eq(result, A)

    def test_neg_operator_in_expression(self):
        """Test negation in larger expression."""
        A = Basic("A")
        B = Basic("B")
        result = (-A) + B
        assert Expr.structural_eq(result, Sum([(-1.0, A), (1.0, B)]))

    def test_mul_operator_with_expr(self):
        """Test A * B creates Prod([A, B])."""
        A = Basic("A")
        B = Basic("B")
        result = A * B
        assert Expr.structural_eq(result, Prod([A, B]))

    def test_mul_operator_with_int(self):
        """Test A * scalar creates Sum([(scalar, A)])."""
        A = Basic("A")
        result = A * 3
        assert Expr.structural_eq(result, Sum([(3.0, A)]))

    def test_mul_operator_with_float(self):
        """Test A * float creates Sum([(float, A)])."""
        A = Basic("A")
        result = A * 2.5
        assert Expr.structural_eq(result, Sum([(2.5, A)]))

    def test_mul_operator_chain(self):
        """Test chaining multiplication: A * B * C."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result = A * B * C
        assert Expr.structural_eq(result, Prod([A, B, C]))

    def test_mul_operator_scalar_then_expr(self):
        """Test (A * 2) * B."""
        A = Basic("A")
        B = Basic("B")
        result = (A * 2) * B
        assert Expr.structural_eq(result, Prod([Sum([(2.0, A)]), B]))

    def test_rmul_operator_int(self):
        """Test scalar * A creates Sum([(scalar, A)])."""
        A = Basic("A")
        result = 3 * A
        assert Expr.structural_eq(result, Sum([(3.0, A)]))

    def test_rmul_operator_float(self):
        """Test float * A creates Sum([(float, A)])."""
        A = Basic("A")
        result = 2.5 * A
        assert Expr.structural_eq(result, Sum([(2.5, A)]))

    def test_rmul_equals_mul_scalar(self):
        """Test that A * scalar equals scalar * A."""
        A = Basic("A")
        assert Expr.structural_eq(A * 3, 3 * A)

    def test_rmul_in_expression(self):
        """Test scalar * A in larger expression."""
        A = Basic("A")
        B = Basic("B")
        result = 2 * A + 3 * B
        assert Expr.structural_eq(result, Sum([(2.0, A), (3.0, B)]))

    def test_tensor_method(self):
        """Test A tensor B creates Tensor([A, B])."""
        A = Basic("A")
        B = Basic("B")
        result = Tensor.of(A, B)
        assert Expr.structural_eq(result, Tensor([A, B]))

    def test_tensor_method_chain(self):
        """Test chaining tensor: A tensor B tensor C."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result = Tensor.of(A, B, C)
        assert Expr.structural_eq(result, Tensor([A, B, C]))

    def test_tensor_method_with_sum(self):
        """Test tensor product of sums."""
        A = Basic("A")
        B = Basic("B")
        sum_a = Sum([(1.0, A)])
        sum_b = Sum([(1.0, B)])
        result = Tensor.of(sum_a, sum_b)
        assert Expr.structural_eq(result, Tensor([Sum([(1.0, A)]), Sum([(1.0, B)])]))

    def test_pow_operator(self):
        """Test A**n creates Prod([A] * n)."""
        A = Basic("A")
        result = A**3
        assert Expr.structural_eq(result, Prod([A, A, A]))

    def test_pow_operator_one(self):
        """Test A**1 creates single-element Prod that normalizes to A."""
        A = Basic("A")
        result = (A**1).optimize()
        assert Expr.structural_eq(result, A)

    def test_pow_operator_zero(self):
        """Test A**0 creates empty Prod that normalizes to Const(1)."""
        A = Basic("A")
        result = (A**0).optimize()
        assert Expr.structural_eq(result, Const(1.0))

    def test_pow_operator_normalizes_to_poly(self):
        """Test A**n normalizes to poly(A, x^n)."""
        A = Basic("A")
        result = (A**4).optimize()
        assert Expr.structural_eq(
            result, Poly(A, Polynomial([0.0, 0.0, 0.0, 0.0, 1.0]))
        )

    def test_pow_operator_large(self):
        """Test larger powers."""
        A = Basic("A")
        result = A**10
        assert Expr.structural_eq(result, Prod([A] * 10))

    def test_combined_operators_add_mul(self):
        """Test A + B * C."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result = A + (B * C)
        assert Expr.structural_eq(result, Sum([(1.0, A), (1.0, Prod([B, C]))]))

    def test_combined_operators_mul_add(self):
        """Test (A + B) * C using operator precedence."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result1 = A + B * C
        result2 = (A + B) * C
        assert not Expr.structural_eq(result1, result2)

    def test_combined_operators_tensor_add(self):
        """Test A tensor (B + C)."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        result = Tensor.of(A, B + C)
        assert Expr.structural_eq(result, Tensor([A, Sum([(1.0, B), (1.0, C)])]))

    def test_combined_operators_pow_mul(self):
        """Test (A * B)**2."""
        A = Basic("A")
        B = Basic("B")
        result = (A * B) ** 2
        assert Expr.structural_eq(result, Prod([Prod([A, B]), Prod([A, B])]))

    def test_combined_operators_complex(self):
        """Test complex expression: (A + B)**2 using operators."""
        A = Basic("A")
        B = Basic("B")
        expr = (A + B) ** 2
        assert Expr.structural_eq(
            expr, Poly(Sum([(1.0, A), (1.0, B)]), Polynomial([0.0, 0.0, 1.0]))
        )

    def test_combined_operators_distributive_manual(self):
        """Test manual distribution: A*B + A*C vs A*(B+C)."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        manual = A * B + A * C
        factored = A * (B + C)
        assert Expr.structural_eq(manual.optimize(), factored.optimize())

    def test_operators_preserve_types(self):
        """Test that operators preserve type information."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        result_add = (A + B).optimize()
        assert result_add.qtype() == BitType(2)
        A_herm = Basic("A", qtype_=BitType(2))
        B_herm = Basic("B", qtype_=BitType(2))
        result_mul = (A_herm * B_herm).optimize()
        assert result_mul.qtype() == BitType(2)

    def test_operators_with_dagger(self):
        """Test operators with Dagger expressions."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        result = A + Dagger(A)
        assert Expr.structural_eq(result, Sum([(1.0, A), (1.0, Dagger(A))]))

    def test_operators_with_poly(self):
        """Test operators with Poly expressions."""
        A = Basic("A")
        P = Poly(A, Polynomial([1.0, 1.0]))
        result = P + A
        assert Expr.structural_eq(result, Poly(A, Polynomial([1.0, 2.0])))

    def test_scalar_multiplication_zero(self):
        """Test multiplication by zero."""
        A = Basic("A")
        result = (0 * A).optimize()
        assert Expr.structural_eq(result, Const(0.0))

    def test_scalar_multiplication_one(self):
        """Test multiplication by one."""
        A = Basic("A")
        result = (1 * A).optimize()
        assert Expr.structural_eq(result, A)

    def test_scalar_multiplication_negative(self):
        """Test multiplication by negative scalar."""
        A = Basic("A")
        result = -2 * A
        assert Expr.structural_eq(result, Sum([(-2.0, A)]))

    def test_operator_associativity_add(self):
        """Test associativity: (A + B) + C == A + (B + C)."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        assert Expr.structural_eq(((A + B) + C).optimize(), (A + (B + C)).optimize())

    def test_operator_associativity_mul(self):
        """Test associativity: (A * B) * C == A * (B * C)."""
        A = Basic("A")
        B = Basic("B")
        C = Basic("C")
        assert Expr.structural_eq(((A * B) * C).optimize(), (A * (B * C)).optimize())

    def test_operator_commutativity_add(self):
        """Test commutativity: A + B == B + A."""
        A = Basic("A")
        B = Basic("B")
        assert Expr.structural_eq((A + B).optimize(), (B + A).optimize())

    def test_operator_tensor_not_commutative(self):
        """Test tensor is not commutative: A tensor B != B tensor A."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        result1 = Tensor.of(A, B)
        result2 = Tensor.of(B, A)
        qtype1 = result1.qtype()
        qtype2 = result2.qtype()
        assert qtype1 != qtype2
        assert qtype1.width() == qtype2.width()
        assert isinstance(qtype1, TensorType)
        assert isinstance(qtype2, TensorType)
        assert qtype1.factors == (BitType(2), BitType(3))
        assert qtype2.factors == (BitType(3), BitType(2))
