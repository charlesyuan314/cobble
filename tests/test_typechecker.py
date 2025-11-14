from cobble.expr import Basic, Condition, Const, Dagger, If, Poly, Prod, Sum, Tensor
from cobble.polynomial import Polynomial
from cobble.qtype import BitType, TensorType, TypeCheckError, make_tensor_type
import pytest


class TestTypeChecker:
    """Test type checking for block encoding expressions."""

    def test_base_default_type(self):
        """Base matrices have default named type with width 1."""
        A = Basic("A")
        t = A.qtype()
        assert isinstance(t, BitType)
        assert t.width() == 1

    def test_base_custom_type(self):
        """Base matrices can have custom types."""
        custom_type = BitType(3)
        A = Basic("A", qtype_=custom_type)
        t = A.qtype()
        assert t == custom_type
        assert t.width() == 3

    def test_const_type(self):
        """Const has type bit[1]."""
        c = Const(3.14)
        t = c.qtype()
        assert isinstance(t, BitType)
        assert t.n == 1
        assert t.width() == 1

    def test_dagger_preserves_type(self):
        """Dagger preserves the type of its operand."""
        A = Basic("A", qtype_=BitType(2))
        dag = Dagger(A)
        assert dag.qtype() == A.qtype()
        assert dag.qtype().width() == 2

    def test_sum_same_type(self):
        """Sum of expressions with same type succeeds."""
        t = BitType(2)
        A = Basic("A", qtype_=t)
        B = Basic("B", qtype_=t)
        s = Sum([(1.0, A), (1.0, B)])
        assert s.qtype() == t

    def test_sum_with_const(self):
        """Sum can include Const with other types."""
        A = Basic("A")
        c = Const(2.0)
        s = Sum([(1.0, A), (1.0, c)])
        assert s.qtype() == A.qtype()

    def test_sum_type_mismatch(self):
        """Sum of expressions with different types raises error."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        s = Sum([(1.0, A), (1.0, B)])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = s.qtype()

    def test_prod_same_type(self):
        """Product of expressions with same type succeeds."""
        t = BitType(2)
        A = Basic("A", qtype_=t)
        B = Basic("B", qtype_=t)
        p = Prod([A, B], hermitian=True)
        assert p.qtype() == t

    def test_prod_with_const(self):
        """Product can include Const with other types."""
        A = Basic("A")
        c = Const(2.0)
        p = Prod([c, A])
        assert p.qtype() == A.qtype()

    def test_prod_type_mismatch(self):
        """Product of expressions with different types raises error."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        p = Prod([A, B])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = p.qtype()

    def test_tensor_creates_product_type(self):
        """Tensor product creates a tensor type."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        t = Tensor([A, B])
        result_type = t.qtype()
        assert isinstance(result_type, TensorType)
        assert result_type.factors == (BitType(2), BitType(3))
        assert result_type.width() == 5

    def test_tensor_flattens(self):
        """Tensor product flattens nested tensors."""
        A = Basic("A", qtype_=BitType(1))
        B = Basic("B", qtype_=BitType(2))
        C = Basic("C", qtype_=BitType(3))
        AB = Tensor([A, B])
        ABC = Tensor([AB, C])
        result_type = ABC.qtype()
        assert isinstance(result_type, TensorType)
        assert result_type.factors == (BitType(1), BitType(2), BitType(3))
        assert result_type.width() == 6

    def test_if_adds_bool(self):
        """If statement adds a bool (bit[1]) to the type."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        cond = Condition("x")
        i = If(cond, A, B)
        result_type = i.qtype()
        assert isinstance(result_type, TensorType)
        assert result_type.factors == (BitType(2), BitType(1))
        assert result_type.width() == 3

    def test_if_branch_type_mismatch(self):
        """If statement with different branch types raises error."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        cond = Condition("x")
        i = If(cond, A, B)
        with pytest.raises(TypeCheckError, match="same type"):
            _ = i.qtype()

    def test_poly_preserves_type(self):
        """Poly preserves the type of its operand."""
        A = Basic("A", qtype_=BitType(3))
        p = Poly(A, Polynomial([1.0, 0.0, 1.0]))
        assert p.qtype() == A.qtype()
        assert p.qtype().width() == 3

    def test_poly_requires_hermitian(self):
        """Poly requires Hermitian operand."""
        A = Basic(
            "A", hermitian=False, hermitian_block_encoding=False, qtype_=BitType(2)
        )
        p = Poly(A, Polynomial([1.0, 1.0]))
        with pytest.raises(TypeCheckError, match="Hermitian"):
            _ = p.qtype()

    def test_complex_expression_type(self):
        """Complex expression: (A + B)^2 has correct type."""
        t = BitType(2)
        A = Basic("A", qtype_=t)
        B = Basic("B", qtype_=t)
        sum_ab = Sum([(1.0, A), (1.0, B)])
        prod = Prod([sum_ab, sum_ab], hermitian=True)
        assert prod.qtype() == t

    def test_tensor_sum_type(self):
        """Tensor of sums has correct type."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        C = Basic("C", qtype_=BitType(3))
        D = Basic("D", qtype_=BitType(3))
        sum1 = Sum([(1.0, A), (1.0, B)])
        sum2 = Sum([(1.0, C), (1.0, D)])
        t = Tensor([sum1, sum2])
        result_type = t.qtype()
        assert isinstance(result_type, TensorType)
        assert result_type.width() == 5

    def test_nested_if_type(self):
        """Nested if statements accumulate bool types."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        cond1 = Condition("x")
        cond2 = Condition("y")
        i1 = If(cond1, A, B)
        i2_branch = If(cond1, A, B)
        i2 = If(cond2, i1, i2_branch)
        result_type = i2.qtype()
        assert isinstance(result_type, TensorType)
        assert result_type.width() == 2 + 1 + 1

    def test_identity_with_sum(self):
        """Identity (I) can be summed with other expressions."""
        I = Const(1.0)
        A = Basic("A")
        s = Sum([(1.0, I), (-1.0, A)])
        result_type = s.qtype()
        assert result_type.width() == 1

    def test_make_tensor_type_flattens(self):
        """Helper function make_tensor_type flattens correctly."""
        t1 = BitType(2)
        t2 = BitType(3)
        inner = TensorType((t1, t2))
        result = make_tensor_type(inner, t1)
        assert isinstance(result, TensorType)
        assert result.factors == (t1, t2, t1)
        assert result.width() == 7

    def test_make_tensor_type_single(self):
        """Helper function returns single type without wrapping."""
        t = BitType(2)
        result = make_tensor_type(t)
        assert result == t

    def test_type_equality(self):
        """Test type equality checks."""
        t1 = BitType(2)
        t2 = BitType(2)
        t3 = BitType(3)
        assert t1 == t2
        assert t1 != t3

        tensor1 = TensorType((BitType(2), BitType(3)))
        tensor2 = TensorType((BitType(2), BitType(3)))
        tensor3 = TensorType((BitType(3), BitType(2)))
        assert tensor1 == tensor2
        assert tensor1 != tensor3

    def test_type_str(self):
        """Test type string representations."""
        assert str(BitType(1)) == "bool"
        assert str(BitType(2)) == "bit[2]"
        assert str(TensorType((BitType(2), BitType(3)))) == "bit[2] ⊗ bit[3]"

    def test_normalized_expr_type(self):
        """Type checking works on normalized expressions."""
        A = Basic("A", qtype_=BitType(2))
        s = Sum([(1.0, A), (1.0, A)])
        normalized = s.optimize()
        assert normalized.qtype() == A.qtype()

    def test_type_check_before_optimize(self):
        """Type checking should work before normalization."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        expr = Prod([Sum([(1.0, A), (1.0, B)]), Sum([(1.0, A), (-1.0, B)])])
        t = expr.qtype()
        assert t == BitType(2)


class TestInvalidPrograms:
    """Test programs that should fail type checking."""

    def test_sum_empty(self):
        """Empty sum should raise error."""
        s = Sum([])
        with pytest.raises(TypeCheckError, match="at least one term"):
            _ = s.qtype()

    def test_prod_empty(self):
        """Empty product should raise error."""
        p = Prod([])
        with pytest.raises(TypeCheckError, match="at least one factor"):
            _ = p.qtype()

    def test_tensor_empty(self):
        """Empty tensor should raise error."""
        t = Tensor([])
        with pytest.raises(TypeCheckError, match="at least one factor"):
            _ = t.qtype()

    def test_prod_incompatible_bit_types(self):
        """Product with different bit types should fail."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        p = Prod([A, B], hermitian=True)
        with pytest.raises(TypeCheckError, match="same type"):
            _ = p.qtype()

    def test_poly_non_hermitian_base(self):
        """Poly on non-Hermitian base should fail."""
        A = Basic("A", hermitian=False, hermitian_block_encoding=False)
        p = Poly(A, Polynomial([1.0, 1.0]))
        with pytest.raises(TypeCheckError, match="Hermitian"):
            _ = p.qtype()

    def test_poly_non_hermitian_sum(self):
        """Poly on non-Hermitian sum should fail."""
        A = Basic("A", hermitian=True)
        B = Basic("B", hermitian=False, hermitian_block_encoding=False)
        s = Sum([(1.0, A), (1.0, B)])
        p = Poly(s, Polynomial([1.0, 1.0]))
        with pytest.raises(TypeCheckError, match="Hermitian"):
            _ = p.qtype()

    def test_poly_non_hermitian_product(self):
        """Poly on non-Hermitian product should fail."""
        A = Basic("A")
        B = Basic("B")
        p_prod = Prod([A, B], hermitian=False)
        poly = Poly(p_prod, Polynomial([1.0, 1.0]))
        with pytest.raises(TypeCheckError, match="Hermitian"):
            _ = poly.qtype()

    def test_if_mismatched_branch_widths(self):
        """If with different branch widths should fail."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        cond = Condition("x")
        i = If(cond, A, B)
        with pytest.raises(TypeCheckError, match="same type"):
            _ = i.qtype()

    def test_nested_type_mismatch_in_sum(self):
        """Nested type mismatch in sum should fail."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        p1 = Poly(A, Polynomial([1.0, 1.0]))
        p2 = Poly(B, Polynomial([1.0, 1.0]))
        s = Sum([(1.0, p1), (1.0, p2)])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = s.qtype()

    def test_nested_type_mismatch_in_prod(self):
        """Nested type mismatch in product should fail."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        p1 = Poly(A, Polynomial([1.0, 1.0]))
        p2 = Poly(B, Polynomial([1.0, 1.0]))
        prod = Prod([p1, p2])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = prod.qtype()

    def test_tensor_prod_type_mismatch(self):
        """Tensor product followed by regular product with type mismatch."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        C = Basic("C", qtype_=BitType(2))
        t = Tensor([A, B])
        prod = Prod([t, C])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = prod.qtype()

    def test_sum_tensor_mismatch(self):
        """Sum of tensor products with different structures."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        C = Basic("C", qtype_=BitType(2))
        D = Basic("D", qtype_=BitType(3))
        t1 = Tensor([A, B])
        t2 = Tensor([C, D])
        s = Sum([(1.0, t1), (1.0, t2)])
        assert s.qtype().width() == 5
        t3 = Tensor([B, A])
        s2 = Sum([(1.0, t1), (1.0, t3)])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = s2.qtype()

    def test_complex_nested_mismatch(self):
        """Complex nested expression with type mismatch."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        C = Basic("C", qtype_=BitType(2))

        sum_ac = Sum([(1.0, A), (1.0, C)])
        poly_ac = Poly(sum_ac, Polynomial([1.0, 1.0]))

        poly_b = Poly(B, Polynomial([1.0, 1.0]))

        s = Sum([(1.0, poly_ac), (1.0, poly_b)])
        with pytest.raises(TypeCheckError, match="same type"):
            _ = s.qtype()

    def test_if_with_tensor_mismatch(self):
        """If statement with tensor products that don't match."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        C = Basic("C", qtype_=BitType(3))

        cond = Condition("x")
        t1 = Tensor([A, B])
        t2 = Tensor([A, C])
        i = If(cond, t1, t2)
        assert i.qtype().width() == 2 + 3 + 1

        t3 = Tensor([B, A])
        i2 = If(cond, t1, t3)
        with pytest.raises(TypeCheckError, match="same type"):
            _ = i2.qtype()
