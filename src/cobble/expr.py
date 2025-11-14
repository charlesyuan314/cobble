from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
import math
import os
from typing import Generator, TYPE_CHECKING, override

from cobble.compile import (
    _compile_base,
    _compile_const,
    _compile_if,
    _compile_poly,
    _compile_prod,
    _compile_sum,
    _compile_tensor,
)
from cobble.optimize import (
    _combine_base_expr_with_poly,
    _combine_polys_with_same_base,
    _factor_sum,
    _flatten_and_combine_terms,
    _merge_const_with_poly,
    _merge_scalars_into_expressions,
    _try_combine_divisions_in_sum,
    _try_poly_product_fusion,
)
from cobble.polynomial import Polynomial, TOL
from cobble.qtype import BitType, Type, TypeCheckError, make_tensor_type

if TYPE_CHECKING:
    from cobble.circuit import Circuit

# Tolerance for QSP, must be less precise than TOL for phase generation
QSP_TOL = 1e-6


class UnresolvedDivisionError(Exception):
    """Raised when a division cannot be resolved/compiled."""


class Expr(ABC):
    @abstractmethod
    def optimize(self) -> Expr:
        """Return an optimized version of this expression."""

    @abstractmethod
    def subnormalization(self) -> float:
        """Return the subnormalization of this expression."""

    @abstractmethod
    def queries(self) -> int:
        """Return the number of queries to black-box block encodings used by this expression."""

    def total_cost(self) -> float:
        return self.subnormalization() * self.queries()

    @abstractmethod
    def parallel_queries(self) -> int:
        """Return the parallel query complexity used by this expression."""

    @abstractmethod
    def ancilla_qubits(self) -> int:
        """Return the number of ancilla qubits used by this expression."""

    @abstractmethod
    def is_hermitian(self) -> bool:
        """Return True if the encoded expression is Hermitian."""

    @abstractmethod
    def is_hermitian_block_encoding(self) -> bool:
        """Return True if this expression is a Hermitian block encoding of a Hermitian expression."""

    @abstractmethod
    def qtype(self) -> Type:
        """Return the type of this expression. Raises TypeCheckError if ill-typed."""

    @abstractmethod
    def circuit(self) -> Circuit:
        """Compile this expression to a quantum circuit."""

    @abstractmethod
    @override
    def __str__(self) -> str:
        """Return a string representation of this expression."""

    def __add__(self, other: Expr) -> Sum:
        """A + B = Sum([(1.0, A), (1.0, B)])"""
        return Sum([(1.0, self), (1.0, other)])

    def __sub__(self, other: Expr) -> Sum:
        """A - B = Sum([(1.0, A), (-1.0, B)])"""
        return Sum([(1.0, self), (-1.0, other)])

    def __neg__(self) -> Sum:
        """-A = Sum([(-1.0, A)])"""
        return Sum([(-1.0, self)])

    def __mul__(self, other: int | float | Expr) -> Sum | Prod:
        """A * B = Prod([A, B]) or scalar * A = Sum([(scalar, A)])"""
        if isinstance(other, (int, float)):
            return Sum(
                [
                    (float(other), self),
                ]
            )
        return Prod([self, other])

    def __rmul__(self, other: int | float) -> Sum:
        """scalar * A = Sum([(scalar, A)])"""
        return Sum([(float(other), self)])

    @classmethod
    def kron(cls, *factors: Expr) -> Tensor:
        """A kron B = Tensor([A, B])"""
        return Tensor.of(*factors)

    def __pow__(self, other: int) -> Prod | Const:
        """A**n = Prod([A] * n)"""
        if other < 0:
            raise ValueError("Negative power not supported")
        if other == 0:
            return Const(1.0)
        return Prod([self] * other)

    def kronpow(self, other: int) -> Tensor:
        """A ^(⊗n) = Tensor([A] * n)"""
        if other <= 0:
            raise ValueError("Negative or zero power not supported")
        return Tensor([self] * other)

    def __truediv__(self, other: Expr | float) -> Expr:
        """A / B = Division(A, B)"""
        if isinstance(other, (int, float)):
            return 1.0 / other * self
        return Division(self, other)

    @classmethod
    def structural_eq(cls, a: Expr, b: Expr, tol: float = TOL) -> bool:
        a = a.optimize()
        b = b.optimize()

        if type(a) is not type(b):
            return False
        if isinstance(a, Basic):
            return a == b
        if isinstance(a, Const):
            assert isinstance(b, type(a))
            return abs(a.value - b.value) < tol
        if isinstance(a, Dagger):
            assert isinstance(b, type(a))
            return cls.structural_eq(a.expr, b.expr, tol)
        if isinstance(a, Poly):
            assert isinstance(b, type(a))
            return cls.structural_eq(a.expr, b.expr, tol) and a.p.approx_eq(b.p, tol)
        if isinstance(a, Sum):
            assert isinstance(b, type(a))

            # Compare as unordered multiset after combining like terms
            def norm_terms(s: Sum) -> list[tuple[float, Expr]]:
                items: list[tuple[float, Expr]] = []
                for c, t in s.terms:
                    t = t.optimize()
                    if abs(c) < TOL:
                        continue
                    # merge with equal terms
                    merged = False
                    for i, (c2, t2) in enumerate(items):
                        if cls.structural_eq(t, t2, tol):
                            items[i] = (c2 + c, t2)
                            merged = True
                            break
                    if not merged:
                        items.append((c, t))
                items.sort(key=lambda ct: (repr(ct[1]), round(ct[0], 12)))
                return items

            return norm_terms(a) == norm_terms(b)
        if isinstance(a, Prod) or isinstance(a, Tensor):
            assert isinstance(b, type(a))
            return len(a.factors) == len(b.factors) and all(
                cls.structural_eq(x, y, tol) for x, y in zip(a.factors, b.factors)
            )
        if isinstance(a, If):
            assert isinstance(b, type(a))
            return (
                a.cond == b.cond
                and cls.structural_eq(a.then_branch, b.then_branch, tol)
                and cls.structural_eq(a.else_branch, b.else_branch, tol)
            )
        if isinstance(a, Division):
            assert isinstance(b, type(a))
            return cls.structural_eq(
                a.numerator, b.numerator, tol
            ) and cls.structural_eq(a.denominator, b.denominator, tol)
        return repr(a) == repr(b)


@dataclass(frozen=True)
class Basic(Expr):
    """Black-box oracle for block encoding a matrix.

    Attributes:
        name: Name of the black-box oracle.
            Can be a unitary logic gate such as "X" or "H".
            Can start with "$" to compile to a placeholder u3 gate in QASM.
        hermitian: Whether the encoded matrix is Hermitian.
        hermitian_block_encoding: Whether the block encoding is itself Hermitian.
        subnormalization_: Explicitly specified subnormalization of the matrix.
        ancilla_qubits_: Explicitly specified number of ancilla qubits used.
        qtype_: Explicitly specified type of the matrix.
        circuit_: Optional: explicitly specified circuit for the block encoding.
    """

    name: str
    hermitian: bool = True
    hermitian_block_encoding: bool = True
    subnormalization_: float = 1.0
    ancilla_qubits_: int = 0
    qtype_: Type | None = None
    circuit_: Circuit | None = None

    def __post_init__(self):
        if self.hermitian_block_encoding and not self.hermitian:
            raise ValueError(
                "Base expression must be Hermitian if it is a Hermitian block encoding"
            )

    @override
    def optimize(self) -> Expr:
        return self

    @override
    def subnormalization(self) -> float:
        return float(self.subnormalization_)

    @override
    def queries(self) -> int:
        return 1

    @override
    def parallel_queries(self) -> int:
        return self.queries()

    @override
    def ancilla_qubits(self) -> int:
        return int(self.ancilla_qubits_)

    @override
    def is_hermitian(self) -> bool:
        return bool(self.hermitian)

    @override
    def is_hermitian_block_encoding(self) -> bool:
        return bool(self.hermitian_block_encoding)

    @override
    def qtype(self) -> Type:
        if self.qtype_ is not None:
            return self.qtype_
        return BitType(1)

    @override
    def circuit(self) -> Circuit:
        if self.circuit_ is not None:
            return self.circuit_

        return _compile_base(self)

    @override
    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Const(Expr):
    value: complex

    @override
    def optimize(self) -> Expr:
        return self

    @override
    def subnormalization(self) -> float:
        return abs(self.value)

    @override
    def queries(self) -> int:
        return 0

    @override
    def parallel_queries(self) -> int:
        return 0

    @override
    def ancilla_qubits(self) -> int:
        return 0

    @override
    def is_hermitian(self) -> bool:
        return True

    @override
    def is_hermitian_block_encoding(self) -> bool:
        return True

    @override
    def qtype(self) -> Type:
        return BitType(1)

    @override
    def circuit(self) -> Circuit:
        return _compile_const(self)

    @override
    def __str__(self) -> str:
        return "(" + f"{self.value:.12g} · I" + ")"


@dataclass(frozen=True)
class Dagger(Expr):
    expr: Expr

    @override
    def optimize(self) -> Expr:
        e = self.expr.optimize()
        # Push dagger and simplify
        if isinstance(e, Basic) and e.is_hermitian():
            return e
        if isinstance(e, Const):
            return e
        if isinstance(e, Dagger):
            return e.expr
        if isinstance(e, Sum):
            terms = [(coef, Dagger(term).optimize()) for coef, term in e.terms]
            return Sum(terms).optimize()
        if isinstance(e, Prod):
            return Prod(
                list(reversed([Dagger(t).optimize() for t in e.factors]))
            ).optimize()
        if isinstance(e, Tensor):
            return Tensor([Dagger(t).optimize() for t in e.factors]).optimize()
        if isinstance(e, If):
            return If(
                e.cond,
                Dagger(e.then_branch).optimize(),
                Dagger(e.else_branch).optimize(),
            ).optimize()
        if isinstance(e, Poly):
            # For real polynomials, if operand is Hermitian, adjoint is itself
            if e.expr.is_hermitian():
                return e
            # Otherwise push through
            return Poly(Dagger(e.expr).optimize(), e.p).optimize()
        if isinstance(e, Division):
            # Dagger(A/B) = Dagger(A * B^{-1}) = (B^{-1})^† * A^† = (1/B)^† * A^†
            # This is equivalent to: A^† * (B^†)^{-1} = A^† / B^† only when they commute
            # In general: (A/B)^† = (1/Dagger(B)) * Dagger(A)
            return Prod(
                [
                    Division(Const(1.0), Dagger(e.denominator).optimize()),
                    Dagger(e.numerator).optimize(),
                ]
            ).optimize()
        return Dagger(e)

    @override
    def subnormalization(self) -> float:
        return self.expr.subnormalization()

    @override
    def queries(self) -> int:
        return self.expr.queries()

    @override
    def parallel_queries(self) -> int:
        return self.expr.parallel_queries()

    @override
    def ancilla_qubits(self) -> int:
        return self.expr.ancilla_qubits()

    @override
    def is_hermitian(self) -> bool:
        return self.expr.is_hermitian()

    @override
    def is_hermitian_block_encoding(self) -> bool:
        return self.expr.is_hermitian_block_encoding()

    @override
    def qtype(self) -> Type:
        return self.expr.qtype()

    @override
    def circuit(self) -> Circuit:
        return self.expr.circuit().adjoint()

    @override
    def __str__(self) -> str:
        return f"{self.expr}^dagger"


@dataclass(frozen=True)
class Sum(Expr):
    terms: Sequence[tuple[float, Expr]]  # list of (lambda_i, M_i)

    @classmethod
    def of(cls, term: Generator[Expr, None, None] | Expr, *terms: Expr) -> Sum:
        if isinstance(term, Generator):
            if terms:
                raise ValueError(
                    "Sum.of() only accepts one generator or one expression"
                )
            return cls([(1.0, t) for t in term])
        return cls([(1.0, term)] + [(1.0, t) for t in terms])

    @override
    def optimize(self) -> Expr:
        combined = _flatten_and_combine_terms(self.terms)
        if not combined:
            return Const(0.0)
        if len(combined) == 1 and abs(combined[0][0] - 1.0) < TOL:
            return combined[0][1]

        factored = _factor_sum(combined)
        if not isinstance(factored, Sum):
            return factored.optimize()

        merged = _merge_scalars_into_expressions(list(factored.terms))
        merged = _combine_polys_with_same_base(merged)
        merged = _combine_base_expr_with_poly(merged)
        final_result = _merge_const_with_poly(merged)

        if isinstance(final_result, Sum):
            combined_divs = _try_combine_divisions_in_sum(list(final_result.terms))
            if combined_divs is not None:
                return (
                    combined_divs.optimize()
                    if not isinstance(combined_divs, (Basic, Const))
                    else combined_divs
                )

        return final_result

    @override
    def subnormalization(self) -> float:
        return sum(abs(c) * t.subnormalization() for c, t in self.terms)

    @override
    def queries(self) -> int:
        return sum(t.queries() for _, t in self.terms)

    @override
    def parallel_queries(self) -> int:
        vals = [t.parallel_queries() for _, t in self.terms]
        return max(vals) if vals else 0

    @override
    def ancilla_qubits(self) -> int:
        max_n = max((t.ancilla_qubits() for _, t in self.terms), default=0)
        anc = 0 if len(self.terms) <= 1 else math.ceil(math.log2(len(self.terms)))
        return max_n + anc

    @override
    def is_hermitian(self) -> bool:
        return all(t.is_hermitian() for _, t in self.terms)

    @override
    def is_hermitian_block_encoding(self) -> bool:
        # LCU construction for sum is a Hermitian block encoding
        return all(t.is_hermitian_block_encoding() for _, t in self.terms)

    @override
    def qtype(self) -> Type:
        # All terms must have the same type (except Const which is polymorphic)
        if not self.terms:
            raise TypeCheckError("Sum must have at least one term")

        # Collect types, treating Const specially
        common_type: Type | None = None
        for _, term in self.terms:
            t = term.qtype()
            if common_type is None:
                common_type = t
            elif t != common_type:
                raise TypeCheckError(
                    f"Sum terms must have the same type, got {common_type} and {t}"
                )

        assert common_type is not None  # Guaranteed by len(self.terms) > 0
        return common_type

    @override
    def circuit(self) -> Circuit:
        return _compile_sum(self)

    @override
    def __str__(self) -> str:
        parts: list[str] = []
        for i, (coef, term) in enumerate(self.terms):
            s = str(term)
            sign = " + " if coef > 0 and i > 0 else (" - " if coef < 0 else "")
            if abs(abs(coef) - 1.0) < TOL:
                parts.append(sign + s)
            else:
                parts.append(f"{sign}{coef:.12g}·{s}")
        return "(" + "".join(parts) + ")"


@dataclass(frozen=True)
class Prod(Expr):
    factors: Sequence[Expr]
    hermitian: bool = False

    @classmethod
    def of(cls, factor: Generator[Expr, None, None] | Expr, *factors: Expr) -> Prod:
        if isinstance(factor, Generator):
            if factors:
                raise ValueError(
                    "Prod.of() only accepts one generator or one expression"
                )
            return cls([t for t in factor])
        return cls([factor] + [t for t in factors])

    @override
    def optimize(self) -> Expr:
        flats: list[Expr] = []
        for f in self.factors:
            fn = f.optimize()
            if isinstance(fn, Prod):
                flats.extend(fn.factors)
            else:
                flats.append(fn)

        for f in flats:
            if isinstance(f, Const) and abs(f.value) < TOL:
                return Const(0.0)

        fused = _try_poly_product_fusion(flats)
        if fused is not None:
            return fused.optimize()

        # Collapse multiple Ifs with the same condition across the product:
        # (if c then A else B) * (if c then C else D) * ... => if c then A*C*... else B*D*...
        ifs = [f for f in flats if isinstance(f, If)]
        if ifs and all(f.cond == ifs[0].cond for f in ifs):
            cond = ifs[0].cond
            then_terms: list[Expr] = []
            else_terms: list[Expr] = []
            for f in flats:
                if isinstance(f, If):
                    then_terms.append(f.then_branch)
                    else_terms.append(f.else_branch)
                else:
                    then_terms.append(f)
                    else_terms.append(f)
            return If(cond, Prod(then_terms), Prod(else_terms)).optimize()

        # Remove multiplicative identity Const(1)
        flats = [
            f for f in flats if not isinstance(f, Const) or abs(f.value - 1.0) >= TOL
        ]
        if not flats:
            return Const(1.0)
        if len(flats) == 1:
            return flats[0].optimize()

        # If all factors identical: M*M*...*M -> poly(M, x^k)
        head = repr(flats[0])
        if all(repr(f) == head for f in flats[1:]):
            return Poly(flats[0], Polynomial.pow_x(len(flats))).optimize()

        return Prod(flats)

    @override
    def subnormalization(self) -> float:
        a = 1.0
        for f in self.factors:
            a *= f.subnormalization()
        return a

    @override
    def queries(self) -> int:
        return sum(f.queries() for f in self.factors)

    @override
    def parallel_queries(self) -> int:
        return sum(f.parallel_queries() for f in self.factors)

    @override
    def ancilla_qubits(self) -> int:
        max_factor_n = max((f.ancilla_qubits() for f in self.factors), default=0)
        if len(self.factors) <= 1:
            return max_factor_n
        # For N factors (N >= 2), need ceil(log2(N)) flag qubits
        num_flags = math.ceil(math.log2(len(self.factors)))
        return max_factor_n + num_flags

    @override
    def is_hermitian(self) -> bool:
        # if any factor is not Hermitian, then the product is not Hermitian
        if not all(f.is_hermitian() for f in self.factors):
            return False
        # if all factors are identical, then the product is Hermitian
        if all(f == self.factors[0] for f in self.factors):
            return True
        # Otherwise, product is Hermitian only if factors commute; return user-set flag
        return self.hermitian

    @override
    def is_hermitian_block_encoding(self) -> bool:
        # Sequencing of Hermitian block encodings is Hermitian if the encoded matrix is Hermitian
        return self.is_hermitian() and all(
            f.is_hermitian_block_encoding() for f in self.factors
        )

    @override
    def qtype(self) -> Type:
        # All factors must have the same type
        if not self.factors:
            raise TypeCheckError("Prod must have at least one factor")

        common_type: Type | None = None
        for factor in self.factors:
            t = factor.qtype()
            if common_type is None:
                common_type = t
            elif t != common_type:
                raise TypeCheckError(
                    f"Prod factors must have the same type, got {common_type} and {t}"
                )

        assert common_type is not None  # Guaranteed by len(self.factors) > 0
        return common_type

    @override
    def circuit(self) -> Circuit:
        return _compile_prod(self)

    @override
    def __str__(self) -> str:
        return "(" + " * ".join(str(f) for f in self.factors) + ")"


@dataclass(frozen=True)
class Tensor(Expr):
    factors: Sequence[Expr]

    @classmethod
    def of(cls, factor: Generator[Expr, None, None] | Expr, *factors: Expr) -> Tensor:
        if isinstance(factor, Generator):
            if factors:
                raise ValueError(
                    "Tensor.of() only accepts one generator or one expression"
                )
            return cls([t for t in factor])
        return cls([factor] + [t for t in factors])

    @override
    def optimize(self) -> Expr:
        flats: list[Expr] = []
        for f in self.factors:
            fn = f.optimize()
            if isinstance(fn, Tensor):
                flats.extend(fn.factors)
            else:
                flats.append(fn)

        flats2: list[Expr] = []
        const_prod = 1.0
        for f in flats:
            if isinstance(f, Const) and abs(f.value - 1.0) < TOL:
                continue
            if isinstance(f, Const):
                const_prod *= f.value
            else:
                flats2.append(f)

        if abs(const_prod - 1.0) > TOL:
            flats2.insert(0, Const(const_prod))

        if not flats2:
            return Const(1.0)
        if len(flats2) == 1:
            return flats2[0]
        return Tensor(flats2)

    @override
    def subnormalization(self) -> float:
        a = 1.0
        for f in self.factors:
            a *= f.subnormalization()
        return a

    @override
    def queries(self) -> int:
        return sum(f.queries() for f in self.factors)

    @override
    def parallel_queries(self) -> int:
        return max((f.parallel_queries() for f in self.factors), default=0)

    @override
    def ancilla_qubits(self) -> int:
        return sum(f.ancilla_qubits() for f in self.factors)

    @override
    def is_hermitian(self) -> bool:
        return all(f.is_hermitian() for f in self.factors)

    @override
    def is_hermitian_block_encoding(self) -> bool:
        return all(f.is_hermitian_block_encoding() for f in self.factors)

    @override
    def qtype(self) -> Type:
        # Tensor product of factor types
        if not self.factors:
            raise TypeCheckError("Tensor must have at least one factor")

        factor_types = [f.qtype() for f in self.factors]
        return make_tensor_type(*factor_types)

    @override
    def circuit(self) -> Circuit:
        return _compile_tensor(self)

    @override
    def __str__(self) -> str:
        return "(" + " ⊗ ".join(str(f) for f in self.factors) + ")"


@dataclass(frozen=True)
class Condition:
    var: str  # e.g., 'x'
    active: bool = True

    @override
    def __str__(self) -> str:
        return "(" + f"not {self.var}" if not self.active else self.var + ")"


@dataclass(frozen=True)
class If(Expr):
    cond: Condition
    then_branch: Expr
    else_branch: Expr

    @override
    def optimize(self) -> Expr:
        tb = self.then_branch.optimize()
        eb = self.else_branch.optimize()
        c = self.cond

        # Normalize not x by swapping branches
        if not c.active:
            c = Condition(c.var, True)
            tb, eb = eb, tb

        # if x then A else A => I ⊗ A
        if Expr.structural_eq(tb, eb):
            return Const(1.0).kron(tb)

        # Push into operands: if x then (C op P) else (C op Q) => C op (if x then P else Q)
        # Handle for Prod and Tensor and Sum
        if isinstance(tb, Prod) and isinstance(eb, Prod):
            if len(tb.factors) == len(eb.factors) and len(tb.factors) == 2:
                # Left factor equal and right differs
                if Expr.structural_eq(tb.factors[0], eb.factors[0]):
                    return Prod([tb.factors[0], If(c, tb.factors[1], eb.factors[1])])
                if Expr.structural_eq(tb.factors[1], eb.factors[1]):
                    return Prod([If(c, tb.factors[0], eb.factors[0]), tb.factors[1]])
        if isinstance(tb, Tensor) and isinstance(eb, Tensor):
            if len(tb.factors) == len(eb.factors) and len(tb.factors) == 2:
                if Expr.structural_eq(tb.factors[0], eb.factors[0]):
                    return Tensor(
                        [tb.factors[0], If(c, tb.factors[1], eb.factors[1])]
                    ).optimize()
                if Expr.structural_eq(tb.factors[1], eb.factors[1]):
                    return Tensor(
                        [If(c, tb.factors[0], eb.factors[0]), tb.factors[1]]
                    ).optimize()
        return If(c, tb, eb)

    @override
    def subnormalization(self) -> float:
        return self.then_branch.subnormalization()

    @override
    def queries(self) -> int:
        return self.then_branch.queries() + self.else_branch.queries()

    @override
    def parallel_queries(self) -> int:
        return max(
            self.then_branch.parallel_queries(),
            self.else_branch.parallel_queries(),
        )

    @override
    def ancilla_qubits(self) -> int:
        return max(self.then_branch.ancilla_qubits(), self.else_branch.ancilla_qubits())

    @override
    def is_hermitian(self) -> bool:
        return self.then_branch.is_hermitian() and self.else_branch.is_hermitian()

    @override
    def is_hermitian_block_encoding(self) -> bool:
        # Direct sum of Hermitian block encodings is a Hermitian block encoding
        return (
            self.then_branch.is_hermitian_block_encoding()
            and self.else_branch.is_hermitian_block_encoding()
        )

    @override
    def qtype(self) -> Type:
        # Both branches must have the same type t, result is t ⊗ bool
        then_type = self.then_branch.qtype()
        else_type = self.else_branch.qtype()

        if then_type != else_type:
            raise TypeCheckError(
                f"If branches must have the same type, got then: {then_type}, else: {else_type}"
            )

        if self.then_branch.subnormalization() != self.else_branch.subnormalization():
            raise TypeCheckError(
                f"If branches must have the same alpha, got then: {self.then_branch.subnormalization()}, else: {self.else_branch.subnormalization()}"
            )

        return make_tensor_type(then_type, BitType(1))

    @override
    def circuit(self) -> Circuit:
        return _compile_if(self)

    @override
    def __str__(self) -> str:
        return (
            "("
            + f"if {self.cond} then {self.then_branch} else {self.else_branch}"
            + ")"
        )


@dataclass(frozen=True)
class Poly(Expr):
    expr: Expr
    p: Polynomial

    def lcu(self) -> Expr:
        """Convert polynomial to explicit LCU form."""
        p = self.p.trim()
        lcu = [
            (coeff, self.expr**i)
            for i, coeff in enumerate(p.coeffs)
            if abs(coeff) > TOL
        ]
        return Sum(lcu)

    def horner(self) -> Expr:
        """Convert polynomial to Horner's method evaluation."""
        p = self.p.trim()
        e = Const(p.coeffs[0])
        for coeff in p.coeffs[1:]:
            e = e * self.expr
            if abs(coeff) > TOL:
                e += Const(coeff)
        return e

    @override
    def optimize(self) -> Expr:
        e = self.expr.optimize()
        p = self.p.trim()

        # Compose nested poly: poly(poly(A,f), g) => poly(A, g∘f)
        if isinstance(e, Poly):
            return Poly(e.expr, p.compose(e.p)).optimize()

        # Turn constant polynomial into Const
        if p.is_constant() and len(self.p.coeffs) == 1:
            return Const(p.coeffs[0])

        # Evaluate polynomial at constant: poly(Const(c), f) => Const(f(c))
        if isinstance(e, Const):
            if e.value.imag != 0:
                # Complex is not supported
                return self
            val = p.eval(e.value.real)
            return Const(val)

        # Identity polynomial: poly(A, x) => A
        if (
            len(p.coeffs) == 2
            and abs(p.coeffs[0]) < TOL
            and abs(p.coeffs[1] - 1.0) < TOL
        ):
            return e

        return Poly(e, self.p)

    @override
    def subnormalization(self) -> float:
        if os.environ.get("USE_GQSP", "0") == "1":
            return self.scaled_p().gqsp_chebyshev().sup_abs_on_circle(1.0) / (
                1 - QSP_TOL
            )

        if self.p.parity() != Polynomial.Parity.MIXED:
            return self.scaled_p().sup_abs_on_interval(1.0) / (1 - QSP_TOL)
        return (
            Poly(self.expr, self.p.even_component())
            + Poly(self.expr, self.p.odd_component())
        ).subnormalization()

    @override
    def queries(self) -> int:
        if os.environ.get("USE_GQSP", "0") == "1":
            return self.expr.queries() * self.p.degree()

        if self.p.parity() != Polynomial.Parity.MIXED:
            return self.expr.queries() * self.p.degree()
        return (
            Poly(self.expr, self.p.even_component())
            + Poly(self.expr, self.p.odd_component())
        ).queries()

    @override
    def parallel_queries(self) -> int:
        return self.queries()

    @override
    def ancilla_qubits(self) -> int:
        if self.p.parity() != Polynomial.Parity.MIXED:
            return 1 + self.expr.ancilla_qubits()
        return (
            Poly(self.expr, self.p.even_component())
            + Poly(self.expr, self.p.odd_component())
        ).ancilla_qubits()

    @override
    def is_hermitian(self) -> bool:
        return self.expr.is_hermitian()

    @override
    def is_hermitian_block_encoding(self) -> bool:
        return False

    @override
    def qtype(self) -> Type:
        if not self.expr.is_hermitian():
            raise TypeCheckError(
                f"Poly requires Hermitian expression, but {self.expr} is not Hermitian"
            )
        return self.expr.qtype()

    def scaled_p(self) -> Polynomial:
        """Scale polynomial coefficients by the sub-expression alpha.

        For example, for Poly(A, 1 - x^2), if A has alpha=2, scale to [1, 0.0, -4.0].
        """
        a = self.expr.subnormalization()
        return Polynomial([c * a**i for i, c in enumerate(self.p.coeffs)])

    @override
    def circuit(self) -> Circuit:
        return _compile_poly(self)

    @override
    def __str__(self) -> str:
        return "(" + f"Poly({self.expr}, {[round(c, 3) for c in self.p.coeffs]})" + ")"


@dataclass(frozen=True)
class Division(Expr):
    numerator: Expr
    denominator: Expr

    @override
    def optimize(self) -> Expr:
        e1 = self.numerator.optimize()
        e2 = self.denominator.optimize()

        # Division by zero
        if isinstance(e2, Const) and abs(e2.value) < TOL:
            raise ValueError("Division by zero")

        # A / A => Const(1.0)
        if Expr.structural_eq(e1, e2):
            return Const(1.0)

        # Const(c1) / Const(c2) => Const(c1/c2)
        if isinstance(e1, Const) and isinstance(e2, Const):
            return Const(e1.value / e2.value)

        # X / Const(c) => (1/c) * X
        if isinstance(e2, Const):
            if e2.value.imag != 0:
                # Complex is not supported
                return self
            return Sum([(1.0 / e2.value.real, e1)]).optimize()

        # Const(1.0) / (Const(1.0) / A) => A
        if isinstance(e1, Const) and abs(e1.value - 1.0) < TOL:
            if isinstance(e2, Division):
                if (
                    isinstance(e2.numerator, Const)
                    and abs(e2.numerator.value - 1.0) < TOL
                ):
                    return e2.denominator

        # Handle polynomial division
        if isinstance(e1, Poly) and isinstance(e2, Poly):
            # Check if bases are structurally equal
            if Expr.structural_eq(e1.expr, e2.expr):
                # Same base, try polynomial division
                import numpy.polynomial.polynomial as npp

                result = npp.polydiv(list(e1.p.coeffs), list(e2.p.coeffs))
                quotient_coeffs = [float(c) for c in result[0]]
                remainder_coeffs = [float(c) for c in result[1]]

                has_remainder = any(abs(c) >= TOL for c in remainder_coeffs)

                if not has_remainder:
                    q = Polynomial([float(c) for c in quotient_coeffs])
                    return Poly(e1.expr, q).optimize()

        # Handle constant polynomial division: poly(X, [c]) / poly(Y, [d]) where c, d are constants
        if isinstance(e1, Poly) and e1.p.is_constant():
            if isinstance(e2, Poly) and e2.p.is_constant():
                c1 = float(e1.p.coeffs[0])
                c2 = float(e2.p.coeffs[0])
                if abs(c2) >= TOL:
                    return Const(c1 / c2)

        # Try to simplify Sum divisions
        # (A + B) / C where can factor out C from the sum
        if isinstance(e1, Sum):
            # Try to divide each term by e2
            all_divisible = True
            new_terms: list[tuple[float, Expr]] = []

            for coef, term in e1.terms:
                div_result = Division(term, e2).optimize()
                if isinstance(div_result, Division):
                    all_divisible = False
                    break
                new_terms.append((coef, div_result))

            if all_divisible:
                return Sum(new_terms).optimize()

        # Try to simplify Prod divisions
        # (A * B * C) / (A * D) => (B * C) / D
        if isinstance(e1, Prod) and isinstance(e2, Prod):
            # Find common factors
            num_factors = list(e1.factors)
            den_factors = list(e2.factors)

            # Cancel common factors
            for df in den_factors[:]:
                for nf in num_factors[:]:
                    if Expr.structural_eq(nf, df):
                        num_factors.remove(nf)
                        den_factors.remove(df)
                        break

            # If cancelled anything, reconstruct
            if len(num_factors) != len(e1.factors) or len(den_factors) != len(
                e2.factors
            ):
                new_num = Prod(num_factors).optimize() if num_factors else Const(1.0)
                new_den = Prod(den_factors).optimize() if den_factors else Const(1.0)
                return Division(new_num, new_den).optimize()

        # (A * B) / A => B
        if isinstance(e1, Prod):
            remaining_factors = []
            cancelled = False
            for f in e1.factors:
                if not cancelled and Expr.structural_eq(f, e2):
                    cancelled = True
                    continue
                remaining_factors.append(f)

            if cancelled:
                if not remaining_factors:
                    return Const(1.0)
                if len(remaining_factors) == 1:
                    return remaining_factors[0]
                return Prod(remaining_factors).optimize()

        # A / (A * B) => 1 / B
        if isinstance(e2, Prod):
            remaining_factors = []
            cancelled = False
            for f in e2.factors:
                if not cancelled and Expr.structural_eq(f, e1):
                    cancelled = True
                    continue
                remaining_factors.append(f)

            if cancelled:
                if not remaining_factors:
                    return Const(1.0)
                new_den = (
                    Prod(remaining_factors).optimize()
                    if len(remaining_factors) > 1
                    else remaining_factors[0]
                )
                return Division(Const(1.0), new_den).optimize()

        # Try to simplify Tensor divisions by factoring
        # (A ⊗ B) / (C ⊗ D) => (A/C) ⊗ (B/D) if both divide cleanly
        if isinstance(e1, Tensor) and isinstance(e2, Tensor):
            if len(e1.factors) == len(e2.factors):
                all_divisible = True
                new_factors: list[Expr] = []

                for f1, f2 in zip(e1.factors, e2.factors):
                    div_result = Division(f1, f2).optimize()
                    if isinstance(div_result, Division):
                        # Could not simplify
                        all_divisible = False
                        break
                    new_factors.append(div_result)

                if all_divisible:
                    return Tensor(new_factors).optimize()

        # Cannot simplify further
        return Division(e1, e2)

    @override
    def subnormalization(self) -> float:
        return self.numerator.subnormalization() / self.denominator.subnormalization()

    @override
    def queries(self) -> int:
        raise UnresolvedDivisionError(
            f"Cannot determine number of queries for unresolved division: {self}"
        )

    @override
    def parallel_queries(self) -> int:
        raise UnresolvedDivisionError(
            f"Cannot determine parallel queries for unresolved division: {self}"
        )

    @override
    def ancilla_qubits(self) -> int:
        raise UnresolvedDivisionError(
            f"Cannot determine ancilla qubits for unresolved division: {self}"
        )

    @override
    def is_hermitian(self) -> bool:
        return self.numerator.is_hermitian() and self.denominator.is_hermitian()

    @override
    def is_hermitian_block_encoding(self) -> bool:
        raise UnresolvedDivisionError(
            f"Cannot determine if Hermitian block encoding for unresolved division: {self}"
        )

    @override
    def qtype(self) -> Type:
        t1 = self.numerator.qtype()
        t2 = self.denominator.qtype()
        if t1 != t2:
            raise TypeCheckError(
                f"Division operands must have the same type, got {t1} and {t2}"
            )
        return t1

    @override
    def circuit(self) -> Circuit:
        raise UnresolvedDivisionError(
            f"Cannot compile unresolved division to circuit: {self}"
        )

    @override
    def __str__(self) -> str:
        return f"({self.numerator} / {self.denominator})"
