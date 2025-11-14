from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import override


class Type(ABC):
    @abstractmethod
    def width(self) -> int:
        """Return the total number of qubits (bit width) for this type."""
        pass

    @abstractmethod
    @override
    def __eq__(self, other: object) -> bool:
        pass

    @abstractmethod
    @override
    def __hash__(self) -> int:
        pass

    @abstractmethod
    @override
    def __str__(self) -> str:
        pass


@dataclass(frozen=True)
class BitType(Type):
    """Type for a space of n qubits: bit[n]."""

    n: int

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, TensorType):
            return other.width() == self.width()
        if isinstance(other, BitType):
            return other.n == self.n
        return False

    @override
    def width(self) -> int:
        return self.n

    @override
    def __str__(self) -> str:
        if self.n == 1:
            return "bool"
        return f"bit[{self.n}]"


@dataclass(frozen=True)
class TensorType(Type):
    """Type for tensor product of types: t1 ⊗ t2 ⊗ ...

    Note: TensorType is always flattened (no nested TensorTypes).
    """

    factors: tuple[Type, ...]

    def __post_init__(self):
        for f in self.factors:
            if isinstance(f, TensorType):
                raise ValueError("TensorType factors should not contain TensorType")

    @override
    def __eq__(self, other: object) -> bool:
        if isinstance(other, BitType):
            return other.width() == self.width()
        if isinstance(other, TensorType):
            return self.factors == other.factors
        return False

    @override
    def width(self) -> int:
        return sum(f.width() for f in self.factors)

    @override
    def __str__(self) -> str:
        return " ⊗ ".join(str(f) for f in self.factors)


# Helper function to simplify type construction
def make_tensor_type(*types: Type) -> Type:
    """Create a tensor type, flattening nested tensors and simplifying single factors."""
    factors: list[Type] = []
    for t in types:
        if isinstance(t, TensorType):
            factors.extend(t.factors)
        else:
            factors.append(t)

    if len(factors) == 0:
        # Empty tensor defaults to bit[0]
        return BitType(0)
    elif len(factors) == 1:
        return factors[0]
    else:
        return TensorType(tuple(factors))


# Type checking exception
class TypeCheckError(Exception):
    """Raised when a type checking error occurs."""

    pass
