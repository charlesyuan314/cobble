from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, replace
from enum import Enum
import math
from typing import override

from cobble.viz import _to_ascii, _to_tex


class ControlPolarity(Enum):
    """Polarity of a control qubit."""

    POSITIVE = 1  # Control on |1⟩
    NEGATIVE = 0  # Control on |0⟩ (anti-control)


@dataclass(frozen=True)
class Control:
    """Represents a control qubit with explicit polarity.

    Args:
        qubit: The qubit index
        polarity: Whether to control on |1⟩ (POSITIVE) or |0⟩ (NEGATIVE)
    """

    qubit: int
    polarity: ControlPolarity = ControlPolarity.POSITIVE

    @staticmethod
    def pos(qubit: int) -> Control:
        """Create a positive control (control on |1⟩)."""
        return Control(qubit, ControlPolarity.POSITIVE)

    @staticmethod
    def neg(qubit: int) -> Control:
        """Create a negative control (control on |0⟩, anti-control)."""
        return Control(qubit, ControlPolarity.NEGATIVE)

    def is_positive(self) -> bool:
        """Check if this is a positive control."""
        return self.polarity == ControlPolarity.POSITIVE

    def is_negative(self) -> bool:
        """Check if this is a negative control."""
        return self.polarity == ControlPolarity.NEGATIVE


@dataclass(kw_only=True, frozen=True)
class Gate(ABC):
    """Abstract base class for quantum gates."""

    is_conjugate_pair: bool = (
        False  # Whether this is part of a conjugate pair and does not need to be controlled
    )

    @abstractmethod
    def adjoint(self) -> Gate:
        """Return the adjoint (conjugate transpose) of this gate."""

    @abstractmethod
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        """Return a new gate with qubit indices remapped according to mapping.

        Args:
            mapping: Dictionary mapping old qubit indices to new qubit indices.
                    Qubits not in the mapping remain unchanged.

        Returns:
            A new gate with remapped qubit indices.
        """

    @abstractmethod
    def to_flat_gates(self) -> list[FlatGate]:
        """Convert to a flat list of gates (no recursive structures)."""


@dataclass(frozen=True)
class FlatGate:
    """A flat gate with explicit target and control qubits."""

    name: str  # Gate name: NOT, H, Rx, Ry, Rz, SWAP, Z, or custom
    targets: tuple[int, ...]  # Target qubit indices
    controls: tuple[int, ...] = ()  # Control qubit indices (for QASM compatibility)
    params: tuple[float, ...] = ()  # Parameters (e.g., rotation angles)
    control_polarities: tuple[ControlPolarity, ...] = ()  # Polarity for each control
    is_conjugate_pair: bool = (
        False  # Whether this is part of a conjugate pair and does not need to be controlled
    )

    def to_qasm(self) -> str:
        """Convert this flat gate to QASM format.

        Note: This assumes the gate has been decomposed. Use qasm.decompose_for_qasm() first.
        """
        from cobble.qasm import flat_gate_to_qasm

        return flat_gate_to_qasm(self)


@dataclass(frozen=True)
class NOT(Gate):
    """Pauli X gate (NOT)."""

    target: int

    @override
    def adjoint(self) -> Gate:
        return self  # NOT is self-adjoint

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return NOT(
            mapping.get(self.target, self.target),
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate("NOT", (self.target,), is_conjugate_pair=self.is_conjugate_pair)
        ]


@dataclass(frozen=True)
class Hadamard(Gate):
    """Hadamard gate."""

    target: int

    @override
    def adjoint(self) -> Gate:
        return self  # Hadamard is self-adjoint

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return Hadamard(
            mapping.get(self.target, self.target),
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [FlatGate("H", (self.target,), is_conjugate_pair=self.is_conjugate_pair)]


@dataclass(frozen=True)
class Rx(Gate):
    """Rotation around X axis."""

    target: int
    angle: float

    @override
    def adjoint(self) -> Gate:
        return Rx(self.target, -self.angle, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return Rx(
            mapping.get(self.target, self.target),
            self.angle,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate("H", (self.target,), is_conjugate_pair=True),
            FlatGate(
                "Rz",
                (self.target,),
                (),
                (self.angle,),
                is_conjugate_pair=self.is_conjugate_pair,
            ),
            FlatGate("H", (self.target,), is_conjugate_pair=True),
        ]


@dataclass(frozen=True)
class Ry(Gate):
    """Rotation around Y axis."""

    target: int
    angle: float

    @override
    def adjoint(self) -> Gate:
        return Ry(self.target, -self.angle, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return Ry(
            mapping.get(self.target, self.target),
            self.angle,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate(
                "Ry",
                (self.target,),
                (),
                (self.angle,),
                is_conjugate_pair=self.is_conjugate_pair,
            )
        ]


@dataclass(frozen=True)
class Rz(Gate):
    """Rotation around Z axis."""

    target: int
    angle: float

    @override
    def adjoint(self) -> Gate:
        return Rz(self.target, -self.angle, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return Rz(
            mapping.get(self.target, self.target),
            self.angle,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate(
                "Rz",
                (self.target,),
                (),
                (self.angle,),
                is_conjugate_pair=self.is_conjugate_pair,
            )
        ]


@dataclass(frozen=True)
class SU2Rotation(Gate):
    """SU(2) rotation gate."""

    target: int
    theta: float
    phi: float
    lbd: float

    @override
    def adjoint(self) -> Gate:
        return SU2Rotation(
            self.target,
            -self.theta,
            -self.phi,
            -self.lbd,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return SU2Rotation(
            mapping.get(self.target, self.target),
            self.theta,
            self.phi,
            self.lbd,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate(
                "SU2Rotation",
                (self.target,),
                (),
                (self.theta, self.phi, self.lbd),
                is_conjugate_pair=self.is_conjugate_pair,
            )
        ]


@dataclass(frozen=True)
class Z(Gate):
    """Pauli Z gate."""

    target: int

    @override
    def adjoint(self) -> Gate:
        return self  # Z is self-adjoint

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return Z(
            mapping.get(self.target, self.target),
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [FlatGate("Z", (self.target,), is_conjugate_pair=self.is_conjugate_pair)]


@dataclass(frozen=True)
class GlobalPhase(Gate):
    """Global phase gate."""

    target: int
    phase: float

    @override
    def adjoint(self) -> Gate:
        return GlobalPhase(
            self.target, -self.phase, is_conjugate_pair=self.is_conjugate_pair
        )

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return GlobalPhase(
            mapping.get(self.target, self.target),
            self.phase,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate(
                "GlobalPhase",
                (self.target,),
                (),
                (self.phase,),
                is_conjugate_pair=self.is_conjugate_pair,
            )
        ]


@dataclass(frozen=True)
class SWAP(Gate):
    """SWAP gate."""

    target1: int
    target2: int

    @override
    def adjoint(self) -> Gate:
        return self  # SWAP is self-adjoint

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        return SWAP(
            mapping.get(self.target1, self.target1),
            mapping.get(self.target2, self.target2),
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        return [
            FlatGate(
                "SWAP",
                (self.target1, self.target2),
                is_conjugate_pair=self.is_conjugate_pair,
            )
        ]


@dataclass(frozen=True)
class Increment(Gate):
    """Increment gate for n-bit unsigned arithmetic.

    Increments an n-bit unsigned integer stored in little-endian format.
    targets[0] is the least significant bit (2^0), targets[-1] is the most significant bit.
    """

    targets: tuple[int, ...]  # Qubit indices in little-endian order

    @override
    def adjoint(self) -> Gate:
        # Decrement is the adjoint of increment
        # For now, just reverse the gate sequence in to_flat_gates
        return Decrement(self.targets, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        remapped = tuple(mapping.get(t, t) for t in self.targets)
        return Increment(remapped, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        """Lower to multi-controlled NOT gates using ripple-carry addition.

        For n qubits (little-endian): targets = (q0, q1, ..., q_{n-1})
        Increment uses ripple-carry from MSB to LSB to avoid conflicts:
        - First, propagate carries from higher to lower bits (MSB to LSB)
        - Finally, flip the LSB

        For n=2: CX(q1, ctrl=q0), X(q0)
        For n=3: CCX(q2, ctrl=[q0,q1]), CX(q1, ctrl=q0), X(q0)
        """
        n = len(self.targets)
        if n == 0:
            return []

        gates: list[FlatGate] = []

        # Process from MSB to LSB: propagate carries before flipping LSB
        for i in range(n - 1, 0, -1):
            # Flip bit i if all lower bits are 1 (carry propagation)
            controls = tuple(self.targets[j] for j in range(i))
            gates.append(
                FlatGate(
                    "NOT",
                    (self.targets[i],),
                    controls,
                    (),
                    tuple(ControlPolarity.POSITIVE for _ in controls),
                    is_conjugate_pair=self.is_conjugate_pair,
                )
            )

        # Finally, flip the least significant bit
        gates.append(
            FlatGate(
                "NOT", (self.targets[0],), is_conjugate_pair=self.is_conjugate_pair
            )
        )

        return gates


@dataclass(frozen=True)
class Decrement(Gate):
    """Decrement gate for n-bit unsigned arithmetic (adjoint of Increment)."""

    targets: tuple[int, ...]

    @override
    def adjoint(self) -> Gate:
        return Increment(self.targets, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        remapped = tuple(mapping.get(t, t) for t in self.targets)
        return Decrement(remapped, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        """Lower to reversed increment gates."""
        inc = Increment(self.targets, is_conjugate_pair=self.is_conjugate_pair)
        return list(reversed(inc.to_flat_gates()))


@dataclass(frozen=True)
class XorInt(Gate):
    """XOR an integer constant into an n-bit register.

    Flips the bits corresponding to 1s in the binary representation of value.
    targets are in little-endian order: targets[0] is the 2^0 bit.
    """

    targets: tuple[int, ...]  # Qubit indices in little-endian order
    value: int  # Integer value to XOR

    @override
    def adjoint(self) -> Gate:
        # XOR is self-adjoint
        return self

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        remapped = tuple(mapping.get(t, t) for t in self.targets)
        return XorInt(remapped, self.value, is_conjugate_pair=self.is_conjugate_pair)

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        """Lower to NOT gates on bits where value has a 1."""
        n = len(self.targets)
        gates: list[FlatGate] = []

        for i in range(n):
            # Check if bit i is set in value
            if (self.value >> i) & 1:
                gates.append(
                    FlatGate(
                        "NOT",
                        (self.targets[i],),
                        is_conjugate_pair=self.is_conjugate_pair,
                    )
                )

        return gates


@dataclass(frozen=True)
class Controlled(Gate):
    """Controlled gate (arbitrary controls).

    Controls can be positive (control on |1⟩) or negative (control on |0⟩).
    Use Control objects for explicit polarity specification.

    For positive controls, can use integers directly. For anti-controls,
    must use Control.neg() explicitly.

    Examples:
        Controlled(gate, (Control.pos(0), Control.neg(1)))  # Explicit
        Controlled(gate, (0, Control.neg(1)))               # Mixed (OK)
        Controlled(gate, (0, 1))                            # All positive (OK)

    Note: After initialization, controls are always normalized to Control objects.
    """

    gate: Gate
    controls: tuple[Control, ...]  # After __post_init__, always Control objects

    def __init__(
        self,
        gate: Gate,
        controls: tuple[Control | int, ...],
        is_conjugate_pair: bool = False,
    ):
        """Initialize with optional int controls that are normalized to Control objects."""
        # Convert to Control objects
        normalized: list[Control] = []
        for c in controls:
            if isinstance(c, Control):
                normalized.append(c)
            elif isinstance(c, int):
                if c < 0:
                    raise ValueError(
                        f"Negative control index {c} not allowed. "
                        f"Use Control.neg({-c}) for anti-control on qubit {-c}."
                    )
                # Positive integer: positive control
                normalized.append(Control.pos(c))
            else:
                raise TypeError(f"Control must be Control or int, got {type(c)}")

        # Use object.__setattr__ since dataclass is frozen
        object.__setattr__(self, "gate", gate)
        object.__setattr__(self, "controls", tuple(normalized))
        object.__setattr__(self, "is_conjugate_pair", is_conjugate_pair)

    @staticmethod
    def add_controls(gate: Gate, new_controls: tuple[Control | int, ...]) -> Gate:
        """Add controls to a gate, merging with existing controls if already a Controlled gate.

        This avoids creating nested Controlled gates, which are semantically equivalent but
        less efficient than a single Controlled gate with merged controls.

        Args:
            gate: The gate to add controls to
            new_controls: The control qubits to add (Control objects or ints)

        Returns:
            A Controlled gate with all controls merged
        """
        if gate.is_conjugate_pair:
            return gate
        if not new_controls:
            return gate
        if isinstance(gate, Controlled):
            # Merge new controls with existing controls
            merged_controls = new_controls + gate.controls
            return Controlled(
                gate.gate, merged_controls, is_conjugate_pair=gate.is_conjugate_pair
            )
        else:
            # Create new Controlled gate
            return Controlled(
                gate, new_controls, is_conjugate_pair=gate.is_conjugate_pair
            )

    @override
    def adjoint(self) -> Gate:
        return Controlled(
            self.gate.adjoint(), self.controls, is_conjugate_pair=self.is_conjugate_pair
        )

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        remapped_gate = self.gate.remap_qubits(mapping)
        remapped_controls = tuple(
            Control(mapping.get(c.qubit, c.qubit), c.polarity) for c in self.controls
        )
        return Controlled(
            remapped_gate, remapped_controls, is_conjugate_pair=self.is_conjugate_pair
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        """Flatten by adding controls to the inner gate's flat representation.

        Negative controls are preserved with polarity information for visualization,
        but also lowered to NOT-control-NOT patterns for execution.
        """
        inner_flat = self.gate.to_flat_gates()
        result: list[FlatGate] = []

        # Separate positive and negative controls
        pos_controls = tuple(c.qubit for c in self.controls if c.is_positive())
        neg_controls = tuple(c.qubit for c in self.controls if c.is_negative())

        # Add NOTs for negative controls
        for qubit in neg_controls:
            result.append(FlatGate("NOT", (qubit,), is_conjugate_pair=True))

        # Add the controlled gates with polarity preserved
        for fg in inner_flat:
            # Combine all controls
            all_controls = pos_controls + neg_controls + fg.controls
            # Build polarity tuple: positive for pos_controls and existing controls, negative for neg_controls
            all_polarities = (
                tuple(ControlPolarity.POSITIVE for _ in pos_controls)
                + tuple(ControlPolarity.NEGATIVE for _ in neg_controls)
                + fg.control_polarities
            )
            result.append(
                FlatGate(
                    fg.name,
                    fg.targets,
                    all_controls,
                    fg.params,
                    all_polarities,
                    is_conjugate_pair=self.is_conjugate_pair,
                )
            )

        # Undo NOTs for negative controls
        for qubit in reversed(neg_controls):
            result.append(FlatGate("NOT", (qubit,), is_conjugate_pair=True))

        return result


@dataclass(frozen=True)
class BlackBox(Gate):
    """Black box oracle gate with name and dimensions.

    The gate acts on explicit data and ancilla target qubits, which do not need to be contiguous.
    """

    name: str
    data_qubits: int
    ancilla_qubits: int = 0
    start_qubit: int = (
        0  # Starting qubit index for this black box (used for initial construction)
    )
    data_targets: tuple[int, ...] | None = None  # Explicit data qubit targets
    ancilla_targets: tuple[int, ...] | None = None  # Explicit ancilla qubit targets

    def __post_init__(self) -> None:
        # If explicit targets not provided, use contiguous range from start_qubit
        if self.data_targets is None:
            object.__setattr__(
                self,
                "data_targets",
                tuple(range(self.start_qubit, self.start_qubit + self.data_qubits)),
            )
        if self.ancilla_targets is None:
            object.__setattr__(
                self,
                "ancilla_targets",
                tuple(
                    range(
                        self.start_qubit + self.data_qubits,
                        self.start_qubit + self.data_qubits + self.ancilla_qubits,
                    )
                ),
            )

    @override
    def adjoint(self) -> Gate:
        if self.name.endswith("_dag"):
            name = self.name[:-4]
        else:
            name = self.name + "_dag"
        return BlackBox(
            name,
            self.data_qubits,
            self.ancilla_qubits,
            self.start_qubit,
            self.data_targets,
            self.ancilla_targets,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def remap_qubits(self, mapping: dict[int, int]) -> Gate:
        # Remap each target qubit individually
        assert self.data_targets is not None
        assert self.ancilla_targets is not None
        new_data_targets = tuple(mapping.get(q, q) for q in self.data_targets)
        new_ancilla_targets = tuple(mapping.get(q, q) for q in self.ancilla_targets)

        # Set start_qubit to the minimum of all targets for consistency
        all_targets = new_data_targets + new_ancilla_targets
        new_start_qubit = min(all_targets) if all_targets else 0

        return BlackBox(
            self.name,
            self.data_qubits,
            self.ancilla_qubits,
            new_start_qubit,
            new_data_targets,
            new_ancilla_targets,
            is_conjugate_pair=self.is_conjugate_pair,
        )

    @override
    def to_flat_gates(self) -> list[FlatGate]:
        # Black box gates are already flat - combine data and ancilla targets
        assert self.data_targets is not None
        assert self.ancilla_targets is not None
        targets = self.data_targets + self.ancilla_targets
        return [FlatGate(self.name, targets, is_conjugate_pair=self.is_conjugate_pair)]


@dataclass(frozen=True)
class QubitAllocation:
    """Tracks qubit allocation for circuit compilation."""

    data_qubits: int  # Number of data qubits (starting from 0)
    ancilla_qubits: list[int]  # List of ancilla qubit indices

    def total_qubits(self) -> int:
        """Total number of qubits."""
        if not self.ancilla_qubits:
            return self.data_qubits
        return max(self.data_qubits, max(self.ancilla_qubits) + 1)

    def allocate_ancillas(self, count: int) -> tuple[QubitAllocation, list[int]]:
        """Allocate new ancilla qubits and return updated allocation and new indices."""
        start = self.total_qubits()
        new_ancillas = list(range(start, start + count))
        return (
            QubitAllocation(self.data_qubits, self.ancilla_qubits + new_ancillas),
            new_ancillas,
        )


@dataclass(frozen=True)
class Circuit:
    """A quantum circuit, represented as a sequence of gates.

    Attributes:
        gates: Sequence of gates in the circuit
        allocation: Qubit allocation information
    """

    gates: Sequence[Gate]
    allocation: QubitAllocation

    @property
    def data_qubits(self) -> int:
        """Number of data qubits."""
        return self.allocation.data_qubits

    @property
    def ancilla_qubits(self) -> int:
        """Number of ancilla qubits."""
        return len(self.allocation.ancilla_qubits)

    def adjoint(self) -> Circuit:
        """Return the adjoint (reverse and conjugate) of the circuit."""
        return Circuit(
            [g.adjoint() for g in reversed(self.gates)],
            self.allocation,
        )

    def to_list(self) -> list[FlatGate]:
        """Convert to a flat list of gates."""
        result: list[FlatGate] = []
        for gate in self.gates:
            result.extend(gate.to_flat_gates())
        return result

    def to_qasm(self, seed: int = 42) -> str:
        """Convert circuit to QASM format with automatic decomposition.

        Args:
            seed: Random seed for "$" gate (random rotation)

        Note: Gates are automatically decomposed to qelib1.inc-compatible format.
              Ancillas are allocated as needed for multi-controlled gates.
        """
        from cobble.qasm import circuit_to_qasm

        return circuit_to_qasm(self, seed)

    def to_ascii(self, max_width: int = 120) -> str:
        """Generate an ASCII diagram of the quantum circuit.

        Args:
            max_width: Maximum width of the diagram before wrapping

        Returns:
            ASCII representation of the circuit
        """
        return _to_ascii(self, max_width)

    def compose(self, other: Circuit) -> Circuit:
        """Compose two circuits in sequence (self then other).

        Both circuits must have compatible allocations.
        """
        if self.allocation.data_qubits != other.allocation.data_qubits:
            msg = (
                f"Cannot compose circuits with different data qubit counts: "
                f"{self.allocation.data_qubits} vs {other.allocation.data_qubits}"
            )
            raise ValueError(msg)
        if len(self.allocation.ancilla_qubits) != len(other.allocation.ancilla_qubits):
            msg = (
                f"Cannot compose circuits with different ancilla qubit counts: "
                f"{len(self.allocation.ancilla_qubits)} vs {len(other.allocation.ancilla_qubits)}"
            )
            raise ValueError(msg)

        return Circuit(
            list(self.gates) + list(other.gates),
            self.allocation,
        )

    def to_tex(self) -> str:
        """Generate a TeX file using the quantikz2 package to visualize the circuit."""
        return _to_tex(self)

    @staticmethod
    def identity(data_qubits: int, ancilla_count: int = 0) -> Circuit:
        """Create an identity circuit (no gates)."""
        ancillas = list(range(data_qubits, data_qubits + ancilla_count))
        return Circuit([], QubitAllocation(data_qubits, ancillas))

    def remap_qubits(
        self,
        ancilla_target: list[int],
        data_offset: int = 0,
        data_mapping: dict[int, int] | None = None,
    ) -> Circuit:
        """Remap this circuit's qubits to new positions.

        Common pattern: sub-circuit assumes qubits start at 0,
        but need to insert it into a larger circuit with different qubit indices.

        Args:
            ancilla_target: List of qubit indices where this circuit's ancillas should go
            data_offset: Offset to add to all data qubit indices (default 0, ignored if data_mapping provided)
            data_mapping: Optional explicit mapping for data qubits (overrides data_offset)

        Returns:
            Circuit with remapped qubit indices
        """
        qubit_map: dict[int, int] = {}

        if data_mapping is not None:
            qubit_map.update(data_mapping)
        else:
            for i in range(self.data_qubits):
                qubit_map[i] = data_offset + i

        circ_ancilla_indices = self.allocation.ancilla_qubits
        for i, anc_idx in enumerate(circ_ancilla_indices):
            if i < len(ancilla_target):
                qubit_map[anc_idx] = ancilla_target[i]

        remapped_gates: list[Gate] = []
        for gate in self.gates:
            remapped_gate = gate.remap_qubits(qubit_map) if qubit_map else gate
            remapped_gates.append(remapped_gate)

        remapped_data_indices = [qubit_map.get(i, i) for i in range(self.data_qubits)]
        new_data_qubits = max(remapped_data_indices) + 1 if remapped_data_indices else 0

        new_ancilla_qubits = [qubit_map.get(anc, anc) for anc in circ_ancilla_indices]

        new_allocation = QubitAllocation(new_data_qubits, new_ancilla_qubits)
        return Circuit(remapped_gates, new_allocation)


def state_preparation_tree(
    coefficients: Sequence[float], ancilla_qubits: list[int]
) -> list[Gate]:
    """Create state preparation circuit for LCU using a tree of Ry rotations.

    Prepares the state sum_i sqrt(|lambda_i|/sum|lambda_j|) |i> on the ancilla register.

    Args:
        coefficients: The lambda coefficients (can be negative)
        ancilla_qubits: Specific ancilla qubit indices to use

    Returns:
        List of gates for state preparation
    """
    n = len(coefficients)
    if n <= 1:
        return []

    abs_coeffs = [abs(c) for c in coefficients]
    total = sum(abs_coeffs)
    if total < 1e-12:
        return []

    probs = [c / total for c in abs_coeffs]

    num_ancillas = math.ceil(math.log2(n))
    if len(ancilla_qubits) < num_ancillas:
        raise ValueError(
            f"Need {num_ancillas} ancillas but only {len(ancilla_qubits)} provided"
        )

    padded_size = 2**num_ancillas
    if len(probs) < padded_size:
        probs = probs + [0.0] * (padded_size - len(probs))

    gates: list[Gate] = []

    # Optimization: uniform distribution uses Hadamards
    if all(abs(p - probs[0]) < 1e-12 for p in probs[:n]):
        for i in range(num_ancillas):
            gates.append(Hadamard(ancilla_qubits[i]))
        return gates

    def prepare_recursive(
        probs: list[float], depth: int, controls: list[Control | int]
    ) -> None:
        """Recursively prepare state using binary tree of controlled rotations."""
        if depth >= num_ancillas or len(probs) <= 1:
            return

        # Split probabilities in half
        mid = len(probs) // 2
        left_probs = probs[:mid]
        right_probs = probs[mid:]

        left_sum = sum(left_probs)
        right_sum = sum(right_probs)
        total_sum = left_sum + right_sum

        if total_sum < 1e-12:
            return

        # Calculate rotation angle
        prob_left = max(0.0, min(1.0, left_sum / total_sum))
        theta = 2 * math.acos(math.sqrt(prob_left))

        # Apply (controlled) rotation
        gate = Ry(ancilla_qubits[depth], theta)
        gates.append(Controlled.add_controls(gate, tuple(controls)))

        # Recurse on left (|0⟩) and right (|1⟩) branches
        prepare_recursive(
            left_probs, depth + 1, controls + [Control.neg(ancilla_qubits[depth])]
        )
        prepare_recursive(right_probs, depth + 1, controls + [ancilla_qubits[depth]])

    prepare_recursive(probs, 0, [])

    return gates


def sign_correction(
    coefficients: Sequence[float], ancilla_qubits: list[int]
) -> list[Gate]:
    """Add sign correction for negative coefficients in LCU.

    For each negative coefficient at index i, apply a Z gate to an ancilla qubit
    controlled on the ancilla register being in state |i⟩.

    Args:
        coefficients: The lambda coefficients
        ancilla_qubits: Specific ancilla qubit indices

    Returns:
        List of gates for sign correction
    """
    n = len(coefficients)
    if n <= 1:
        return []

    num_ancillas = math.ceil(math.log2(n))
    if len(ancilla_qubits) < num_ancillas:
        raise ValueError(
            f"Need {num_ancillas} ancillas but only {len(ancilla_qubits)} provided"
        )

    gates: list[Gate] = []

    for i, coeff in enumerate(coefficients):
        if coeff < 0:
            binary = format(i, f"0{num_ancillas}b")

            # Find first bit that's 1 (or use bit 0 if all are 0)
            target_bit_idx = next(
                (idx for idx, bit in enumerate(binary) if bit == "1"), 0
            )
            target_qubit = ancilla_qubits[target_bit_idx]

            # Build controls for all other bits matching their values in |i⟩
            controls: list[Control | int] = []
            for bit_idx, bit_val in enumerate(binary):
                if bit_idx == target_bit_idx:
                    continue
                qubit = ancilla_qubits[bit_idx]
                controls.append(qubit if bit_val == "1" else Control.neg(qubit))

            # For |00...0⟩, flip target before/after Z
            if i == 0:
                gates.append(
                    replace(
                        Controlled.add_controls(NOT(target_qubit), tuple(controls)),
                        is_conjugate_pair=True,
                    )
                )
                gates.append(Controlled.add_controls(Z(target_qubit), tuple(controls)))
                gates.append(
                    replace(
                        Controlled.add_controls(NOT(target_qubit), tuple(controls)),
                        is_conjugate_pair=True,
                    )
                )
            else:
                # Apply Z controlled on ancilla = |i⟩
                gates.append(Controlled.add_controls(Z(target_qubit), tuple(controls)))

    return gates
