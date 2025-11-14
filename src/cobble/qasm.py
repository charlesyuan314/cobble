from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
import math
import random

from cobble.circuit import Circuit, ControlPolarity, FlatGate


@dataclass
class QASMContext:
    """Context for QASM generation, tracking ancilla allocation."""

    total_qubits: int
    allocated_ancillas: list[int] = field(default_factory=list)
    free_ancillas: list[int] = field(default_factory=list)

    def allocate_ancilla(self) -> int:
        """Allocate an ancilla qubit, reusing freed ones if available."""
        if self.free_ancillas:
            return self.free_ancillas.pop()

        ancilla = self.total_qubits + len(self.allocated_ancillas)
        self.allocated_ancillas.append(ancilla)
        return ancilla

    def free_ancilla(self, ancilla: int) -> None:
        """Mark an ancilla as free for reuse."""
        if ancilla not in self.free_ancillas:
            self.free_ancillas.append(ancilla)

    def get_total_qubits(self) -> int:
        """Get total number of qubits including allocated ancillas."""
        return self.total_qubits + len(self.allocated_ancillas)


def decompose_for_qasm(
    gate: FlatGate, seed: int = 42, ctx: QASMContext | None = None
) -> list[FlatGate]:
    """Decompose a gate into standard gates.

    Args:
        gate: The gate to decompose
        seed: Random seed for gate generation
        ctx: Context for ancilla allocation

    Returns:
        List of standard gates
    """
    if gate.name.startswith("$"):
        return _decompose_random_gate(gate, seed)
    num_controls = len(gate.controls)
    if num_controls == 0:
        return _decompose_uncontrolled_gate(gate)
    if num_controls == 1:
        return _decompose_single_controlled(gate)
    return _decompose_multi_controlled(gate, seed, ctx)


def _decompose_random_gate(gate: FlatGate, seed: int) -> list[FlatGate]:
    """Decompose the special $ gate into random (seeded) u3 rotations."""
    result = []
    for target_qubit in gate.targets:
        # Create deterministic random parameters based on seed and qubit index
        combined_seed = f"{seed}_{target_qubit}_{gate.name}"
        hash_obj = hashlib.sha256(combined_seed.encode())
        hash_int = int.from_bytes(hash_obj.digest()[:8], byteorder="big")
        rng = random.Random(hash_int)
        theta = 0.42424242
        phi = rng.uniform(0, 2 * math.pi)
        lam = rng.uniform(0, 2 * math.pi)
        result.extend(_decompose_u3(theta, phi, lam, target_qubit))
    return result


def _decompose_uncontrolled_gate(gate: FlatGate) -> list[FlatGate]:
    """Decompose an uncontrolled gate into standard gates."""
    if gate.name == "GlobalPhase":
        return []
    if gate.name == "SWAP":
        return _decompose_swap(gate)
    if gate.name == "Ry":
        return _decompose_ry(gate.params[0], gate.targets[0])
    if gate.name == "SU2Rotation":
        return _decompose_u3(
            gate.params[0], gate.params[1], gate.params[2], gate.targets[0]
        )
    return [gate]


def _add_control(
    gate: FlatGate, control: int, control_polarity: ControlPolarity
) -> FlatGate:
    """Add a control to a gate."""
    if gate.is_conjugate_pair:
        return gate
    return FlatGate(
        gate.name,
        gate.targets,
        gate.controls + (control,),
        gate.params,
        gate.control_polarities + (control_polarity,),
    )


def _decompose_single_controlled(gate: FlatGate) -> list[FlatGate]:
    """Decompose single-controlled gates to standard gates."""

    control = gate.controls[0]
    target = gate.targets[0]
    control_polarity = (
        gate.control_polarities[0]
        if gate.control_polarities
        else ControlPolarity.POSITIVE
    )

    if gate.name == "H":
        return _decompose_ch(control, target, control_polarity)

    if gate.name == "Z":
        return [
            FlatGate("H", (target,), is_conjugate_pair=True),
            FlatGate("NOT", (target,), (control,), (), (control_polarity,)),
            FlatGate("H", (target,), is_conjugate_pair=True),
        ]

    if gate.name == "SWAP":
        return [
            _add_control(g, control, control_polarity) for g in _decompose_swap(gate)
        ]

    elif gate.name == "Rx":
        # Decompose CRx(theta) = H, CRz(theta), H
        return [
            FlatGate("H", (target,), is_conjugate_pair=True),
            *_decompose_crz(gate.params[0], control, target, control_polarity),
            FlatGate("H", (target,), is_conjugate_pair=True),
        ]

    elif gate.name == "Ry":
        # Decompose CRy(theta) = S†, CRx(theta), S
        return [
            FlatGate("Rz", (target,), (), (-math.pi / 2,), is_conjugate_pair=True),
            FlatGate("H", (target,), is_conjugate_pair=True),
            *_decompose_crz(gate.params[0], control, target, control_polarity),
            FlatGate("H", (target,), is_conjugate_pair=True),
            FlatGate("Rz", (target,), (), (math.pi / 2,), is_conjugate_pair=True),
        ]

    if gate.name == "Rz" or gate.name == "GlobalPhase":
        return _decompose_crz(
            gate.params[0],
            control,
            target,
            control_polarity,
        )

    elif gate.name == "SU2Rotation":
        return _decompose_cu3(
            gate.params[0],
            gate.params[1],
            gate.params[2],
            control,
            target,
            control_polarity,
        )

    return [gate]


def _decompose_multi_controlled(
    gate: FlatGate, seed: int, ctx: QASMContext | None
) -> list[FlatGate]:
    """Decompose multi-controlled gate: AND controls into ancilla, apply single-controlled U, uncompute."""
    if ctx is None:
        raise ValueError("Context is required to decompose multi-controlled gates.")

    num_controls = len(gate.controls)
    if num_controls < 2:
        return [gate]

    if num_controls == 2 and gate.name == "NOT":
        return [
            FlatGate("ccx", gate.targets, gate.controls, (), gate.control_polarities)
        ]

    if num_controls == 2:
        return _decompose_two_controlled(gate, ctx)

    return _decompose_three_plus_controlled(gate, seed, ctx)


def _decompose_two_controlled(gate: FlatGate, ctx: QASMContext) -> list[FlatGate]:
    """Decompose 2-controlled gate using AND ancilla + single-controlled U."""
    target = gate.targets[0] if len(gate.targets) == 1 else gate.targets
    controls = list(gate.controls)
    polarities = (
        list(gate.control_polarities)
        if gate.control_polarities
        else [ControlPolarity.POSITIVE] * len(controls)
    )

    ancilla = ctx.allocate_ancilla()

    compute_and = FlatGate(
        "ccx",
        (ancilla,),
        tuple(controls),
        (),
        tuple(polarities),
        is_conjugate_pair=True,
    )

    if isinstance(target, tuple):
        controlled_gate = FlatGate(
            gate.name, target, (ancilla,), gate.params, (ControlPolarity.POSITIVE,)
        )
    else:
        controlled_gate = FlatGate(
            gate.name,
            (target,),
            (ancilla,),
            gate.params,
            (ControlPolarity.POSITIVE,),
        )
    apply_u = _decompose_single_controlled(controlled_gate)

    uncompute_and = FlatGate(
        "ccx",
        (ancilla,),
        tuple(controls),
        (),
        tuple(polarities),
        is_conjugate_pair=True,
    )

    ctx.free_ancilla(ancilla)
    return [compute_and] + apply_u + [uncompute_and]


def _decompose_three_plus_controlled(
    gate: FlatGate, seed: int, ctx: QASMContext
) -> list[FlatGate]:
    """Decompose 3+ controlled gate recursively: AND first n-1 controls, then 2-controlled U."""
    target = gate.targets[0] if len(gate.targets) == 1 else gate.targets
    controls = list(gate.controls)
    polarities = (
        list(gate.control_polarities)
        if gate.control_polarities
        else [ControlPolarity.POSITIVE] * len(controls)
    )

    ancilla = ctx.allocate_ancilla()

    and_gate = FlatGate(
        "NOT", (ancilla,), tuple(controls[:-1]), (), tuple(polarities[:-1])
    )
    compute_and = _decompose_multi_controlled(and_gate, seed, ctx)

    if isinstance(target, tuple):
        controlled_gate = FlatGate(
            gate.name,
            target,
            (ancilla, controls[-1]),
            gate.params,
            (ControlPolarity.POSITIVE, polarities[-1]),
        )
    else:
        controlled_gate = FlatGate(
            gate.name,
            (target,),
            (ancilla, controls[-1]),
            gate.params,
            (ControlPolarity.POSITIVE, polarities[-1]),
        )

    apply_u = _decompose_multi_controlled(controlled_gate, seed, ctx)
    uncompute_and = list(reversed(compute_and))

    ctx.free_ancilla(ancilla)
    return compute_and + apply_u + uncompute_and


def _decompose_swap(gate: FlatGate) -> list[FlatGate]:
    """Decompose SWAP into three CNOT gates."""

    a, b = gate.targets
    return [
        FlatGate("NOT", (b,), (a,), is_conjugate_pair=True),
        FlatGate("NOT", (a,), (b,)),
        FlatGate("NOT", (b,), (a,), is_conjugate_pair=True),
    ]


def _decompose_ry(theta: float, target: int) -> list[FlatGate]:
    """Decompose Ry(theta) = Rz(-pi/2) Rx(theta) Rz(pi/2)."""
    return [
        FlatGate("Rz", (target,), (), (-math.pi / 2,), is_conjugate_pair=True),
        FlatGate("H", (target,), is_conjugate_pair=True),
        FlatGate("Rz", (target,), (), (theta,)),
        FlatGate("H", (target,), is_conjugate_pair=True),
        FlatGate("Rz", (target,), (), (math.pi / 2,), is_conjugate_pair=True),
    ]


def _decompose_u3(theta: float, phi: float, lam: float, target: int) -> list[FlatGate]:
    """Decompose U3(theta, phi, lambda) = Rz(lambda - pi/2), Rx(theta), Rz(phi + pi/2)."""
    return [
        FlatGate("Rz", (target,), (), (lam - math.pi / 2,)),
        FlatGate("H", (target,), is_conjugate_pair=True),
        FlatGate("Rz", (target,), (), (theta,)),
        FlatGate("H", (target,), is_conjugate_pair=True),
        FlatGate("Rz", (target,), (), (phi + math.pi / 2,)),
    ]


def _decompose_crz(
    theta: float, control: int, target: int, control_polarity: ControlPolarity
) -> list[FlatGate]:
    """Decompose CRz(theta) = CNOT, Rz(-theta) on target, CNOT, Rz(theta) on target."""
    return [
        FlatGate("NOT", (target,), (control,), (), (control_polarity,)),
        FlatGate("Rz", (target,), (), (-theta,), is_conjugate_pair=True),
        FlatGate("NOT", (target,), (control,), (), (control_polarity,)),
        FlatGate("Rz", (target,), (), (theta,), is_conjugate_pair=True),
    ]


def _decompose_cu3(
    theta: float,
    phi: float,
    lam: float,
    control: int,
    target: int,
    control_polarity: ControlPolarity,
) -> list[FlatGate]:
    """Decompose CU3 using CRz and CRx (where CRx is H-conjugated CRz)."""
    result = []
    for g in _decompose_u3(theta, phi, lam, target):
        g = _add_control(g, control, control_polarity)
        if g.controls:
            result.extend(_decompose_single_controlled(g))
        else:
            result.append(g)
    return result


def _decompose_ch(
    control: int, target: int, control_polarity: ControlPolarity
) -> list[FlatGate]:
    """Decompose CH = Ry(-pi/4), CZ, Ry(pi/4), then decompose the Ry gates."""
    result = []
    result.extend(_decompose_ry(-math.pi / 4, target))
    result.append(FlatGate("H", (target,), is_conjugate_pair=True))
    result.append(FlatGate("NOT", (target,), (control,), (), (control_polarity,)))
    result.append(FlatGate("H", (target,), is_conjugate_pair=True))
    result.extend(_decompose_ry(math.pi / 4, target))
    return result


def _decompose_ccx(control1: int, control2: int, target: int) -> list[FlatGate]:
    """Decompose CCX into Clifford+T gates."""
    return [
        FlatGate("H", (target,)),
        FlatGate("NOT", (target,), (control1,)),
        FlatGate("T_dag", (target,)),
        FlatGate("NOT", (target,), (control2,)),
        FlatGate("T", (target,)),
        FlatGate("NOT", (target,), (control1,)),
        FlatGate("T_dag", (target,)),
        FlatGate("NOT", (target,), (control2,)),
        FlatGate("T", (target,)),
        FlatGate("T", (control1,)),
        FlatGate("H", (target,)),
        FlatGate("NOT", (control1,), (control2,)),
        FlatGate("T_dag", (control1,)),
        FlatGate("T", (control2,)),
        FlatGate("NOT", (control1,), (control2,)),
    ]


def flat_gate_to_qasm(gate: FlatGate) -> str:
    """Convert a FlatGate to QASM format string."""
    gate_map = {
        "NOT": "x",
        "SU2Rotation": "u3",
        "GlobalPhase": "rz",
        "T_dag": "tdg",
    }
    gate_name = gate_map.get(gate.name, gate.name.lower())

    if gate.controls:
        return _format_controlled_gate_qasm(gate, gate_name)

    if gate.params:
        param_str = ",".join(f"{p:.6f}" for p in gate.params)
        qubit_str = _format_qubits(gate.targets)
        return f"{gate_name}({param_str}) {qubit_str};"

    qubit_str = _format_qubits(gate.targets)
    return f"{gate_name} {qubit_str};"


def _format_qubits(qubits: tuple[int, ...]) -> str:
    """Format qubit indices as QASM string."""
    return ",".join(f"q[{i}]" for i in qubits)


def _format_controlled_gate_qasm(gate: FlatGate, gate_name: str) -> str:
    """Format controlled gate as QASM string."""
    ctrl_str = _format_qubits(gate.controls)
    target_str = _format_qubits(gate.targets)

    if len(gate.controls) == 1:
        ctrl = gate.controls[0]
        if gate.name == "NOT":
            return f"cx q[{ctrl}],{target_str};"
        if gate.name == "Z":
            return f"cz q[{ctrl}],{target_str};"
        if gate.name == "Y":
            return f"cy q[{ctrl}],{target_str};"
        if gate.name == "H":
            return f"ch q[{ctrl}],{target_str};"

        if gate_name in ("crz", "cu3"):
            if gate.params:
                param_str = ",".join(f"{p:.6f}" for p in gate.params)
                return f"{gate_name}({param_str}) q[{ctrl}],{target_str};"
            return f"{gate_name} q[{ctrl}],{target_str};"

        return f"c{gate_name} q[{ctrl}],{target_str};"

    if len(gate.controls) == 2:
        if gate.name == "NOT" or gate.name == "ccx":
            return f"ccx {ctrl_str},{target_str};"
        return f"cc{gate_name} {ctrl_str},{target_str};"

    ctrl_prefix = "c" * len(gate.controls)
    return f"{ctrl_prefix}{gate_name} {ctrl_str},{target_str};"


def circuit_to_qasm_gates(
    circuit: Circuit, seed: int, ctx: QASMContext
) -> list[FlatGate]:
    """Decompose circuit gates and cancel adjacent self-inverse gates."""
    flat_gates = circuit.to_list()
    decomposed_gates: list[FlatGate] = []
    for fg in flat_gates:
        decomposed_gates.extend(decompose_for_qasm(fg, seed, ctx))

    self_inverse = {"ccx", "x", "y", "z", "h"}
    result: list[FlatGate] = []
    for gate in decomposed_gates:
        if gate.name in self_inverse and result and result[-1] == gate:
            result.pop()
        else:
            result.append(gate)
    return result


def circuit_to_qasm(circuit: Circuit, seed: int = 42) -> str:
    """Convert circuit to QASM 2.0 format string."""
    initial_qubits = circuit.allocation.total_qubits()
    ctx = QASMContext(total_qubits=initial_qubits)
    gates = circuit_to_qasm_gates(circuit, seed, ctx)
    total_qubits = ctx.get_total_qubits()

    lines = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{total_qubits}];",
        "",
    ]

    for gate in gates:
        if gate.name == "ccx":
            decomposed = _decompose_ccx(
                gate.controls[0], gate.controls[1], gate.targets[0]
            )
            lines.extend(flat_gate_to_qasm(g) for g in decomposed)
        else:
            lines.append(flat_gate_to_qasm(gate))

    return "\n".join(lines)


def circuit_to_gate_count(circuit: Circuit, seed: int = 42) -> int:
    """Count gates in decomposed circuit, treating ccx as multiple gates."""
    initial_qubits = circuit.allocation.total_qubits()
    ctx = QASMContext(total_qubits=initial_qubits)
    gates = circuit_to_qasm_gates(circuit, seed, ctx)
    gates_per_ccx = len(_decompose_ccx(0, 1, 2))
    return sum(gates_per_ccx if gate.name == "ccx" else 1 for gate in gates)
