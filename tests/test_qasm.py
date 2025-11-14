import math
import re

from cobble.circuit import (
    Circuit,
    Control,
    Controlled,
    FlatGate,
    Hadamard,
    NOT,
    QubitAllocation,
    Rx,
    Ry,
    Rz,
    SU2Rotation,
    SWAP,
    Z,
)
from cobble.polynomial import TOL
from cobble.qasm import QASMContext, decompose_for_qasm


class TestBasicGateQASM:
    """Test QASM generation for basic single-qubit gates."""

    def test_not_gate(self):
        """Test NOT gate QASM output."""
        gate = FlatGate("NOT", (0,))
        assert gate.to_qasm() == "x q[0];"

    def test_hadamard_gate(self):
        """Test Hadamard gate QASM output."""
        gate = FlatGate("H", (1,))
        assert gate.to_qasm() == "h q[1];"

    def test_z_gate(self):
        """Test Z gate QASM output."""
        gate = FlatGate("Z", (2,))
        assert gate.to_qasm() == "z q[2];"

    def test_multiple_qubits(self):
        """Test gates on different qubits."""
        gates = [FlatGate("NOT", (i,)) for i in range(5)]
        qasms = [g.to_qasm() for g in gates]
        assert qasms == [f"x q[{i}];" for i in range(5)]


class TestParametrizedGates:
    """Test QASM generation for parameterized gates."""

    def test_rx_gate(self):
        """Test Rx gate with parameter."""
        gate = FlatGate("Rx", (0,), (), (math.pi / 2,))
        qasm = gate.to_qasm()
        assert "rx(" in qasm
        assert "1.570796" in qasm
        assert "q[0]" in qasm

    def test_ry_gate(self):
        """Test Ry gate with parameter."""
        gate = FlatGate("Ry", (1,), (), (math.pi,))
        qasm = gate.to_qasm()
        assert "ry(" in qasm
        assert "3.141593" in qasm
        assert "q[1]" in qasm

    def test_rz_gate(self):
        """Test Rz gate with parameter."""
        gate = FlatGate("Rz", (2,), (), (math.pi / 4,))
        qasm = gate.to_qasm()
        assert "rz(" in qasm
        assert "0.785398" in qasm
        assert "q[2]" in qasm

    def test_su2_rotation(self):
        """Test SU2Rotation (u3) gate decomposes to Rz, H, Rz, H, Rz."""
        gate = FlatGate("SU2Rotation", (0,), (), (math.pi, math.pi / 2, math.pi / 4))
        decomposed = decompose_for_qasm(gate)
        # Should decompose to 5 gates: Rz, H, Rz, H, Rz
        assert len(decomposed) == 5
        assert decomposed[0].name == "Rz"
        assert decomposed[1].name == "H"
        assert decomposed[2].name == "Rz"
        assert decomposed[3].name == "H"
        assert decomposed[4].name == "Rz"
        assert abs(decomposed[0].params[0] - (math.pi / 4 - math.pi / 2)) < TOL
        assert abs(decomposed[2].params[0] - math.pi) < TOL
        assert abs(decomposed[4].params[0] - (math.pi / 2 + math.pi / 2)) < TOL


class TestControlledGates:
    """Test QASM generation for controlled gates."""

    def test_cnot(self):
        """Test CNOT gate."""
        gate = FlatGate("NOT", (1,), (0,))
        assert gate.to_qasm() == "cx q[0],q[1];"

    def test_controlled_z(self):
        """Test controlled-Z gate."""
        gate = FlatGate("Z", (1,), (0,))
        assert gate.to_qasm() == "cz q[0],q[1];"

    def test_controlled_hadamard(self):
        """Test controlled-H gate."""
        gate = FlatGate("H", (1,), (0,))
        assert gate.to_qasm() == "ch q[0],q[1];"

    def test_toffoli(self):
        """Test Toffoli (CCX) gate."""
        gate = FlatGate("NOT", (2,), (0, 1))
        assert gate.to_qasm() == "ccx q[0],q[1],q[2];"

    def test_ccx_alias(self):
        """Test ccx gate name directly."""
        gate = FlatGate("ccx", (2,), (0, 1))
        assert gate.to_qasm() == "ccx q[0],q[1],q[2];"


class TestControlledParametrizedGates:
    """Test QASM generation for controlled parameterized gates."""

    def test_controlled_rz_decomposition(self):
        """Test controlled Rz decomposes to CNOT, Rz, CNOT, Rz."""
        gate = FlatGate("Rz", (1,), (0,), (math.pi / 2,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 4
        # Should be: CNOT, Rz(-theta), CNOT, Rz(theta)
        assert decomposed[0].name == "NOT"  # CNOT
        assert decomposed[1].name == "Rz"
        assert decomposed[2].name == "NOT"  # CNOT
        assert decomposed[3].name == "Rz"
        assert abs(decomposed[1].params[0] - (-math.pi / 2)) < TOL
        assert abs(decomposed[3].params[0] - (math.pi / 2)) < TOL

    def test_controlled_rx_decomposition(self):
        """Test controlled Rx decomposes to H, CRz, H."""
        gate = FlatGate("Rx", (1,), (0,), (math.pi,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 1 + 4 + 1
        # Should be: H, CNOT, Rz(-theta), CNOT, Rz(theta), H
        assert decomposed[0].name == "H"
        assert decomposed[1].name == "NOT"  # CNOT
        assert decomposed[2].name == "Rz"
        assert decomposed[3].name == "NOT"  # CNOT
        assert decomposed[4].name == "Rz"
        assert decomposed[5].name == "H"

    def test_controlled_ry_decomposition(self):
        """Test controlled Ry decomposes to S†, CRx, S."""
        gate = FlatGate("Ry", (1,), (0,), (math.pi / 4,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 1 + 1 + 4 + 1 + 1
        # Should be: S†, H, CNOT, Rz(-theta), CNOT, Rz(theta), H, S
        assert decomposed[0].name == "Rz"  # S†
        assert decomposed[1].name == "H"
        assert decomposed[2].name == "NOT"  # CNOT
        assert decomposed[3].name == "Rz"
        assert decomposed[4].name == "NOT"  # CNOT
        assert decomposed[5].name == "Rz"
        assert decomposed[6].name == "H"
        assert decomposed[7].name == "Rz"  # S

    def test_controlled_su2rotation_decomposition(self):
        """Test controlled SU2Rotation decomposes to CRz, CRx, CRz."""
        gate = FlatGate("SU2Rotation", (1,), (0,), (0.1, 0.2, 0.3))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 4 + 6 + 4
        # First 4: CRz(lambda - pi/2)
        assert decomposed[0].name == "NOT"  # CNOT
        assert decomposed[1].name == "Rz"
        assert decomposed[2].name == "NOT"  # CNOT
        assert decomposed[3].name == "Rz"
        # Next 6: CRx(theta) = H + CRz + H
        assert decomposed[4].name == "H"
        assert decomposed[5].name == "NOT"  # CNOT
        assert decomposed[6].name == "Rz"
        assert decomposed[7].name == "NOT"  # CNOT
        assert decomposed[8].name == "Rz"
        assert decomposed[9].name == "H"
        # Last 4: CRz(phi + pi/2)
        assert decomposed[10].name == "NOT"  # CNOT
        assert decomposed[11].name == "Rz"
        assert decomposed[12].name == "NOT"  # CNOT
        assert decomposed[13].name == "Rz"


class TestSWAPDecomposition:
    """Test SWAP gate decomposition."""

    def test_swap_basic(self):
        """Test basic SWAP decomposes to three CNOTs."""
        gate = FlatGate("SWAP", (0, 1))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 3
        assert all(g.name == "NOT" for g in decomposed)
        assert decomposed[0].targets == (1,) and decomposed[0].controls == (0,)
        assert decomposed[1].targets == (0,) and decomposed[1].controls == (1,)
        assert decomposed[2].targets == (1,) and decomposed[2].controls == (0,)

    def test_swap_different_qubits(self):
        """Test SWAP on different qubit pairs."""
        gate = FlatGate("SWAP", (3, 7))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 3
        assert decomposed[0].targets == (7,) and decomposed[0].controls == (3,)
        assert decomposed[1].targets == (3,) and decomposed[1].controls == (7,)
        assert decomposed[2].targets == (7,) and decomposed[2].controls == (3,)

    def test_controlled_swap(self):
        """Test controlled SWAP (Fredkin gate)."""
        gate = FlatGate("SWAP", (1, 2), (0,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 3
        assert all(g.name == "NOT" for g in decomposed)
        assert len(decomposed[0].controls) == 1
        assert len(decomposed[1].controls) == 2
        assert len(decomposed[2].controls) == 1


class TestGlobalPhase:
    """Test GlobalPhase gate handling."""

    def test_uncontrolled_globalphase_disappears(self):
        """Test uncontrolled GlobalPhase disappears."""
        gate = FlatGate("GlobalPhase", (0,), (), (math.pi / 2,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 0

    def test_controlled_globalphase_becomes_crz(self):
        """Test controlled GlobalPhase decomposes to CNOT, Rz, CNOT, Rz."""
        gate = FlatGate("GlobalPhase", (1,), (0,), (math.pi / 4,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 4
        # Should be: CNOT, Rz(-theta), CNOT, Rz(theta)
        assert decomposed[0].name == "NOT"  # CNOT
        assert decomposed[1].name == "Rz"
        assert decomposed[2].name == "NOT"  # CNOT
        assert decomposed[3].name == "Rz"
        assert abs(decomposed[1].params[0] - (-math.pi / 4)) < TOL
        assert abs(decomposed[3].params[0] - (math.pi / 4)) < TOL


class TestMultiControlledGates:
    """Test multi-controlled gate decomposition."""

    def test_ccx_stays_as_is(self):
        """Test CCX (Toffoli) is kept as-is."""
        gate = FlatGate("NOT", (2,), (0, 1))
        ctx = QASMContext(total_qubits=3)
        decomposed = decompose_for_qasm(gate, ctx=ctx)
        assert len(decomposed) == 1
        assert decomposed[0].name == "ccx"

    def test_ccz_decomposition(self):
        """Test CCZ decomposes using ancilla."""
        gate = FlatGate("Z", (2,), (0, 1))
        ctx = QASMContext(total_qubits=3)
        decomposed = decompose_for_qasm(gate, ctx=ctx)
        assert len(decomposed) == 5
        gate_names = [g.name.lower() for g in decomposed]
        assert "ccx" in gate_names
        assert "not" in gate_names
        assert "h" in gate_names


class TestSpecialDollarGate:
    """Test the special $ gate (random rotation)."""

    def test_dollar_gate_deterministic(self):
        """Test $ gate gives same result with same seed."""
        gate1 = FlatGate("$", (0,))
        gate2 = FlatGate("$", (0,))
        decomposed1 = decompose_for_qasm(gate1, seed=42)
        decomposed2 = decompose_for_qasm(gate2, seed=42)
        assert decomposed1[0].params == decomposed2[0].params

    def test_dollar_gate_different_seeds(self):
        """Test $ gate gives different results with different seeds."""
        gate1 = FlatGate("$", (0,))
        gate2 = FlatGate("$", (0,))
        decomposed1 = decompose_for_qasm(gate1, seed=42)
        decomposed2 = decompose_for_qasm(gate2, seed=123)
        assert decomposed1[0].params != decomposed2[0].params

    def test_dollar_gate_different_qubits(self):
        """Test $ gate gives different results for different qubits."""
        gate1 = FlatGate("$", (0,))
        gate2 = FlatGate("$", (1,))
        decomposed1 = decompose_for_qasm(gate1, seed=42)
        decomposed2 = decompose_for_qasm(gate2, seed=42)
        assert decomposed1[0].params != decomposed2[0].params

    def test_dollar_gate_multi_qubit(self):
        """Test $ gate on multiple qubits applies u3 to each."""
        gate = FlatGate("$", (0, 1, 2))
        decomposed = decompose_for_qasm(gate, seed=42)
        assert len(decomposed) == 15

    def test_dollar_gate_multi_qubit_different_params(self):
        """Test $ gate on multiple qubits gives different params per qubit."""
        gate = FlatGate("$", (0, 1))
        decomposed = decompose_for_qasm(gate, seed=42)
        assert len(decomposed) == 10
        assert decomposed[0].params != decomposed[1].params


class TestCircuitQASM:
    """Test full circuit QASM generation."""

    def test_simple_circuit(self):
        """Test simple circuit QASM."""
        gates = [NOT(0), Hadamard(1), Z(2)]
        circuit = Circuit(gates, QubitAllocation(3, []))
        qasm = circuit.to_qasm()

        assert "OPENQASM 2.0" in qasm
        assert 'include "qelib1.inc"' in qasm
        assert "qreg q[3]" in qasm
        assert "x q[0];" in qasm
        assert "h q[1];" in qasm
        assert "z q[2];" in qasm

    def test_circuit_with_parameters(self):
        """Test circuit with parameterized gates."""
        gates = [Rx(0, math.pi), Ry(1, math.pi / 2), Rz(2, math.pi / 4)]
        circuit = Circuit(gates, QubitAllocation(3, []))
        qasm = circuit.to_qasm()

        # Rx should be decomposed
        assert "rx(" not in qasm
        assert "h " in qasm
        assert "rz(" in qasm
        # Ry should be decomposed
        assert "ry(" not in qasm
        assert "rz(" in qasm
        assert "q[1]" in qasm

    def test_circuit_with_controlled_gates(self):
        """Test circuit with controlled gates."""
        gates = [
            NOT(0),
            Controlled(NOT(1), (0,)),
        ]
        circuit = Circuit(gates, QubitAllocation(3, []))
        qasm = circuit.to_qasm()

        assert "x q[0];" in qasm
        assert "cx q[0],q[1];" in qasm

    def test_circuit_with_swap(self):
        """Test circuit with SWAP gate."""
        gates = [SWAP(0, 1)]
        circuit = Circuit(gates, QubitAllocation(2, []))
        qasm = circuit.to_qasm()

        assert qasm.count("cx ") == 3


class TestMultiControlledComplexGates:
    """Test complex multi-controlled gate decompositions with ancillas."""

    def test_three_controlled_not(self):
        """Test 3-controlled NOT (C3X) decomposes using ancilla."""

        gate = FlatGate("NOT", (3,), (0, 1, 2))
        ctx = QASMContext(total_qubits=4)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(ctx.allocated_ancillas) == 1
        assert len(decomposed) >= 1
        assert any(g.name == "ccx" for g in decomposed)

    def test_four_controlled_not(self):
        """Test 4-controlled NOT (C4X) decomposes using ancillas."""

        gate = FlatGate("NOT", (4,), (0, 1, 2, 3))
        ctx = QASMContext(total_qubits=5)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(ctx.allocated_ancillas) >= 2
        assert len(decomposed) >= 3
        for g in decomposed:
            assert g.name in ("ccx", "NOT")

    def test_five_controlled_not(self):
        """Test 5-controlled NOT (C5X) decomposes correctly."""

        gate = FlatGate("NOT", (5,), (0, 1, 2, 3, 4))
        ctx = QASMContext(total_qubits=6)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(ctx.allocated_ancillas) >= 3
        assert len(decomposed) >= 5

    def test_three_controlled_z(self):
        """Test 3-controlled Z (C3Z) decomposes using ancilla."""

        gate = FlatGate("Z", (3,), (0, 1, 2))
        ctx = QASMContext(total_qubits=4)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(ctx.allocated_ancillas) >= 1
        assert len(decomposed) >= 3
        gate_names = [g.name for g in decomposed]
        assert "ccx" in gate_names

    def test_double_controlled_rz(self):
        """Test double-controlled Rz (CC-Rz) decomposes properly."""

        gate = FlatGate("Rz", (2,), (0, 1), (math.pi / 4,))
        ctx = QASMContext(total_qubits=3)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(decomposed) == 1 + 4 + 1
        gate_names = [g.name for g in decomposed]
        assert "ccx" in gate_names
        assert "NOT" in gate_names  # CNOT gates from CRz decomposition
        assert "Rz" in gate_names
        assert len(ctx.allocated_ancillas) >= 1

    def test_double_controlled_rx(self):
        """Test double-controlled Rx (CC-Rx) decomposes properly."""

        gate = FlatGate("Rx", (2,), (0, 1), (math.pi / 2,))
        ctx = QASMContext(total_qubits=3)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(decomposed) == 1 + 6 + 1
        gate_names = [g.name for g in decomposed]
        assert "ccx" in gate_names
        assert "H" in gate_names  # From CRx decomposition
        assert "NOT" in gate_names  # CNOT gates from CRz decomposition
        assert "Rz" in gate_names
        assert len(ctx.allocated_ancillas) >= 1

    def test_double_controlled_ry(self):
        """Test double-controlled Ry (CC-Ry) decomposes properly."""

        gate = FlatGate("Ry", (2,), (0, 1), (math.pi / 3,))
        ctx = QASMContext(total_qubits=3)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(decomposed) == 1 + 1 + 1 + 4 + 1 + 1 + 1
        gate_names = [g.name for g in decomposed]
        assert "ccx" in gate_names
        assert "H" in gate_names  # From CRy decomposition
        assert "NOT" in gate_names  # CNOT gates from CRz decomposition
        assert "Rz" in gate_names
        assert len(ctx.allocated_ancillas) >= 1

    def test_double_controlled_u3(self):
        """Test double-controlled u3 (CC-u3) decomposes properly."""

        gate = FlatGate("SU2Rotation", (2,), (0, 1), (0.5, 0.6, 0.7))
        ctx = QASMContext(total_qubits=3)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        gate_names = [g.name for g in decomposed]
        assert "ccx" in gate_names
        assert "H" in gate_names  # From CU3 decomposition
        assert "NOT" in gate_names  # CNOT gates from CRz decomposition
        assert "Rz" in gate_names
        assert len(ctx.allocated_ancillas) >= 1

    def test_triple_controlled_rz(self):
        """Test triple-controlled Rz (C3-Rz) uses ancilla."""

        gate = FlatGate("Rz", (3,), (0, 1, 2), (math.pi / 8,))
        ctx = QASMContext(total_qubits=4)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(ctx.allocated_ancillas) >= 1
        assert len(decomposed) >= 5

    def test_triple_controlled_rx(self):
        """Test triple-controlled Rx (C3-Rx) uses ancilla."""

        gate = FlatGate("Rx", (3,), (0, 1, 2), (math.pi / 4,))
        ctx = QASMContext(total_qubits=4)
        decomposed = decompose_for_qasm(gate, ctx=ctx)

        assert len(ctx.allocated_ancillas) >= 1
        assert len(decomposed) >= 5

    def test_ancilla_allocation_is_sequential(self):
        """Test that ancillas are allocated sequentially."""

        ctx = QASMContext(total_qubits=3)
        gate1 = FlatGate("NOT", (3,), (0, 1, 2))
        gate2 = FlatGate("NOT", (4,), (0, 1, 2, 3))

        decomposed1 = decompose_for_qasm(gate1, ctx=ctx)
        initial_ancillas = len(ctx.allocated_ancillas)

        decomposed2 = decompose_for_qasm(gate2, ctx=ctx)

        assert len(ctx.allocated_ancillas) > initial_ancillas
        for i, anc in enumerate(ctx.allocated_ancillas):
            assert anc == 3 + i

    def test_circuit_with_multi_controlled_gates(self):
        """Test full circuit with multi-controlled gates generates correct QASM."""
        gates = [
            Hadamard(0),
            Hadamard(1),
            Hadamard(2),
            Controlled(NOT(3), (0, 1, 2)),
        ]
        circuit = Circuit(gates, QubitAllocation(4, []))
        qasm = circuit.to_qasm()

        lines = qasm.split("\n")
        qreg_line = [l for l in lines if l.startswith("qreg")][0]
        assert "q[" in qreg_line
        import re

        match = re.search(r"q\[(\d+)\]", qreg_line)
        assert match
        num_qubits = int(match.group(1))
        assert num_qubits > 4

    def test_nested_multi_controlled_gates(self):
        """Test circuit with multiple multi-controlled gates."""
        gates = [
            Controlled(NOT(4), (0, 1, 2, 3)),
            Controlled(Rz(5, math.pi / 4), (1, 2, 3)),
        ]
        circuit = Circuit(gates, QubitAllocation(6, []))
        qasm = circuit.to_qasm()

        lines = qasm.split("\n")
        qreg_line = [l for l in lines if l.startswith("qreg")][0]

        match = re.search(r"q\[(\d+)\]", qreg_line)
        assert match
        num_qubits = int(match.group(1))
        assert num_qubits > 6


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_circuit(self):
        """Test empty circuit."""
        circuit = Circuit([], QubitAllocation(0, []))
        qasm = circuit.to_qasm()
        assert "OPENQASM 2.0" in qasm
        assert "qreg q[0]" in qasm

    def test_single_qubit_circuit(self):
        """Test single qubit circuit."""
        gates = [Hadamard(0)]
        circuit = Circuit(gates, QubitAllocation(1, []))
        qasm = circuit.to_qasm()
        assert "qreg q[1]" in qasm
        assert "h q[0];" in qasm

    def test_gate_with_no_params_decomposition(self):
        """Test gate with no parameters doesn't break decomposition."""
        gate = FlatGate("H", (0,))
        decomposed = decompose_for_qasm(gate)
        assert len(decomposed) == 1
        assert decomposed[0].name == "H"

    def test_already_decomposed_gate(self):
        """Test already decomposed gate stays the same."""
        gate = FlatGate("H", (0,))
        decomposed = decompose_for_qasm(gate)
        assert decomposed == [gate]


class TestSophisticatedConstructions:
    """Test sophisticated quantum circuits."""

    def test_bell_state_circuit(self):
        """Test Bell state preparation circuit."""
        gates = [Hadamard(0), Controlled(NOT(1), (0,))]
        circuit = Circuit(gates, QubitAllocation(2, []))
        qasm = circuit.to_qasm()

        assert "h q[0];" in qasm
        assert "cx q[0],q[1];" in qasm

    def test_qft_style_circuit(self):
        """Test QFT-style circuit with many rotations."""
        gates = [
            Hadamard(0),
            Controlled(Rz(1, math.pi / 2), (0,)),
            Hadamard(1),
            SWAP(0, 1),
        ]
        circuit = Circuit(gates, QubitAllocation(2, []))
        qasm = circuit.to_qasm()

        assert "h q[0];" in qasm
        assert "h q[1];" in qasm
        assert "cx q[0],q[1];" in qasm  # CNOT from CRz decomposition
        assert "rz(" in qasm  # Rz gates from CRz decomposition
        assert qasm.count("cx ") == 3 + 2

    def test_multiply_controlled_circuit(self):
        """Test circuit with multiple control patterns."""
        gates = [
            Hadamard(0),
            Controlled(NOT(1), (0,)),  # CNOT
            Controlled(NOT(2), (0, 1)),  # Toffoli
            Controlled(Hadamard(3), (2,)),  # Controlled-H
        ]
        circuit = Circuit(gates, QubitAllocation(4, []))
        qasm = circuit.to_qasm()

        assert "h q[0];" in qasm
        assert "cx q[0],q[1];" in qasm
        # CH should be decomposed, so check for its components
        assert "ch " not in qasm
        assert "rz(" in qasm  # Ry decomposition components
        assert "cx q[2],q[3];" in qasm  # CNOT from CZ from CH decomposition
        # Toffoli should be decomposed, so check for its components
        assert "ccx " not in qasm
        assert "t " in qasm
        assert "tdg " in qasm

    def test_chain_of_swaps(self):
        """Test chain of SWAP gates."""
        gates = [
            SWAP(0, 1),
            SWAP(1, 2),
            SWAP(2, 3),
        ]
        circuit = Circuit(gates, QubitAllocation(4, []))
        qasm = circuit.to_qasm()

        assert qasm.count("cx ") == 3 * 3

    def test_complex_parameterized_circuit(self):
        """Test complex circuit with many parameterized gates."""
        gates = [
            Rx(0, 0.1),
            Ry(1, 0.2),
            Rz(2, 0.3),
            SU2Rotation(3, 0.4, 0.5, 0.6),
            Controlled(Rx(1, 0.7), (0,)),
            Controlled(Ry(2, 0.8), (1,)),
            Controlled(Rz(3, 0.9), (2,)),
        ]
        circuit = Circuit(gates, QubitAllocation(4, []))
        qasm = circuit.to_qasm()

        assert "rx(" not in qasm
        assert "h " in qasm
        assert "rz(" in qasm  # Rx decomposes to Rz, H, Rz, H, Rz
        assert "ry(" not in qasm
        assert "h " in qasm
        assert "rz(" in qasm  # Ry decomposes to Rz, Rx, Rz
        assert "u3(" not in qasm
        assert "cu3(" not in qasm
        assert "crz(" not in qasm
        assert "cx " in qasm

    def test_anti_control_circuit(self):
        """Test circuit with anti-controls (negative polarity)."""
        gates = [
            Hadamard(0),
            Controlled(NOT(1), (Control.neg(0),)),  # Anti-controlled NOT
        ]
        circuit = Circuit(gates, QubitAllocation(2, []))
        qasm = circuit.to_qasm()

        # Anti-control is wrapped with NOT gates
        assert "x q[0];" in qasm
        assert "cx q[0],q[1];" in qasm
        # Count NOT gates (should be 2 for wrapping + 1 for cx = 3 x gates total)
        # But cx is separate, so should see 2 x gates for wrapping
        assert qasm.count("x q[0];") >= 2

    def test_mixed_gate_types_circuit(self):
        """Test circuit mixing all gate types."""
        gates = [
            Hadamard(0),
            NOT(1),
            Z(2),
            Rx(0, math.pi / 4),
            Ry(1, math.pi / 2),
            Rz(2, 3 * math.pi / 4),
            Controlled(NOT(1), (0,)),
            Controlled(Hadamard(2), (1,)),
            Controlled(NOT(3), (0, 2)),
            SWAP(1, 2),
        ]
        circuit = Circuit(gates, QubitAllocation(4, []))
        qasm = circuit.to_qasm()

        assert "OPENQASM 2.0" in qasm
        assert 'include "qelib1.inc"' in qasm
        assert "qreg q[4]" in qasm
        assert "h " in qasm
        assert "x " in qasm
        assert "z " in qasm
        assert "rz(" in qasm
        assert "h " in qasm
        assert "rz(" in qasm  # Ry decomposes to Rz, Rx, Rz
        assert "cx " in qasm
        assert "cx " in qasm  # CNOT from CZ from CH decomposition
        assert "ry(" not in qasm
        assert "ch " not in qasm
        assert "ccx " not in qasm


class TestDecompositionInvariants:
    """Test that decomposition preserves important properties."""

    def test_decomposition_preserves_qubits(self):
        """Test that decomposition doesn't introduce new qubits."""
        gate = FlatGate("SWAP", (0, 1))
        decomposed = decompose_for_qasm(gate)

        all_qubits = set()
        for g in decomposed:
            all_qubits.update(g.targets)
            all_qubits.update(g.controls)

        assert all_qubits == {0, 1}

    def test_controlled_swap_preserves_qubits(self):
        """Test controlled SWAP doesn't introduce new qubits."""
        gate = FlatGate("SWAP", (1, 2), (0,))
        decomposed = decompose_for_qasm(gate)

        all_qubits = set()
        for g in decomposed:
            all_qubits.update(g.targets)
            all_qubits.update(g.controls)

        assert all_qubits == {0, 1, 2}

    def test_decomposition_returns_list(self):
        """Test that decompose_for_qasm always returns a list."""
        gates = [
            FlatGate("H", (0,)),
            FlatGate("NOT", (1,)),
            FlatGate("SWAP", (0, 1)),
            FlatGate("GlobalPhase", (0,), (), (math.pi,)),
            FlatGate("$", (0,)),
        ]

        for gate in gates:
            result = decompose_for_qasm(gate)
            assert isinstance(result, list)
            assert all(isinstance(g, FlatGate) for g in result)


class TestQASMFormatValidity:
    """Test that generated QASM follows correct format."""

    def test_qasm_header(self):
        """Test QASM has correct header."""
        circuit = Circuit([Hadamard(0)], QubitAllocation(1, []))
        qasm = circuit.to_qasm()
        lines = qasm.split("\n")

        assert lines[0] == "OPENQASM 2.0;"
        assert lines[1] == 'include "qelib1.inc";'
        assert lines[2].startswith("qreg q[")

    def test_qasm_gate_format(self):
        """Test QASM gates follow correct format."""
        circuit = Circuit([NOT(0), Hadamard(1)], QubitAllocation(2, []))
        qasm = circuit.to_qasm()

        for line in qasm.split("\n"):
            if line and not line.startswith(("OPENQASM", "include", "qreg")):
                assert line.endswith(";")

    def test_qasm_qubit_format(self):
        """Test QASM qubits use correct format."""
        circuit = Circuit([NOT(0), NOT(5)], QubitAllocation(6, []))
        qasm = circuit.to_qasm()

        assert "q[0]" in qasm
        assert "q[5]" in qasm
        assert "qreg q[6]" in qasm

    def test_qasm_parameter_format(self):
        """Test QASM parameters use correct format."""
        circuit = Circuit([Rz(0, 1.5)], QubitAllocation(1, []))
        qasm = circuit.to_qasm()

        assert "rz(1.500000)" in qasm or "rz(1.5" in qasm


class TestRegressionCases:
    """Test cases to prevent regressions."""

    def test_multiple_controlled_params(self):
        """Test multiple controlled parameterized gates in sequence."""
        gates = [
            Controlled(Rx(1, 0.5), (0,)),
            Controlled(Ry(2, 0.6), (1,)),
            Controlled(Rz(3, 0.7), (2,)),
        ]
        circuit = Circuit(gates, QubitAllocation(4, []))
        qasm = circuit.to_qasm()

        assert "cu3(" not in qasm
        assert "crz(" not in qasm
        assert "cx " in qasm
        assert "rz(" in qasm
        assert "h " in qasm

    def test_nested_controlled_gates(self):
        """Test nested controlled gates flatten correctly."""
        inner_gate = NOT(2)
        controlled_once = Controlled(inner_gate, (1,))
        controlled_twice = Controlled(controlled_once, (0,))

        circuit = Circuit([controlled_twice], QubitAllocation(3, []))
        qasm = circuit.to_qasm()

        assert "ccx " not in qasm
        assert "x " in qasm
        assert "t " in qasm
        assert "tdg " in qasm
        assert "h " in qasm
        assert "cx " in qasm

    def test_swap_followed_by_gate(self):
        """Test SWAP followed by gate on swapped qubits."""
        gates = [
            SWAP(0, 1),
            Controlled(NOT(0), (1,)),
        ]
        circuit = Circuit(gates, QubitAllocation(2, []))
        qasm = circuit.to_qasm()

        assert qasm.count("cx ") == 3 + 1
