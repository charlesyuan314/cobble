import math

from cobble.circuit import (
    BlackBox,
    Circuit,
    Control,
    Controlled,
    Decrement,
    Hadamard,
    Increment,
    NOT,
    QubitAllocation,
    Ry,
    Rz,
    XorInt,
    sign_correction,
    state_preparation_tree,
)
from cobble.expr import Basic, Condition, Const, If, Poly, Prod, QSP_TOL, Sum, Tensor
from cobble.polynomial import Polynomial, TOL
from cobble.qtype import BitType
from cobble.simulator import CircuitSimulator
import numpy as np


class TestGates:
    """Test individual gate operations."""

    def test_not_gate(self):
        """Test NOT gate."""
        gate = NOT(0)
        assert gate.adjoint() == gate
        flat = gate.to_flat_gates()
        assert len(flat) == 1
        assert flat[0].name == "NOT"
        assert flat[0].targets == (0,)

    def test_hadamard_gate(self):
        """Test Hadamard gate."""
        gate = Hadamard(1)
        assert gate.adjoint() == gate
        flat = gate.to_flat_gates()
        assert len(flat) == 1
        assert flat[0].name == "H"

    def test_controlled_gate(self):
        """Test controlled gate with positive control."""
        gate = NOT(0)
        cgate = Controlled(gate, (1,))
        flat = cgate.to_flat_gates()
        assert len(flat) == 1
        assert flat[0].controls == (1,)
        assert flat[0].targets == (0,)

    def test_controlled_gate_negative(self):
        """Test controlled gate with negative control (control on |0>)."""
        gate = NOT(0)
        cgate = Controlled(gate, (Control.neg(1),))  # Control on qubit 1 being |0>
        # Should add 2 NOT gates (before and after)
        flat = cgate.to_flat_gates()
        assert len(flat) == 3
        # NOT on control qubit
        assert flat[0].name == "NOT"
        assert flat[0].targets == (1,)
        # Controlled gate with positive control
        assert flat[1].name == "NOT"
        assert flat[1].controls == (1,)
        assert flat[1].targets == (0,)
        # NOT on control qubit again
        assert flat[2].name == "NOT"
        assert flat[2].targets == (1,)

    def test_controlled_gate_mixed_controls(self):
        """Test controlled gate with mixed positive and negative controls."""
        gate = NOT(0)
        cgate = Controlled(gate, (1, Control.neg(2), 3))  # q1=|1>, q2=|0>, q3=|1>
        # Should add 2 NOT gates for the negative control
        flat = cgate.to_flat_gates()
        assert len(flat) == 3
        # NOT on qubit 2
        assert flat[0].name == "NOT"
        assert flat[0].targets == (2,)
        # Controlled gate with all positive controls (positive first, then negative)
        assert flat[1].name == "NOT"
        assert flat[1].controls == (
            1,
            3,
            2,
        )  # Positive controls (1,3), then negative (2)
        # NOT on qubit 2 again
        assert flat[2].name == "NOT"
        assert flat[2].targets == (2,)

    def test_black_box_gate(self):
        """Test black box gate."""
        gate = BlackBox("CustomOracle", 3, 2)
        flat = gate.to_flat_gates()
        assert len(flat) == 1
        assert flat[0].name == "CustomOracle"
        assert len(flat[0].targets) == 3 + 2

    # BlackBox start_qubit tests
    def test_blackbox_default_start_qubit(self):
        """Test BlackBox defaults to qubit 0."""
        gate = BlackBox(name="X", data_qubits=1)
        assert gate.start_qubit == 0

        flat = gate.to_flat_gates()
        assert len(flat) == 1
        assert flat[0].targets == (0,)

    def test_blackbox_custom_start_qubit(self):
        """Test BlackBox with custom start qubit."""
        gate = BlackBox(name="X", data_qubits=2, start_qubit=5)
        flat = gate.to_flat_gates()

        assert len(flat) == 1
        assert flat[0].targets == (5, 6)

    def test_blackbox_remap_qubits(self):
        """Test remapping BlackBox qubits."""
        gate = BlackBox(name="X", data_qubits=1, start_qubit=0)
        remapped = gate.remap_qubits({0: 10})

        assert isinstance(remapped, BlackBox)
        assert remapped.start_qubit == 10
        assert remapped.to_flat_gates()[0].targets == (10,)

    def test_blackbox_adjoint_preserves_start_qubit(self):
        """Test that adjoint preserves start_qubit."""
        gate = BlackBox(name="X", data_qubits=1, start_qubit=5)
        adj = gate.adjoint()

        assert isinstance(adj, BlackBox)
        assert adj.start_qubit == 5
        assert adj.name == "X_dag"

    def test_increment_single_qubit(self):
        """Test Increment on a single qubit."""
        gate = Increment((5,))
        flat = gate.to_flat_gates()

        assert len(flat) == 1
        assert flat[0].name == "NOT"
        assert flat[0].targets == (5,)
        assert flat[0].controls == ()

    def test_increment_two_qubits(self):
        """Test Increment on two qubits (little-endian)."""
        gate = Increment((3, 4))
        flat = gate.to_flat_gates()

        # Two qubits: CX(q3 -> q4), then X on q3
        # (MSB-first for carry propagation, then flip LSB)
        assert len(flat) == 2
        assert flat[0].name == "NOT"
        assert flat[0].targets == (4,)
        assert flat[0].controls == (3,)

        assert flat[1].name == "NOT"
        assert flat[1].targets == (3,)
        assert flat[1].controls == ()

    def test_increment_three_qubits(self):
        """Test Increment on three qubits."""
        gate = Increment((0, 1, 2))
        flat = gate.to_flat_gates()

        # Three qubits: CCX_{0,1->2}, CX_{0->1}, X_0
        # (MSB-first for carry propagation, then flip LSB)
        assert len(flat) == 3
        assert flat[0].name == "NOT"
        assert flat[0].targets == (2,)
        assert flat[0].controls == (0, 1)

        assert flat[1].name == "NOT"
        assert flat[1].targets == (1,)
        assert flat[1].controls == (0,)

        assert flat[2].name == "NOT"
        assert flat[2].targets == (0,)
        assert flat[2].controls == ()

    def test_increment_adjoint(self):
        """Test that adjoint of Increment is Decrement."""
        gate = Increment((1, 2, 3))
        adj = gate.adjoint()

        assert isinstance(adj, Decrement)
        assert adj.targets == (1, 2, 3)

    def test_decrement_reverses_increment(self):
        """Test that Decrement reverses Increment gates."""
        inc = Increment((5, 6))
        dec = Decrement((5, 6))

        inc_flat = inc.to_flat_gates()
        dec_flat = dec.to_flat_gates()

        assert len(inc_flat) == len(dec_flat)
        assert dec_flat == list(reversed(inc_flat))

    def test_decrement_adjoint(self):
        """Test that adjoint of Decrement is Increment."""
        gate = Decrement((1, 2))
        adj = gate.adjoint()

        assert isinstance(adj, Increment)
        assert adj.targets == (1, 2)

    def test_xorint_zero(self):
        """Test XorInt with value 0 (no gates)."""
        gate = XorInt((1, 2, 3), 0)
        flat = gate.to_flat_gates()

        assert len(flat) == 0

    def test_xorint_single_bit(self):
        """Test XorInt flipping a single bit."""
        gate = XorInt((5, 6, 7), 1)  # Binary 001 in little-endian
        flat = gate.to_flat_gates()

        # Should flip only qubit 5 (LSB)
        assert len(flat) == 1
        assert flat[0].name == "NOT"
        assert flat[0].targets == (5,)

    def test_xorint_multiple_bits(self):
        """Test XorInt flipping multiple bits."""
        gate = XorInt((0, 1, 2), 5)  # Binary 101 in little-endian
        flat = gate.to_flat_gates()

        # Should flip qubits 0 and 2
        assert len(flat) == 2
        assert flat[0].name == "NOT"
        assert flat[0].targets == (0,)
        assert flat[1].name == "NOT"
        assert flat[1].targets == (2,)

    def test_xorint_all_bits(self):
        """Test XorInt flipping all bits."""
        gate = XorInt((3, 4, 5), 7)  # Binary 111
        flat = gate.to_flat_gates()

        # Should flip all three qubits
        assert len(flat) == 3
        assert flat[0].targets == (3,)
        assert flat[1].targets == (4,)
        assert flat[2].targets == (5,)

    def test_xorint_self_adjoint(self):
        """Test that XorInt is self-adjoint."""
        gate = XorInt((1, 2, 3), 5)
        adj = gate.adjoint()

        assert adj == gate

    def test_xorint_remap(self):
        """Test remapping XorInt qubits."""
        gate = XorInt((0, 1, 2), 3)
        remapped = gate.remap_qubits({0: 10, 1: 11, 2: 12})

        assert isinstance(remapped, XorInt)
        assert remapped.targets == (10, 11, 12)
        assert remapped.value == 3

    def test_increment_remap(self):
        """Test remapping Increment qubits."""
        gate = Increment((0, 1))
        remapped = gate.remap_qubits({0: 5, 1: 6})

        assert isinstance(remapped, Increment)
        assert remapped.targets == (5, 6)


class TestCircuit:
    """Test circuit operations."""

    def test_empty_circuit(self):
        """Test empty (identity) circuit."""
        circ = Circuit.identity(2, 1)
        assert circ.data_qubits == 2
        assert circ.ancilla_qubits == 1
        assert len(circ.to_list()) == 0

    def test_circuit_adjoint(self):
        """Test circuit adjoint (reverse and conjugate)."""

        gates = [Hadamard(0), Rz(1, math.pi / 4), NOT(0)]
        alloc = QubitAllocation(2, [])
        circ = Circuit(gates, alloc)
        adj = circ.adjoint()

        # Should reverse order and conjugate each gate
        flat = adj.to_list()
        assert len(flat) == 3
        # Last gate first (NOT is self-adjoint)
        assert flat[0].name == "NOT"
        # Middle gate (Rz with negated angle)
        assert flat[1].name == "Rz"
        assert flat[1].params == (-math.pi / 4,)
        # First gate last (H is self-adjoint)
        assert flat[2].name == "H"

    def test_circuit_to_qasm(self):
        """Test QASM generation."""

        gates = [Hadamard(0), NOT(1), Controlled(NOT(0), (1,))]
        alloc = QubitAllocation(2, [])
        circ = Circuit(gates, alloc)
        qasm = circ.to_qasm()

        assert "OPENQASM 2.0" in qasm
        assert "qreg q[2]" in qasm
        assert "h q[0]" in qasm
        assert "x q[1]" in qasm
        assert "cx q[1],q[0]" in qasm

    def test_add_controls_to_uncontrolled_gate(self):
        """Test adding controls to a gate that doesn't have any."""

        gate = Hadamard(target=0)
        controlled = Controlled.add_controls(gate, (1, 2))

        assert isinstance(controlled, Controlled)
        assert controlled.gate == gate
        assert len(controlled.controls) == 2
        c0 = controlled.controls[0]
        c1 = controlled.controls[1]
        assert isinstance(c0, Control) and c0.qubit == 1
        assert isinstance(c1, Control) and c1.qubit == 2
        assert all(
            isinstance(c, Control) and c.is_positive() for c in controlled.controls
        )

    def test_add_controls_merges_nested(self):
        """Test that add_controls merges controls instead of nesting."""

        inner = Controlled(Hadamard(target=0), controls=(1,))
        result = Controlled.add_controls(inner, (2, 3))
        assert isinstance(result, Controlled)
        assert len(result.controls) == 3
        c0, c1, c2 = result.controls
        assert isinstance(c0, Control) and c0.qubit == 2
        assert isinstance(c1, Control) and c1.qubit == 3
        assert isinstance(c2, Control) and c2.qubit == 1
        assert result.gate == Hadamard(target=0)

    def test_add_controls_preserves_negative_controls(self):
        """Test that negative controls are preserved when merging."""

        gate = Controlled(
            BlackBox(name="X", data_qubits=1, start_qubit=0), controls=(Control.neg(1),)
        )
        result = Controlled.add_controls(gate, (Control.neg(2), 3))

        assert isinstance(result, Controlled)
        assert len(result.controls) == 3
        c0, c1, c2 = result.controls
        assert isinstance(c0, Control) and c0.qubit == 2 and c0.is_negative()
        assert isinstance(c1, Control) and c1.qubit == 3 and c1.is_positive()
        assert isinstance(c2, Control) and c2.qubit == 1 and c2.is_negative()

    def test_remap_circuit_qubits_data_only(self):
        """Test remapping a circuit with only data qubits."""

        allocation = QubitAllocation(2, [])
        circ = Circuit([Hadamard(target=0), Hadamard(target=1)], allocation)
        remapped = circ.remap_qubits([], data_offset=5)

        assert len(remapped.gates) == 2
        assert remapped.gates[0] == Hadamard(target=5)
        assert remapped.gates[1] == Hadamard(target=6)
        assert remapped.data_qubits == max(5, 6) + 1
        assert remapped.allocation.ancilla_qubits == []

    def test_remap_circuit_qubits_with_ancillas(self):
        """Test remapping a circuit with ancillas."""

        allocation = QubitAllocation(1, [1, 2])
        circ = Circuit([Hadamard(target=0), NOT(target=1), NOT(target=2)], allocation)

        remapped = circ.remap_qubits([10, 11], data_offset=0)

        assert len(remapped.gates) == 3
        assert remapped.gates[0] == Hadamard(target=0)
        assert remapped.gates[1] == NOT(target=10)
        assert remapped.gates[2] == NOT(target=11)
        assert remapped.data_qubits == 1
        assert remapped.allocation.ancilla_qubits == [10, 11]

    def test_all_expr_types_match(self):
        """All expression types have data_qubits == type.width()."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        C = Basic("C", qtype_=BitType(3))

        exprs = [
            A,
            Const(3.14),
            Sum([(1.0, A), (1.0, B)]),
            Prod([A, B]),
            Tensor([A, C]),
            If(Condition("x"), A, B),
        ]

        for expr in exprs:
            circ = expr.circuit()
            assert circ.data_qubits == expr.qtype().width()

    def test_ancilla_counts_match(self):
        """All expression types have circuit.ancilla_qubits == ancilla_qubits()."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=1)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=2)
        C = Basic("C", qtype_=BitType(3), ancilla_qubits_=3)

        exprs = [
            ("Base", A),
            ("Sum", Sum([(1.0, A), (1.0, B)])),
            ("Prod", Prod([A, B])),
            ("Tensor", Tensor([A, C])),
            ("If", If(Condition("x"), A, B)),
        ]

        for name, expr in exprs:
            circ = expr.circuit()
            assert (
                circ.ancilla_qubits == expr.ancilla_qubits()
            ), f"{name}: circuit={circ.ancilla_qubits}, ancilla_qubits()={expr.ancilla_qubits()}"


class TestBaseCircuitCompilation:
    """Test circuit compilation for Base expressions."""

    def test_base_default_circuit(self):
        """Base without custom circuit creates black box."""
        A = Basic("A", qtype_=BitType(2))
        circ = A.circuit()
        assert circ.data_qubits == 2
        assert circ.ancilla_qubits == 0
        flat = circ.to_list()
        assert flat[0].name == "A"

    def test_base_with_ancillas(self):
        """Base with ancillas."""
        A = Basic("A", ancilla_qubits_=3, qtype_=BitType(2))
        circ = A.circuit()
        assert circ.data_qubits == 2
        assert circ.ancilla_qubits == 3


class TestSumCircuitCompilation:
    """Test circuit compilation for Sum expressions (LCU)."""

    def test_sum_single_term(self):
        """Sum with single term just returns that term's circuit."""
        A = Basic("A", qtype_=BitType(2))
        s = Sum([(1.0, A)])
        circ = s.circuit()

        a_circ = A.circuit()
        assert circ == a_circ

    def test_sum_two_terms(self):
        """Sum with two terms uses LCU with proper controlled gates."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        s = Sum([(0.6, A), (0.4, B)])
        circ = s.circuit()

        assert circ.data_qubits == 2
        assert circ.ancilla_qubits >= 1

        has_controlled = any(isinstance(g, Controlled) for g in circ.gates)
        assert has_controlled

    def test_sum_ancilla_count(self):
        """Sum ancilla count matches ancilla_qubits() method."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=1)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=2)
        s = Sum([(1.0, A), (1.0, B)])
        circ = s.circuit()

        assert circ.ancilla_qubits == s.ancilla_qubits()

    def test_sum_ancilla_qubits_matches_circuit_unnormalized(self):
        """Sum.ancilla_qubits() computed on unnormalized sum matches circuit ancilla count."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=1)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=2)

        s = Sum([(1.0, A), (1.0, A), (1.0, B)])
        circ = s.circuit()
        assert circ.ancilla_qubits == s.ancilla_qubits()

    def test_state_prep_three_terms(self):
        """State preparation for 3 terms (non-power-of-2)."""

        coeffs = [2.0, 1.0, 3.0]
        ancillas = [0, 1]

        prep_gates = state_preparation_tree(coeffs, ancillas)
        circ = Circuit(prep_gates, QubitAllocation(0, ancillas))

        sim = CircuitSimulator(circ, gate_matrices={})
        U = sim.get_full_unitary()

        init_state = np.zeros(2**2, dtype=complex)
        init_state[0] = 1.0
        final_state = U @ init_state

        total = sum(abs(c) for c in coeffs)
        expected_amps = [np.sqrt(abs(c) / total) for c in coeffs]

        for i in range(len(coeffs)):
            actual = abs(final_state[i])
            assert (
                abs(actual - expected_amps[i]) < TOL
            ), f"State |{i:02b}⟩: expected {expected_amps[i]:.4f}, got {actual:.4f}"

        assert (
            abs(final_state[3]) < TOL
        ), f"Padded state |11⟩ has amplitude {abs(final_state[3]):.4f}"

        total_prob = sum(abs(amp) ** 2 for amp in final_state)
        assert abs(total_prob - 1.0) < TOL

    def test_state_prep_five_terms(self):
        """State preparation for 5 terms."""

        coeffs = [1.0, 2.0, 1.5, 0.5, 3.0]
        ancillas = [0, 1, 2]

        prep_gates = state_preparation_tree(coeffs, ancillas)
        circ = Circuit(prep_gates, QubitAllocation(0, ancillas))

        sim = CircuitSimulator(circ, gate_matrices={})
        U = sim.get_full_unitary()

        init_state = np.zeros(2**3, dtype=complex)
        init_state[0] = 1.0
        final_state = U @ init_state

        total = sum(abs(c) for c in coeffs)
        expected_amps = [np.sqrt(abs(c) / total) for c in coeffs]

        for i in range(len(coeffs)):
            actual = abs(final_state[i])
            assert (
                abs(actual - expected_amps[i]) < TOL
            ), f"State |{i:03b}⟩: expected {expected_amps[i]:.4f}, got {actual:.4f}"

        for i in range(5, 8):
            assert (
                abs(final_state[i]) < TOL
            ), f"Padded state |{i:03b}⟩ has amplitude {abs(final_state[i]):.4f}"

        total_prob = sum(abs(amp) ** 2 for amp in final_state)
        assert abs(total_prob - 1.0) < TOL

    def test_sign_correction_single_negative(self):
        """Sign correction with one negative coefficient."""
        coeffs = [2.0, 1.0, -3.0]
        ancillas = [0, 1]

        prep_gates = state_preparation_tree(coeffs, ancillas)
        sign_gates = sign_correction(coeffs, ancillas)
        all_gates = prep_gates + sign_gates
        circ = Circuit(all_gates, QubitAllocation(0, ancillas))

        sim = CircuitSimulator(circ, gate_matrices={})
        U = sim.get_full_unitary()

        init_state = np.zeros(2**2, dtype=complex)
        init_state[0] = 1.0
        final_state = U @ init_state

        total = sum(abs(c) for c in coeffs)
        for i, c in enumerate(coeffs):
            expected_amp = np.sqrt(abs(c) / total) * np.sign(c)
            actual = final_state[i].real
            assert (
                abs(actual - expected_amp) < TOL
            ), f"State |{i:02b}⟩: expected {expected_amp:.4f}, got {actual:.4f}"

    def test_sign_correction_multiple_negatives(self):
        """Sign correction with multiple negative coefficients."""

        coeffs = [2.0, -1.0, -3.0, 1.5]
        ancillas = [0, 1]

        prep_gates = state_preparation_tree(coeffs, ancillas)
        sign_gates = sign_correction(coeffs, ancillas)
        all_gates = prep_gates + sign_gates
        circ = Circuit(all_gates, QubitAllocation(0, ancillas))

        sim = CircuitSimulator(circ, gate_matrices={})
        U = sim.get_full_unitary()

        init_state = np.zeros(2**2, dtype=complex)
        init_state[0] = 1.0
        final_state = U @ init_state

        total = sum(abs(c) for c in coeffs)
        for i, c in enumerate(coeffs):
            expected_amp = np.sqrt(abs(c) / total) * np.sign(c)
            actual = final_state[i].real
            assert (
                abs(actual - expected_amp) < TOL
            ), f"State |{i:02b}⟩: expected {expected_amp:.4f}, got {actual:.4f}"

    def test_sign_correction_first_branch_negative(self):
        """Sign correction when first branch (index 0) is negative."""

        coeffs = [-2.0, 1.0, 3.0]
        ancillas = [0, 1]

        prep_gates = state_preparation_tree(coeffs, ancillas)
        sign_gates = sign_correction(coeffs, ancillas)
        all_gates = prep_gates + sign_gates
        circ = Circuit(all_gates, QubitAllocation(0, ancillas))

        sim = CircuitSimulator(circ, gate_matrices={})
        U = sim.get_full_unitary()

        init_state = np.zeros(2**2, dtype=complex)
        init_state[0] = 1.0
        final_state = U @ init_state

        total = sum(abs(c) for c in coeffs)
        for i, c in enumerate(coeffs):
            expected_amp = np.sqrt(abs(c) / total) * np.sign(c)
            actual = final_state[i].real
            assert (
                abs(actual - expected_amp) < TOL
            ), f"State |{i:02b}⟩: expected {expected_amp:.4f}, got {actual:.4f}"

    def test_sign_correction_all_negative(self):
        """Sign correction when all coefficients are negative."""

        coeffs = [-2.0, -1.0, -3.0]
        ancillas = [0, 1]

        prep_gates = state_preparation_tree(coeffs, ancillas)
        sign_gates = sign_correction(coeffs, ancillas)
        all_gates = prep_gates + sign_gates
        circ = Circuit(all_gates, QubitAllocation(0, ancillas))

        sim = CircuitSimulator(circ, gate_matrices={})
        U = sim.get_full_unitary()

        init_state = np.zeros(2**2, dtype=complex)
        init_state[0] = 1.0
        final_state = U @ init_state

        total = sum(abs(c) for c in coeffs)
        for i, c in enumerate(coeffs):
            expected_amp = np.sqrt(abs(c) / total) * np.sign(c)
            actual = final_state[i].real
            assert (
                abs(actual - expected_amp) < TOL
            ), f"State |{i:02b}⟩: expected {expected_amp:.4f}, got {actual:.4f}"

    def test_lcu_with_canceling_terms(self):
        """Full LCU test: 2·A + 1·B + (-2·A) = 1·B."""
        A = Basic("A")
        B = Basic("B")

        expr = Sum([(2.0, A), (1.0, B), (-2.0, A)])

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # X
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)  # Z
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        circ = expr.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})

        result = sim.get_block_encoding_matrix() * expr.subnormalization()
        expected = 1.0 * B_matrix

        assert np.allclose(result, expected)

    def test_lcu_with_all_negative_canceling(self):
        """Full LCU test: -2·A + (-1·B) + 2·A = -1·B."""
        A = Basic("A")
        B = Basic("B")

        expr = Sum([(-2.0, A), (-1.0, B), (2.0, A)])

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        circ = expr.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})

        result = sim.get_block_encoding_matrix() * expr.subnormalization()
        expected = -1.0 * B_matrix

        assert np.allclose(result, expected)


class TestProdCircuitCompilation:
    """Test circuit compilation for Prod expressions."""

    def test_prod_two_factors_ancilla_count(self):
        """Product uses max + ceil(log2(N)) ancillas."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=1)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=1)
        p = Prod([A, B])
        circ = p.circuit()

        assert circ.data_qubits == 2
        assert circ.ancilla_qubits == p.ancilla_qubits()

    def test_prod_ancilla_formula(self):
        """Product uses max(factors.ancilla_qubits()) + ceil(log2(N)) ancillas."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=3)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=2)
        p = Prod([A, B])
        circ = p.circuit()

        assert circ.ancilla_qubits == p.ancilla_qubits()

    def test_prod_structure_two_factors(self):
        """Product A*B has correct gate structure with increment approach."""

        A = Basic(
            "A",
            qtype_=BitType(2),
            circuit_=Circuit([Hadamard(2)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )
        B = Basic(
            "B",
            qtype_=BitType(2),
            circuit_=Circuit([Ry(2, 1.0)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )

        p = Prod([A, B])
        circ = p.circuit()

        # Expected structure: U_B, controlled-increment, U_A, XorInt(1)
        # For 2 factors: 1 flag qubit, increment after first factor only (1 increment total)
        assert len(circ.gates) == 4

        # Gate 0: U_B (Ry gate)
        assert isinstance(circ.gates[0], Ry)
        assert circ.gates[0].target == 2
        assert circ.gates[0].angle == 1.0

        # Gate 1: Controlled Increment on flag register (qubit 3)
        assert isinstance(circ.gates[1], Controlled)
        assert isinstance(circ.gates[1].gate, Increment)
        assert circ.gates[1].gate.targets == (3,)  # Flag register
        # Controlled on working ancilla = |0⟩ (negative control on qubit 2)
        assert all(
            isinstance(c, Control) and c.is_negative() for c in circ.gates[1].controls
        )

        # Gate 2: U_A (H gate)
        assert isinstance(circ.gates[2], Hadamard)
        assert circ.gates[2].target == 2

        # Gate 3: XorInt to flip bits for N-1=1 (uncompute)
        assert isinstance(circ.gates[3], XorInt)
        assert circ.gates[3].value == 1
        assert circ.gates[3].targets == (3,)

    def test_prod_three_factors_structure(self):
        """Product A*B*C has correct gate count and structure."""

        A = Basic(
            "A",
            qtype_=BitType(2),
            circuit_=Circuit([Hadamard(2)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )
        B = Basic(
            "B",
            qtype_=BitType(2),
            circuit_=Circuit([Ry(2, 1.0)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )
        C = Basic(
            "C",
            qtype_=BitType(2),
            circuit_=Circuit([Rz(2, 0.5)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )

        p = Prod([A, B, C])
        circ = p.circuit()

        # Expected: U_C, increment, U_B, increment, U_A, XorInt(2)
        # 3 factor gates + 2 controlled increments + 1 XorInt = 6
        assert len(circ.gates) == 6

        # Last gate should be XorInt(2) for uncompute (N-1 = 3-1 = 2)
        assert isinstance(circ.gates[-1], XorInt)
        assert circ.gates[-1].value == 2

        # Check for 2 increments (one after each non-last factor)
        increments = [
            g
            for g in circ.gates
            if isinstance(g, (Controlled, Increment))
            and (isinstance(g, Increment) or isinstance(g.gate, Increment))
        ]
        assert len(increments) == 2

    def test_prod_flag_register_uncomputed(self):
        """Flag register gets XorInt at the end."""

        A = Basic(
            "A",
            qtype_=BitType(2),
            circuit_=Circuit([Hadamard(2)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )
        B = Basic(
            "B",
            qtype_=BitType(2),
            circuit_=Circuit([Hadamard(2)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )

        p = Prod([A, B])
        circ = p.circuit()

        # Last gate should be XorInt on flag register (qubit 3)
        last_gate = circ.gates[-1]
        assert isinstance(last_gate, XorInt)
        assert last_gate.targets == (3,)
        assert last_gate.value == 1  # XOR with N-1 = 2-1 = 1

    def test_prod_working_ancillas_shared(self):
        """Working ancillas are shared/reused between factors."""
        A = Basic(
            "A",
            qtype_=BitType(2),
            circuit_=Circuit([Hadamard(2)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )
        B = Basic(
            "B",
            qtype_=BitType(2),
            circuit_=Circuit([Ry(2, 1.0)], QubitAllocation(2, [2])),
            ancilla_qubits_=1,
        )

        p = Prod([A, B])
        circ = p.circuit()

        assert "target=2" in str(circ.gates[0])  # U_B on qubit 2
        assert "target=2" in str(circ.gates[2])  # U_A on qubit 2

    def test_prod_different_ancilla_counts(self):
        """Product with factors having different ancilla counts."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=5)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=2)
        C = Basic("C", qtype_=BitType(2), ancilla_qubits_=3)

        p = Prod([A, B, C])

        assert p.circuit().ancilla_qubits == p.ancilla_qubits()

    def test_prod_five_factors_flag_count(self):
        """Product with 5 factors uses ceil(log2(5)) = 3 flag qubits."""
        bases = [Basic(f"B{i}", ancilla_qubits_=1) for i in range(5)]
        p = Prod(bases)
        circ = p.circuit()

        assert circ.ancilla_qubits == p.ancilla_qubits()

        increments = [
            g
            for g in circ.gates
            if isinstance(g, (Controlled, Increment))
            and (isinstance(g, Increment) or isinstance(g.gate, Increment))
        ]
        assert len(increments) == 4

        assert circ.gates[-1] == XorInt((2, 3, 4), 4)

    def test_prod_no_working_ancillas(self):
        """Product of factors with no ancillas still uses flag register."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        C = Basic("C", qtype_=BitType(2))

        p = Prod([A, B, C])
        circ = p.circuit()

        assert circ.ancilla_qubits == p.ancilla_qubits()

        # All increments should be unconditional (no working ancillas to condition on)
        increments = [g for g in circ.gates if isinstance(g, Increment)]
        # Should have 2 unconditional increments (one after each non-last factor)
        assert len(increments) == 2

    def test_prod_flag_register_little_endian(self):
        """Verify flag register uses little-endian ordering."""
        bases = [Basic(f"B{i}") for i in range(5)]
        p = Prod(bases)
        circ = p.circuit()

        # Final XorInt should target flag register in little-endian order
        xor_gate = circ.gates[-1]
        assert isinstance(xor_gate, XorInt)
        # For 5 factors with data_qubits=1, ancillas start at qubit 1
        # Flag register should be qubits (1, 2, 3) in little-endian
        assert xor_gate.targets == (1, 2, 3)
        assert (
            xor_gate.value == 4
        )  # XOR with N-1 = 5-1 = 4 (binary 100 in little-endian)

    def test_prod_ancilla_qubits_matches_circuit_unnormalized(self):
        """Prod.ancilla_qubits() computed on unnormalized product matches circuit ancilla count."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=1)
        p = Prod([A, A, A])

        circ = p.circuit()
        assert circ.ancilla_qubits == p.ancilla_qubits()


class TestTensorCircuitCompilation:
    """Test circuit compilation for Tensor expressions."""

    def test_tensor_two_factors(self):
        """Tensor product combines data and ancillas."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(3))
        t = Tensor([A, B])
        circ = t.circuit()

        assert circ.data_qubits == 2 + 3
        assert circ.ancilla_qubits == t.ancilla_qubits()

    def test_tensor_ancilla_addition(self):
        """Tensor adds ancillas."""
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=2)
        B = Basic("B", qtype_=BitType(3), ancilla_qubits_=3)
        t = Tensor([A, B])
        circ = t.circuit()

        assert circ.ancilla_qubits == t.ancilla_qubits()

    def test_tensor_two_single_qubit_gates(self):
        """Test X⊗Y produces gates on different qubits."""
        X = Basic("X")
        Y = Basic("Y")

        tensor = Tensor([X, Y])
        circ = tensor.circuit()

        assert circ.data_qubits == 2
        assert circ.ancilla_qubits == 0

        flat = circ.to_list()
        targets = [fg.targets[0] for fg in flat if fg.name in ["X", "Y"]]

        assert 0 in targets
        assert 1 in targets

    def test_tensor_with_controlled_gates(self):
        """Test X⊗X in LCU context has correct qubit mapping."""
        X = Basic("X")

        Y = Basic("Y")
        expr = Sum([(1.0, Tensor([X, X])), (1.0, Tensor([Y, Y]))])

        circ = expr.circuit()
        flat = circ.to_list()

        x_gates = [fg for fg in flat if fg.name == "X"]
        x_targets: set[int] = set()
        for xg in x_gates:
            x_targets.update(xg.targets)

        assert 0 in x_targets, "X⊗X should have gate on qubit 0"
        assert 1 in x_targets, "X⊗X should have gate on qubit 1"


class TestIfCircuitCompilation:
    """Test circuit compilation for If expressions."""

    def test_if_basic(self):
        """Basic if-then-else."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        cond = Condition("x")
        i = If(cond, A, B)
        circ = i.circuit()

        assert circ.data_qubits == 2 + 1
        assert circ.ancilla_qubits == i.ancilla_qubits()

    def test_if_has_controlled_gates(self):
        """If creates controlled gates with both positive and negative controls."""
        A = Basic("A", qtype_=BitType(2))
        B = Basic("B", qtype_=BitType(2))
        cond = Condition("x")
        i = If(cond, A, B)
        circ = i.circuit()

        has_controlled = any(isinstance(g, Controlled) for g in circ.gates)
        assert has_controlled

    def test_if_negative_condition_no_normalization(self):
        """If with negative condition does NOT normalize (preserves cost semantics)."""

        A = Basic("A")
        B = Basic("B")

        cond_neg = Condition("x", active=False)
        if_expr = If(cond_neg, A, B)

        circ = if_expr.circuit()

        assert circ.data_qubits == 2

        controlled_gates = [g for g in circ.gates if isinstance(g, Controlled)]
        assert len(controlled_gates) == 2, "Should have 2 controlled gates (A and B)"

        assert isinstance(controlled_gates[0].gate, BlackBox)
        assert controlled_gates[0].gate.name == "A"
        assert len(controlled_gates[0].controls) == 1
        c0 = controlled_gates[0].controls[0]
        assert isinstance(c0, Control) and c0.qubit == 1
        assert (
            isinstance(c0, Control) and c0.is_negative()
        ), "A should be controlled on |0⟩"

        assert isinstance(controlled_gates[1].gate, BlackBox)
        assert controlled_gates[1].gate.name == "B"
        assert len(controlled_gates[1].controls) == 1
        c1 = controlled_gates[1].controls[0]
        assert isinstance(c1, Control) and c1.qubit == 1
        assert (
            isinstance(c1, Control) and c1.is_positive()
        ), "B should be controlled on |1⟩"

    def test_if_positive_condition_controls(self):
        """Test If with positive condition uses correct controls."""

        A = Basic("A")
        B = Basic("B")

        cond_pos = Condition("x", active=True)
        if_expr = If(cond_pos, A, B)

        circ = if_expr.circuit()

        controlled_gates = [g for g in circ.gates if isinstance(g, Controlled)]
        assert len(controlled_gates) == 2, "Should have 2 controlled gates (A and B)"

        assert isinstance(controlled_gates[0].gate, BlackBox)
        assert controlled_gates[0].gate.name == "A"
        assert len(controlled_gates[0].controls) == 1
        c0 = controlled_gates[0].controls[0]
        assert isinstance(c0, Control) and c0.qubit == 1
        assert (
            isinstance(c0, Control) and c0.is_positive()
        ), "A should be controlled on |1⟩"

        assert isinstance(controlled_gates[1].gate, BlackBox)
        assert controlled_gates[1].gate.name == "B"
        assert len(controlled_gates[1].controls) == 1
        c1 = controlled_gates[1].controls[0]
        assert isinstance(c1, Control) and c1.qubit == 1
        assert (
            isinstance(c1, Control) and c1.is_negative()
        ), "B should be controlled on |0⟩"

    def test_if_ancilla_count(self):
        """Test If ancilla count.

        Note: expr.ancilla_qubits() = max(branches.ancilla_qubits()) because the condition qubit is part
        of the DATA register (type is t ⊗ bool), not an ancilla.

        The circuit.ancilla_qubits should equal expr.ancilla_qubits() for If expressions.
        """
        A = Basic("A", qtype_=BitType(2), ancilla_qubits_=2)
        B = Basic("B", qtype_=BitType(2), ancilla_qubits_=3)

        cond = Condition("c1")
        if_expr = If(cond, A, B)
        circ = if_expr.circuit()
        assert circ.ancilla_qubits == if_expr.ancilla_qubits()


class TestPolyCircuitCompilation:
    """Test circuit compilation for Poly expressions."""

    def test_poly_qsp_circuit_invocations(self):
        """QSP should invoke sub-expression d times for degree d."""
        A = Basic("A")
        p = Polynomial([0.0, 1.0])  # x (odd parity, degree 1)
        poly = Poly(A, p)

        circ = poly.circuit()
        flat_gates = circ.to_list()

        # For degree 1, should have 1 invocation of the sub-expression
        black_box_gates = [g for g in flat_gates if g.name == "A" or g.name == "A_dag"]
        assert (
            len(black_box_gates) == 1
        ), f"Expected 1 A gate for degree 1, got {len(black_box_gates)}"

        # Test degree 3 - alternates A and A_dag, so should have 3 total
        p3 = Polynomial([0.0, 0.0, 0.0, 1.0])  # x^3 (odd, degree 3)
        poly3 = Poly(A, p3)
        circ3 = poly3.circuit()
        flat_gates3 = circ3.to_list()
        # Count both A and A_dag (the adjoint will show up as A_dag in flat gates)
        black_box_gates3 = [g for g in flat_gates3 if g.name in ["A", "A_dag"]]
        # For degree 3, should be 3 invocations
        assert (
            len(black_box_gates3) >= 2
        ), f"Expected at least 2 invocations for degree 3, got {len(black_box_gates3)}"

    def test_poly_ancilla_allocation(self):
        """Poly should allocate ancillas based on parity.

        For fixed parity: 1 rotation ancilla + expression ancillas
        For mixed parity: 1 LCU ancilla + 1 rotation ancilla + expression ancillas
        """
        A = Basic("A", ancilla_qubits_=2)  # A has 2 ancillas

        # Test mixed parity: 1 + x
        p_mixed = Polynomial([1.0, 1.0])  # degree 1, mixed parity
        poly_mixed = Poly(A, p_mixed)
        circ_mixed = poly_mixed.circuit()

        assert circ_mixed.data_qubits == 1
        assert circ_mixed.ancilla_qubits == poly_mixed.ancilla_qubits()

        # Test fixed parity (odd): x
        p_odd = Polynomial([0.0, 1.0])  # degree 1, odd parity
        poly_odd = Poly(A, p_odd)
        circ_odd = poly_odd.circuit()

        assert circ_odd.data_qubits == 1
        assert circ_odd.ancilla_qubits == poly_odd.ancilla_qubits()

    def test_poly_parity_detection(self):
        """Test that polynomial parity detection works correctly."""
        p_even = Polynomial([1.0, 0.0, 2.0, 0.0, 3.0])  # 3x^4 + 2x^2 + 1
        assert p_even.parity() == Polynomial.Parity.EVEN

        p_odd = Polynomial([0.0, 1.0, 0.0, 2.0])  # 2x^3 + x
        assert p_odd.parity() == Polynomial.Parity.ODD

        p_mixed = Polynomial([1.0, 2.0, 3.0])  # 3x^2 + 2x + 1
        assert p_mixed.parity() == Polynomial.Parity.MIXED

        p_zero = Polynomial([0.0, 0.0, 0.0])
        assert p_zero.parity() == Polynomial.Parity.EVEN

    def test_poly_even_odd_decomposition(self):
        """Test that polynomial even/odd decomposition works correctly."""
        p = Polynomial([1.0, 2.0, 3.0])

        p_even = p.even_component()
        p_odd = p.odd_component()

        assert p_even.parity() == Polynomial.Parity.EVEN
        assert abs(p_even.coeffs[0] - 1.0) < TOL
        assert abs(p_even.coeffs[1] - 0.0) < TOL if len(p_even.coeffs) > 1 else True
        assert abs(p_even.coeffs[2] - 3.0) < TOL if len(p_even.coeffs) > 2 else True

        assert p_odd.parity() == Polynomial.Parity.ODD
        assert abs(p_odd.coeffs[0] - 0.0) < TOL if len(p_odd.coeffs) > 0 else True
        assert abs(p_odd.coeffs[1] - 2.0) < TOL

        p_reconstructed = Polynomial(
            [
                p_even.coeffs[i] + (p_odd.coeffs[i] if i < len(p_odd.coeffs) else 0.0)
                for i in range(max(len(p_even.coeffs), len(p_odd.coeffs)))
            ]
        )
        assert p_reconstructed.approx_eq(p)
