from cobble.circuit import (
    BlackBox,
    Circuit,
    Control,
    Controlled,
    Hadamard,
    NOT,
    QubitAllocation,
    Rx,
    Ry,
    Rz,
    SWAP,
    Z,
)
from cobble.expr import Basic, Condition, Const, Dagger, If, Poly, Prod, Sum, Tensor
from cobble.polynomial import Polynomial, TOL
from cobble.qtype import BitType
from cobble.simulator import CircuitSimulator, simulate_circuit
import numpy as np
import pytest

from utils import qsp_unitary_dilation


class TestSimulator:
    """Tests for the circuit simulator."""

    def test_empty_circuit(self):
        """Test simulation of empty circuit (identity)."""

        circuit = Circuit.identity(data_qubits=2, ancilla_count=0)
        simulator = CircuitSimulator(circuit)

        # Test with |00⟩
        initial = np.array([1, 0, 0, 0], dtype=complex)
        final = simulator.simulate(initial)
        assert np.allclose(final, initial)

        # Test with superposition
        initial = np.array([1, 1, 1, 1], dtype=complex) / 2
        final = simulator.simulate(initial)
        assert np.allclose(final, initial)

    def test_single_qubit_gates(self):
        """Test basic single-qubit gates."""

        # NOT gate: |0⟩ → |1⟩
        circuit = Circuit([NOT(0)], QubitAllocation(1, []))
        initial = np.array([1, 0], dtype=complex)
        expected = np.array([0, 1], dtype=complex)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # Hadamard: |0⟩ → |+⟩
        circuit = Circuit([Hadamard(0)], QubitAllocation(1, []))
        initial = np.array([1, 0], dtype=complex)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # Z gate: |+⟩ → |−⟩
        circuit = Circuit([Z(0)], QubitAllocation(1, []))
        initial = np.array([1, 1], dtype=complex) / np.sqrt(2)
        expected = np.array([1, -1], dtype=complex) / np.sqrt(2)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

    def test_rotation_gates(self):
        """Test rotation gates."""

        # Rx(π): acts like -iX (up to global phase)
        circuit = Circuit([Rx(0, np.pi)], QubitAllocation(1, []))
        initial = np.array([1, 0], dtype=complex)
        final = simulate_circuit(circuit, initial)
        # Rx(π)|0⟩ = -i|1⟩ (up to global phase)
        expected = np.array([0, -1j], dtype=complex)
        assert np.allclose(final, expected)

        # Ry(π/2): |0⟩ → (|0⟩ + |1⟩)/√2
        circuit = Circuit([Ry(0, np.pi / 2)], QubitAllocation(1, []))
        initial = np.array([1, 0], dtype=complex)
        expected = np.array([1, 1], dtype=complex) / np.sqrt(2)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # Rz(π/2): phase gate
        circuit = Circuit([Rz(0, np.pi / 2)], QubitAllocation(1, []))
        initial = np.array([1, 1], dtype=complex) / np.sqrt(2)
        # Rz(θ)|ψ⟩ = exp(-iθ/2)|0⟩⟨0| + exp(iθ/2)|1⟩⟨1|
        expected = np.array(
            [np.exp(-1j * np.pi / 4), np.exp(1j * np.pi / 4)], dtype=complex
        ) / np.sqrt(2)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

    def test_two_qubit_gates(self):
        """Test two-qubit gates."""

        # SWAP: |01⟩ → |10⟩
        circuit = Circuit([SWAP(0, 1)], QubitAllocation(2, []))
        initial = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        expected = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

    def test_controlled_gates(self):
        """Test controlled gates."""

        # CNOT: |11⟩ → |10⟩
        circuit = Circuit([Controlled(NOT(1), (0,))], QubitAllocation(2, []))
        initial = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        expected = np.array([0, 0, 1, 0], dtype=complex)  # |10⟩
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # CNOT: |01⟩ → |01⟩ (control is 0)
        initial = np.array([0, 1, 0, 0], dtype=complex)
        expected = initial
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # Controlled-Z
        circuit = Circuit([Controlled(Z(1), (0,))], QubitAllocation(2, []))
        initial = np.array([0, 0, 0, 1], dtype=complex)  # |11⟩
        expected = np.array([0, 0, 0, -1], dtype=complex)  # -|11⟩
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

    def test_anticontrol_on_qubit_zero(self):
        """Test anti-control (control on |0⟩) on qubit 0."""

        # Anti-controlled NOT: flip qubit 1 when qubit 0 is |0⟩
        circuit = Circuit(
            [Controlled(NOT(1), (Control.neg(0),))], QubitAllocation(2, [])
        )

        # |00⟩ → |01⟩ (qubit 0 is |0⟩, so gate applies)
        initial = np.array([1, 0, 0, 0], dtype=complex)
        expected = np.array([0, 1, 0, 0], dtype=complex)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # |01⟩ → |00⟩ (qubit 0 is |0⟩, so gate applies)
        initial = np.array([0, 1, 0, 0], dtype=complex)
        expected = np.array([1, 0, 0, 0], dtype=complex)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # |10⟩ → |10⟩ (qubit 0 is |1⟩, so gate doesn't apply)
        initial = np.array([0, 0, 1, 0], dtype=complex)
        expected = initial
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # |11⟩ → |11⟩ (qubit 0 is |1⟩, so gate doesn't apply)
        initial = np.array([0, 0, 0, 1], dtype=complex)
        expected = initial
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

    def test_bell_state_creation(self):
        """Test creating a Bell state."""

        # H on qubit 0, then CNOT(0 → 1)
        circuit = Circuit(
            [Hadamard(0), Controlled(NOT(1), (0,))], QubitAllocation(2, [])
        )
        initial = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        expected = np.array([1, 0, 0, 1], dtype=complex) / np.sqrt(
            2
        )  # (|00⟩ + |11⟩)/√2
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

    def test_circuit_with_ancilla_no_entanglement(self):
        """Test circuit with ancilla that doesn't entangle with data."""

        # Apply X to ancilla only
        circuit = Circuit([NOT(1)], QubitAllocation(1, [1]))

        simulator = CircuitSimulator(circuit)

        # Initial data state: |+⟩
        initial = np.array([1, 1], dtype=complex) / np.sqrt(2)

        # After X on ancilla (which starts as |0⟩), ancilla becomes |1⟩
        # Postselection to |0⟩ should give zero amplitude
        with pytest.raises(ValueError, match="Postselection resulted in zero state"):
            simulator.simulate(initial)

    def test_circuit_with_ancilla_hadamard(self):
        """Test circuit with ancilla in superposition."""

        # Apply H to ancilla
        circuit = Circuit([Hadamard(1)], QubitAllocation(1, [1]))

        simulator = CircuitSimulator(circuit)

        # Initial data state: |0⟩
        initial = np.array([1, 0], dtype=complex)

        # After H on ancilla: (|0⟩ + |1⟩)/√2 for ancilla
        # Postselection to |0⟩ keeps data state unchanged but renormalizes
        final = simulator.simulate(initial)
        assert np.allclose(final, initial)  # Data state unchanged

    def test_explicit_block_encoding(self):
        """Test an explicitly provided block encoding matrix."""
        A = Basic("A", ancilla_qubits_=1)
        circ = A.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix() * A.subnormalization()

        expected = A_matrix

        assert np.allclose(actual, expected)

    def test_simple_block_encoding(self):
        """Test block encoding with controlled operation."""

        # Controlled-NOT from ancilla to data
        # This creates a block encoding of X/2 (up to scaling)
        circuit = Circuit(
            [Hadamard(1), Controlled(NOT(0), (1,))], QubitAllocation(1, [1])
        )

        simulator = CircuitSimulator(circuit)

        block_matrix = simulator.get_block_encoding_matrix()
        expected_matrix = np.eye(2, dtype=complex) / np.sqrt(2)
        assert np.allclose(block_matrix, expected_matrix)

    def test_controlled_rotation_block_encoding(self):
        """Test block encoding with controlled rotation."""

        # Controlled Ry rotation from ancilla to data
        # Circuit: H on ancilla, then C-Ry with ancilla as control
        angle = np.pi / 3
        circuit = Circuit(
            [Hadamard(1), Controlled(Ry(0, angle), (1,))], QubitAllocation(1, [1])
        )

        simulator = CircuitSimulator(circuit)
        block_matrix = simulator.get_block_encoding_matrix()
        expected = np.eye(2, dtype=complex) / np.sqrt(2)

        assert np.allclose(block_matrix, expected)

    def test_get_full_unitary(self):
        """Test getting full unitary matrix."""

        # Simple circuit: H on qubit 0
        circuit = Circuit([Hadamard(0)], QubitAllocation(1, []))
        simulator = CircuitSimulator(circuit)

        unitary = simulator.get_full_unitary()
        expected = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        assert np.allclose(unitary, expected)

        # Verify unitarity
        product = unitary @ unitary.conj().T
        assert np.allclose(product, np.eye(2))

    def test_full_unitary_with_ancilla(self):
        """Test full unitary with ancilla qubits."""

        # H on data, X on ancilla
        circuit = Circuit([Hadamard(0), NOT(1)], QubitAllocation(1, [1]))
        simulator = CircuitSimulator(circuit)

        unitary = simulator.get_full_unitary()

        # Should be H ⊗ X (in qubit order: data ⊗ ancilla)
        h_matrix = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        x_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        expected = np.kron(h_matrix, x_matrix)

        assert np.allclose(unitary, expected)

        # Verify unitarity
        product = unitary @ unitary.conj().T
        assert np.allclose(product, np.eye(4))

    def test_multi_ancilla_circuit(self):
        """Test circuit with multiple ancilla qubits."""

        circuit = Circuit(
            [Hadamard(2), Controlled(NOT(3), (2,))], QubitAllocation(2, [2, 3])
        )

        simulator = CircuitSimulator(circuit)

        initial = np.array([1, 0, 0, 0], dtype=complex)
        final = simulator.simulate(initial)
        assert np.allclose(final, initial)

    def test_sequential_gates(self):
        """Test multiple gates in sequence."""

        # X then X should be identity
        circuit = Circuit([NOT(0), NOT(0)], QubitAllocation(1, []))
        initial = np.array([1, 0], dtype=complex)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, initial)

        # H then H should be identity
        circuit = Circuit([Hadamard(0), Hadamard(0)], QubitAllocation(1, []))
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, initial)

    def test_multi_qubit_controlled_gate(self):
        """Test gate with multiple control qubits."""

        # Toffoli gate: CCNOT (control on qubits 0 and 1, target qubit 2)
        circuit = Circuit([Controlled(NOT(2), (0, 1))], QubitAllocation(3, []))

        # Test |110⟩ → |111⟩
        initial = np.array([0, 0, 0, 0, 0, 0, 1, 0], dtype=complex)  # |110⟩
        expected = np.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=complex)  # |111⟩
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, expected)

        # Test |100⟩ → |100⟩ (second control is 0)
        initial = np.array([0, 0, 0, 0, 1, 0, 0, 0], dtype=complex)
        final = simulate_circuit(circuit, initial)
        assert np.allclose(final, initial)

    def test_block_encoding_amplitude_amplification(self):
        """Test block encoding for amplitude amplification style circuit."""

        # Circuit: H on ancilla, controlled-Z from ancilla to data, H on ancilla
        # This creates a reflection
        circuit = Circuit(
            [Hadamard(1), Controlled(Z(0), (1,)), Hadamard(1)], QubitAllocation(1, [1])
        )

        simulator = CircuitSimulator(circuit)
        block_matrix = simulator.get_block_encoding_matrix()
        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(block_matrix, expected)

    def test_return_full_state(self):
        """Test returning full state before postselection."""

        circuit = Circuit([Hadamard(1)], QubitAllocation(1, [1]))
        simulator = CircuitSimulator(circuit)

        initial = np.array([1, 0], dtype=complex)
        full_state = simulator.simulate(initial, return_full=True)

        expected = np.array([1, 1, 0, 0], dtype=complex) / np.sqrt(2)
        assert np.allclose(full_state, expected)

    def test_invalid_input_dimension(self):
        """Test error handling for wrong input dimension."""

        circuit = Circuit.identity(2, 0)
        simulator = CircuitSimulator(circuit)

        initial = np.array([1, 0, 0], dtype=complex)
        with pytest.raises(ValueError, match="Initial vector has dimension"):
            simulator.simulate(initial)

    def test_normalization(self):
        """Test that simulator properly normalizes states."""

        circuit = Circuit([Hadamard(0)], QubitAllocation(1, []))

        initial = np.array([3, 4], dtype=complex)
        final = simulate_circuit(circuit, initial)
        assert np.abs(np.linalg.norm(final) - 1.0) < TOL

    def test_complex_circuit_composition(self):
        """Test a more complex circuit with multiple operations."""

        circuit = Circuit(
            [
                Hadamard(0),
                Controlled(NOT(1), (0,)),
                Hadamard(2),
                Controlled(Z(0), (2,)),
            ],
            QubitAllocation(2, [2]),
        )

        simulator = CircuitSimulator(circuit)

        initial = np.array([1, 0, 0, 0], dtype=complex)
        final = simulator.simulate(initial)
        assert np.abs(np.linalg.norm(final) - 1.0) < TOL

    def test_blackbox_gate_substitution(self):
        """Test providing explicit matrices for BlackBox gates."""

        circuit = Circuit([BlackBox("U", 2, 1, 0)], QubitAllocation(2, [2]))

        swap_2q = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        U_matrix = np.kron(swap_2q, np.eye(2))
        assert np.linalg.norm(U_matrix, ord=2) <= 1

        sim = CircuitSimulator(
            circuit, gate_matrices={"U": U_matrix}, ancilla_data_ordering=False
        )

        initial = np.array([0, 1, 0, 0], dtype=complex)
        final = sim.simulate(initial)

        expected = np.array([0, 0, 1, 0], dtype=complex)
        assert np.allclose(final, expected)

    def test_block_encoding_product_composition(self):
        """Test that product of block encodings with separate ancillas composes correctly."""

        circuit1 = Circuit(
            [Hadamard(1), Controlled(NOT(0), (1,))], QubitAllocation(1, [1])
        )

        circuit2 = Circuit(
            [Hadamard(1), Controlled(Z(0), (1,))],
            QubitAllocation(1, [1]),
        )

        sim1 = CircuitSimulator(circuit1)
        sim2 = CircuitSimulator(circuit2)

        A1 = sim1.get_block_encoding_matrix()
        A2 = sim2.get_block_encoding_matrix()

        composed = Circuit(
            [
                Hadamard(1),
                Controlled(NOT(0), (1,)),
                Hadamard(2),
                Controlled(Z(0), (2,)),
            ],
            QubitAllocation(1, [1, 2]),
        )
        sim_composed = CircuitSimulator(composed)
        A_composed = sim_composed.get_block_encoding_matrix()

        expected = A2 @ A1

        assert np.allclose(A_composed, expected)

    def test_block_encoding_subnormalization(self):
        """Test that block encodings have expected subnormalization."""

        circuit = Circuit([Hadamard(1)], QubitAllocation(1, [1]))
        sim = CircuitSimulator(circuit)
        A = sim.get_block_encoding_matrix()

        singular_values = np.linalg.svd(A, compute_uv=False)
        operator_norm = np.max(singular_values)

        expected_norm = 1.0 / np.sqrt(2)
        assert np.abs(operator_norm - expected_norm) < TOL

    def test_block_encoding_with_multiple_ancillas(self):
        """Test block encoding with multiple ancilla qubits."""

        circuit = Circuit([Hadamard(1), Hadamard(2)], QubitAllocation(1, [1, 2]))

        sim = CircuitSimulator(circuit)
        A = sim.get_block_encoding_matrix()

        expected = np.eye(2, dtype=complex) / 2.0
        assert np.allclose(A, expected)

        singular_values = np.linalg.svd(A, compute_uv=False)
        operator_norm = np.max(singular_values)
        expected_norm = 0.5
        assert np.abs(operator_norm - expected_norm) < TOL

    def test_simulate_vs_block_encoding(self):
        """Test that simulation is consistent with block encoding."""

        circuit = Circuit(
            [Hadamard(2), Controlled(NOT(0), (2,)), Controlled(Z(1), (2,))],
            QubitAllocation(2, [2]),
        )

        sim = CircuitSimulator(circuit)
        A = sim.get_block_encoding_matrix()

        test_states = [
            np.array([1, 0, 0, 0], dtype=complex),  # |00⟩
            np.array([0, 1, 0, 0], dtype=complex),  # |01⟩
            np.array([0, 0, 1, 0], dtype=complex),  # |10⟩
            np.array([0, 0, 0, 1], dtype=complex),  # |11⟩
            np.array([1, 1, 1, 1], dtype=complex) / 2,  # uniform superposition
        ]

        for initial in test_states:
            final_sim = sim.simulate(initial)
            final_matrix = A @ initial
            final_sim_norm = final_sim / np.linalg.norm(final_sim)
            final_matrix_norm = final_matrix / np.linalg.norm(final_matrix)

            assert np.allclose(final_sim_norm, final_matrix_norm)

    def test_block_encoding_adjoint(self):
        """Test that adjoint of block encoding works correctly."""

        angle = np.pi / 4
        circuit = Circuit(
            [Hadamard(1), Controlled(Ry(0, angle), (1,))], QubitAllocation(1, [1])
        )

        circuit_dag = circuit.adjoint()

        sim = CircuitSimulator(circuit)
        sim_dag = CircuitSimulator(circuit_dag)

        A = sim.get_block_encoding_matrix()
        A_dag = sim_dag.get_block_encoding_matrix()

        expected_A_dag = A.conj().T
        assert np.allclose(A_dag, expected_A_dag)


class TestExprBlockEncodings:
    """Tests for block encodings compiled from expression language."""

    def test_uninstantiated_base_raises_error(self):
        """Test that CircuitSimulator refuses uninstantiated Base expressions."""

        A = Basic("A")
        circ = A.circuit()

        with pytest.raises(
            ValueError, match="BlackBox gate 'A' requires an explicit matrix"
        ):
            CircuitSimulator(circ)

    def test_base_identity_block_encoding(self):
        """Test that Base with no ancillas produces correct block encoding."""

        A = Basic("A", subnormalization_=1.7)
        circ = A.circuit()

        assert circ.ancilla_qubits == 0

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix})
        actual = sim.get_block_encoding_matrix()

        expected = A_matrix * A.subnormalization_
        assert np.allclose(actual * A.subnormalization(), expected)

        assert abs(A.subnormalization() - A.subnormalization_) < TOL

    def test_base_with_ancillas_subnormalization(self):
        """Test that Base with ancillas has correct subnormalization."""

        A = Basic("A", ancilla_qubits_=1)
        circ = A.circuit()

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        assert np.linalg.norm(CNOT, ord=2) <= A.subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"A": CNOT}, ancilla_data_ordering=False
        )
        actual = sim.get_block_encoding_matrix()

        expected = np.array([[1, 0], [0, 0]], dtype=complex)
        assert np.allclose(actual * A.subnormalization(), expected)

        assert abs(A.subnormalization() - 1.0) < TOL

    def test_dagger_block_encoding(self):
        """Test that Dagger expression produces adjoint block encoding."""

        A = Basic("A", subnormalization_=3, ancilla_qubits_=1)
        A_dag = Dagger(A)

        circ_A = A.circuit()
        circ_A_dag = A_dag.circuit()

        A_matrix = np.array(
            [[1, 1, 0, 0], [0, 2, 1, 0], [0, 0, 1, 1], [0, 0, 0, 2]], dtype=complex
        )
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        A_matrix_dag = A_matrix.conj().T

        sim_A = CircuitSimulator(
            circ_A, gate_matrices={"A": A_matrix}, ancilla_data_ordering=False
        )
        sim_A_dag = CircuitSimulator(
            circ_A_dag,
            gate_matrices={"A_dag": A_matrix_dag},
            ancilla_data_ordering=False,
        )

        actual_A = sim_A.get_block_encoding_matrix()
        actual_A_dag = sim_A_dag.get_block_encoding_matrix()

        assert np.allclose(
            actual_A_dag * A_dag.subnormalization(),
            (actual_A * A.subnormalization()).conj().T,
        )

        assert abs(A.subnormalization() - A.subnormalization_) < TOL
        assert abs(A_dag.subnormalization() - A.subnormalization_) < TOL

    def test_sum_negation(self):
        X = Basic("X")

        X_matrix = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_

        gate_matrices = {"X": X_matrix}

        prog = Sum([(-1.0, X)])
        expected = -X_matrix

        circ_unnorm = prog.circuit()
        sim_unnorm = CircuitSimulator(circ_unnorm, gate_matrices=gate_matrices)
        matrix_unnorm = sim_unnorm.get_block_encoding_matrix() * prog.subnormalization()

        assert np.allclose(matrix_unnorm, expected)

    def test_sum_two_terms_block_encoding(self):
        """Test Sum with two terms produces correct linear combination."""

        A = Basic("A", subnormalization_=1.7)
        B = Basic("B", subnormalization_=1.3)

        s = Sum([(0.3, A), (0.7, B)])

        circ = s.circuit()

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})
        actual = sim.get_block_encoding_matrix()

        assert actual.shape == (2, 2)

        expected = (
            0.3 * A.subnormalization_ * A_matrix + 0.7 * B.subnormalization_ * B_matrix
        )
        assert np.allclose(actual * s.subnormalization(), expected)

        assert (
            abs(
                s.subnormalization()
                - (0.3 * A.subnormalization_ + 0.7 * B.subnormalization_)
            )
            < TOL
        )

    def test_sum_weighted_coefficients(self):
        """Test Sum with weighted coefficients produces correct linear combination."""

        A = Basic("A", subnormalization_=2.0)
        B = Basic("B", subnormalization_=1.5)

        s = Sum([(0.6, A), (0.4, B)])
        circ = s.circuit()

        A_matrix = np.array([[1, 0], [0, 0]], dtype=complex)  # |0⟩⟨0|
        B_matrix = np.array([[0, 0], [0, 1]], dtype=complex)  # |1⟩⟨1|
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})
        actual = sim.get_block_encoding_matrix()

        expected = (
            0.6 * A.subnormalization_ * A_matrix + 0.4 * B.subnormalization_ * B_matrix
        )
        assert np.allclose(actual * s.subnormalization(), expected)

        assert (
            abs(
                s.subnormalization()
                - (0.6 * A.subnormalization_ + 0.4 * B.subnormalization_)
            )
            < TOL
        )

    def test_sum_explicit_ancillas(self):
        """Test Sum with explicit matrix and ancillas."""
        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)
        sum_expr = Sum([(1, A), (1, B)])

        circ = sum_expr.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.4, 0.8], [0.8, -0.4]], dtype=complex)
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_
        U_B = qsp_unitary_dilation(B_matrix)
        assert np.allclose(U_B @ U_B.conj().T, np.eye(4))

        sim = CircuitSimulator(
            circ,
            gate_matrices={
                "A": U_A,
                "B": U_B,
            },
        )
        actual = sim.get_block_encoding_matrix()

        S = A_matrix + B_matrix
        expected = S

        assert np.allclose(actual * sum_expr.subnormalization(), expected)

    def test_prod_two_factors(self):
        """Test Prod with two factors produces correct product matrix."""

        A = Basic("A", subnormalization_=1.7)
        B = Basic("B", subnormalization_=1.9)

        p = Prod([A, B])
        circ = p.circuit()

        assert circ.data_qubits == 1
        assert circ.ancilla_qubits >= 1
        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        B_matrix = np.array([[0.3, 0], [0, -0.5]], dtype=complex)  # Pauli Z
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})
        actual = sim.get_block_encoding_matrix()

        assert actual.shape == (2, 2)

        expected = (A_matrix * A.subnormalization_) @ (B_matrix * B.subnormalization_)
        assert np.allclose(actual * p.subnormalization(), expected)

        assert (
            abs(p.subnormalization() - (A.subnormalization_ * B.subnormalization_))
            < TOL
        )

    def test_prod_with_negative_constant(self):
        X = Basic("X")

        X_matrix = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_

        prog = Prod([Const(-1.0), X])
        expected = -X_matrix

        circ_unnorm = prog.circuit()
        sim_unnorm = CircuitSimulator(circ_unnorm, gate_matrices={"X": X_matrix})
        matrix_unnorm = sim_unnorm.get_block_encoding_matrix() * prog.subnormalization()

        assert np.allclose(matrix_unnorm, expected)

    def test_prod_explicit_ancillas(self):
        """Test Prod where sub-expression has ancillas."""
        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)
        poly = (A + B) * (A + B)
        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.4, 0.8], [0.8, -0.4]], dtype=complex)
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_
        U_B = qsp_unitary_dilation(B_matrix)
        assert np.allclose(U_B @ U_B.conj().T, np.eye(4))

        sim = CircuitSimulator(
            circ,
            gate_matrices={
                "A": U_A,
                "B": U_B,
            },
        )
        actual = sim.get_block_encoding_matrix()
        S = A_matrix + B_matrix
        expected = S @ S

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_if_expression_block_encoding(self):
        """Test If expression produces correct conditional block encoding."""

        A = Basic("A")
        B = Basic("B")

        cond = Condition("x", active=True)
        if_expr = If(cond, A, B)

        circ = if_expr.circuit()

        assert circ.data_qubits == 2

        ancilla_qubits = circ.ancilla_qubits

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)

        A_full = np.kron(X, np.eye(2**ancilla_qubits))
        B_full = np.kron(Z, np.eye(2**ancilla_qubits))
        assert np.linalg.norm(A_full, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_full, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"A": A_full, "B": B_full}, ancilla_data_ordering=False
        )
        matrix = sim.get_block_encoding_matrix()

        assert matrix.shape == (4, 4)

        state_00 = np.array([1, 0, 0, 0], dtype=complex)  # |00⟩
        result_00 = matrix @ state_00
        expected_00 = np.array([1, 0, 0, 0], dtype=complex)
        assert np.linalg.norm(result_00 - expected_00) < TOL

        state_01 = np.array([0, 1, 0, 0], dtype=complex)  # |01⟩
        result_01 = matrix @ state_01
        expected_01 = np.array([0, 0, 0, 1], dtype=complex)
        assert np.linalg.norm(result_01 - expected_01) < TOL

        assert (
            abs(
                if_expr.subnormalization()
                - max(A.subnormalization(), B.subnormalization())
            )
            < TOL
        )

    def test_nested_expressions(self):
        """Test nested expressions produce correct composed matrix."""

        A = Basic("A", subnormalization_=1.5)
        B = Basic("B", subnormalization_=1.4)
        C = Basic("C", subnormalization_=1.8)

        sum_expr = Sum([(0.5, A), (0.5, B)])
        prod_expr = Prod([sum_expr, C])

        circ = prod_expr.circuit()

        assert circ.data_qubits == 1
        assert circ.ancilla_qubits > 0

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        B_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        C_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_
        assert np.linalg.norm(C_matrix, ord=2) <= C.subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"A": A_matrix, "B": B_matrix, "C": C_matrix}
        )
        actual = sim.get_block_encoding_matrix()

        assert actual.shape == (2, 2)

        expected = (
            0.5 * A.subnormalization_ * A_matrix + 0.5 * B.subnormalization_ * B_matrix
        ) @ (C.subnormalization_ * C_matrix)
        assert np.allclose(actual * prod_expr.subnormalization(), expected)

        assert (
            abs(
                prod_expr.subnormalization()
                - (0.5 * A.subnormalization_ + 0.5 * B.subnormalization_)
                * C.subnormalization_
            )
            < TOL
        )

    def test_expression_hermiticity_preserved(self):
        """Test that Hermitian expressions compile to Hermitian block encodings."""

        A = Basic("A", subnormalization_=1.5)
        A_dag = Dagger(A)

        hermitian_expr = Sum([(0.5, A), (0.5, A_dag)])

        circ = hermitian_expr.circuit()

        A_matrix = np.array([[1, 2], [0, 1]], dtype=complex) / np.sqrt(5)
        A_dag_matrix = A_matrix.conj().T
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"A": A_matrix, "A_dag": A_dag_matrix}
        )
        matrix = sim.get_block_encoding_matrix()

        matrix_dag = matrix.conj().T
        assert np.allclose(matrix, matrix_dag)

    def test_multi_qubit_expression(self):
        """Test expression with multiple data qubits."""

        A = Basic("A", subnormalization_=1.6, qtype_=BitType(2))
        B = Basic("B", subnormalization_=1.4, qtype_=BitType(2))

        s = Sum([(0.6, A), (0.4, B)])

        circ = s.circuit()
        assert circ.data_qubits == 2

        A_matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )  # CNOT
        B_matrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )  # CZ
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})
        actual = sim.get_block_encoding_matrix()

        assert actual.shape == (4, 4)

        expected = (
            0.6 * A.subnormalization_ * A_matrix + 0.4 * B.subnormalization_ * B_matrix
        )
        assert np.allclose(actual * s.subnormalization(), expected)

        assert (
            abs(
                s.subnormalization()
                - (0.6 * A.subnormalization_ + 0.4 * B.subnormalization_)
            )
            < TOL
        )

    def test_tensor_expression(self):
        """Test Tensor expression produces correct tensor product."""

        A = Basic("A", subnormalization_=1.6)
        B = Basic("B", subnormalization_=1.7)

        t = Tensor([A, B])

        circ = t.circuit()

        assert circ.data_qubits == 2

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)  # Pauli X
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)  # Pauli Z
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix, "B": B_matrix})
        actual = sim.get_block_encoding_matrix()

        assert actual.shape == (4, 4)

        expected = np.kron(
            A_matrix * A.subnormalization_, B_matrix * B.subnormalization_
        )
        assert np.allclose(actual * t.subnormalization(), expected)

        assert (
            abs(t.subnormalization() - (A.subnormalization_ * B.subnormalization_))
            < TOL
        )

    def test_simulate_sum_on_basis_states(self):
        """Test simulating Sum expression on computational basis states."""

        A = Basic("A", subnormalization_=1.6)
        B = Basic("B", subnormalization_=1.8)
        s = Sum([(0.4, A), (0.6, B)])

        circ = s.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Z, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": X, "B": Z})

        v0 = np.array([1, 0], dtype=complex)
        v1 = np.array([0, 1], dtype=complex)

        result_0 = sim.simulate(v0)
        result_1 = sim.simulate(v1)

        block_matrix = 0.4 * A.subnormalization_ * X + 0.6 * B.subnormalization_ * Z
        expected_0_unnorm = block_matrix @ v0
        expected_1_unnorm = block_matrix @ v1
        expected_0 = expected_0_unnorm / np.linalg.norm(expected_0_unnorm)
        expected_1 = expected_1_unnorm / np.linalg.norm(expected_1_unnorm)

        assert np.allclose(result_0, expected_0)
        assert np.allclose(result_1, expected_1)

    def test_simulate_sum_on_superposition(self):
        """Test simulating Sum expression on superposition states."""

        A = Basic("A", subnormalization_=1.7)
        B = Basic("B", subnormalization_=1.5)
        s = Sum([(0.5, A), (0.5, B)])

        circ = s.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Y, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": X, "B": Y})

        np.random.seed(42)
        initial_state = np.random.randn(2) + 1j * np.random.randn(2)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        block_matrix = 0.5 * A.subnormalization_ * X + 0.5 * B.subnormalization_ * Y
        expected_unnorm = block_matrix @ initial_state
        expected = expected_unnorm / np.linalg.norm(expected_unnorm)

        assert np.allclose(result, expected)

    def test_simulate_prod_two_factors(self):
        """Test simulating Prod expression."""

        A = Basic("A", subnormalization_=1.8)
        B = Basic("B", subnormalization_=1.6)
        p = Prod([A, B])

        circ = p.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Z, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": X, "B": Z})

        np.random.seed(123)
        initial_state = np.random.randn(2) + 1j * np.random.randn(2)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        block_matrix = (A.subnormalization_ * X) @ (B.subnormalization_ * Z)
        expected_unnorm = block_matrix @ initial_state
        expected = expected_unnorm / np.linalg.norm(expected_unnorm)

        assert np.allclose(result, expected)

    def test_simulate_nested_expression(self):
        """Test simulating nested C*(A+B) expression."""

        A = Basic("A", subnormalization_=1.6)
        B = Basic("B", subnormalization_=1.4)
        C = Basic("C", subnormalization_=1.9)

        sum_expr = Sum([(0.5, A), (0.5, B)])
        prod_expr = Prod([sum_expr, C])

        circ = prod_expr.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Y, ord=2) <= B.subnormalization_
        assert np.linalg.norm(Z, ord=2) <= C.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": X, "B": Y, "C": Z})

        np.random.seed(456)
        initial_state = np.random.randn(2) + 1j * np.random.randn(2)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        sum_matrix = 0.5 * A.subnormalization_ * X + 0.5 * B.subnormalization_ * Y
        block_matrix = sum_matrix @ (C.subnormalization_ * Z)
        expected_unnorm = block_matrix @ initial_state
        expected = expected_unnorm / np.linalg.norm(expected_unnorm)

        assert np.allclose(result, expected)

    def test_simulate_tensor_expression(self):
        """Test simulating Tensor expression on 2-qubit states."""

        A = Basic("A", subnormalization_=1.7)
        B = Basic("B", subnormalization_=1.5)
        t = Tensor([A, B])

        circ = t.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Z, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": X, "B": Z})

        np.random.seed(789)
        initial_state = np.random.randn(4) + 1j * np.random.randn(4)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        block_matrix = np.kron(A.subnormalization_ * X, B.subnormalization_ * Z)
        expected_unnorm = block_matrix @ initial_state
        expected = expected_unnorm / np.linalg.norm(expected_unnorm)

        assert np.allclose(result, expected)

    def test_simulate_if_expression(self):
        """Test simulating If expression with condition qubit."""

        A = Basic("A", subnormalization_=1.8)
        B = Basic("B", subnormalization_=1.6)

        cond = Condition("x", active=True)
        if_expr = If(cond, A, B)

        circ = if_expr.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Z, ord=2) <= B.subnormalization_

        sim = CircuitSimulator(
            circ,
            gate_matrices={"A": X * A.subnormalization_, "B": Z * B.subnormalization_},
        )

        np.random.seed(321)
        initial_state = np.random.randn(4) + 1j * np.random.randn(4)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        P0 = np.array([[1, 0], [0, 0]], dtype=complex)  # Project condition to |0⟩
        P1 = np.array([[0, 0], [0, 1]], dtype=complex)  # Project condition to |1⟩

        block_matrix = np.kron(Z * B.subnormalization_, P0) + np.kron(
            X * A.subnormalization_, P1
        )

        expected = block_matrix @ initial_state
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(result, expected)

    def test_simulate_multiple_random_vectors(self):
        """Test simulation on multiple random vectors for consistency."""

        A = Basic("A", subnormalization_=1.75)
        B = Basic("B", subnormalization_=1.55)
        C = Basic("C", subnormalization_=1.85)

        expr = Prod([Sum([(0.3, A), (0.7, B)]), C])
        circ = expr.circuit()

        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.linalg.norm(H, ord=2) <= A.subnormalization_
        assert np.linalg.norm(X, ord=2) <= B.subnormalization_
        assert np.linalg.norm(Y, ord=2) <= C.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": H, "B": X, "C": Y})
        block_matrix = sim.get_block_encoding_matrix()

        np.random.seed(999)
        for _ in range(5):
            initial_state = np.random.randn(2) + 1j * np.random.randn(2)
            initial_state = initial_state / np.linalg.norm(initial_state)

            result = sim.simulate(initial_state)
            sum_matrix = 0.3 * A.subnormalization_ * H + 0.7 * B.subnormalization_ * X
            expected_unnorm = sum_matrix @ (C.subnormalization_ * Y) @ initial_state
            expected = expected_unnorm / np.linalg.norm(expected_unnorm)

            assert np.allclose(result, expected)

    def test_simulate_normalization_check(self):
        """Verify that simulated states maintain proper normalization."""

        A = Basic("A", subnormalization_=1.6)
        B = Basic("B", subnormalization_=1.4)
        s = Sum([(0.5, A), (0.5, B)])

        circ = s.circuit()

        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(X, ord=2) <= A.subnormalization_
        assert np.linalg.norm(Z, ord=2) <= B.subnormalization_
        sim = CircuitSimulator(circ, gate_matrices={"A": X, "B": Z})

        np.random.seed(111)
        initial_state = np.random.randn(2) + 1j * np.random.randn(2)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        norm = np.linalg.norm(result)
        assert abs(norm - 1.0) < TOL, f"Result not normalized: norm={norm}"

    def test_simulate_dagger_reverses_operation(self):
        """Test that simulating A† after A (approximately) reverses the operation."""

        A = Basic("A", subnormalization_=2.0)
        A_dag = Dagger(A)

        circ_A = A.circuit()
        circ_A_dag = A_dag.circuit()

        Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.linalg.norm(Y, ord=2) <= A.subnormalization_

        sim_A = CircuitSimulator(circ_A, gate_matrices={"A": Y})
        sim_A_dag = CircuitSimulator(circ_A_dag, gate_matrices={"A_dag": Y.conj().T})

        np.random.seed(222)
        initial_state = np.random.randn(2) + 1j * np.random.randn(2)
        initial_state = initial_state / np.linalg.norm(initial_state)

        after_A = sim_A.simulate(initial_state)
        after_A_dag_A = sim_A_dag.simulate(after_A)

        assert np.allclose(after_A_dag_A, initial_state)

    def test_large_tensor_product_matrix(self):
        """Test large block encoding with 8 qubits via tensor products."""

        bases = [
            Basic(
                f"A{i}",
                subnormalization_=1.6 + i * 0.1,
                qtype_=BitType(2),
            )
            for i in range(4)
        ]

        expr = Tensor(bases)
        circ = expr.circuit()

        assert circ.data_qubits == 8
        assert circ.ancilla_qubits == 0

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        CZ = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )
        SWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        # Controlled-Y
        CY = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, -1j], [0, 0, 1j, 0]], dtype=complex
        )
        assert np.linalg.norm(CNOT, ord=2) <= bases[0].subnormalization_
        assert np.linalg.norm(CZ, ord=2) <= bases[1].subnormalization_
        assert np.linalg.norm(SWAP, ord=2) <= bases[2].subnormalization_
        assert np.linalg.norm(CY, ord=2) <= bases[3].subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"A0": CNOT, "A1": CZ, "A2": SWAP, "A3": CY}
        )
        matrix = sim.get_block_encoding_matrix()

        assert matrix.shape == (256, 256)

        expected = np.kron(
            np.kron(
                np.kron(
                    CNOT * bases[0].subnormalization_, CZ * bases[1].subnormalization_
                ),
                SWAP * bases[2].subnormalization_,
            ),
            CY * bases[3].subnormalization_,
        )
        assert np.allclose(matrix * expr.subnormalization(), expected)

        expected_alpha = (
            bases[0].subnormalization_
            * bases[1].subnormalization_
            * bases[2].subnormalization_
            * bases[3].subnormalization_
        )
        assert abs(expr.subnormalization() - expected_alpha) < TOL

    def test_large_tensor_product_simulation(self):
        """Test simulation of 8-qubit tensor product on random state."""

        bases = [
            Basic(
                f"B{i}",
                subnormalization_=1.7 + i * 0.05,
                qtype_=BitType(2),
            )
            for i in range(4)
        ]

        expr = Tensor(bases)
        circ = expr.circuit()

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        CZ = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        HH = np.kron(H, H)
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        XZ = np.kron(X, Z)
        assert np.linalg.norm(CNOT, ord=2) <= bases[0].subnormalization_
        assert np.linalg.norm(CZ, ord=2) <= bases[1].subnormalization_
        assert np.linalg.norm(HH, ord=2) <= bases[2].subnormalization_
        assert np.linalg.norm(XZ, ord=2) <= bases[3].subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"B0": CNOT, "B1": CZ, "B2": HH, "B3": XZ}
        )

        np.random.seed(1000)
        initial_state = np.random.randn(256) + 1j * np.random.randn(256)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        block_matrix = sim.get_block_encoding_matrix()
        expected = block_matrix @ initial_state
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(result, expected)

    def test_nested_sum_prod_8qubits_matrix(self):
        """Test complex nested (A+B)*(C+D) with tensor to reach 8 qubits."""

        A = Basic("A", subnormalization_=1.5, qtype_=BitType(2))
        B = Basic("B", subnormalization_=1.4, qtype_=BitType(2))
        C = Basic("C", subnormalization_=1.6, qtype_=BitType(2))
        D = Basic("D", subnormalization_=1.7, qtype_=BitType(2))

        sum1 = Sum([(0.4, A), (0.6, B)])
        sum2 = Sum([(0.5, C), (0.5, D)])
        prod = Prod([sum1, sum2])

        E = Basic("E", subnormalization_=1.8, qtype_=BitType(2))
        F = Basic("F", subnormalization_=1.9, qtype_=BitType(2))

        expr = Tensor([prod, E, F])
        circ = expr.circuit()

        assert circ.data_qubits == 6

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        CZ = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )
        SWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        iSWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1j, 0], [0, 1j, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        CH = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, H[0, 0], H[0, 1]],
                [0, 0, H[1, 0], H[1, 1]],
            ],
            dtype=complex,
        )
        theta = np.pi / 4
        RY_tensor = np.kron(
            np.array(
                [
                    [np.cos(theta / 2), -np.sin(theta / 2)],
                    [np.sin(theta / 2), np.cos(theta / 2)],
                ],
                dtype=complex,
            ),
            np.eye(2),
        )
        assert np.linalg.norm(CNOT, ord=2) <= A.subnormalization_
        assert np.linalg.norm(CZ, ord=2) <= B.subnormalization_
        assert np.linalg.norm(SWAP, ord=2) <= C.subnormalization_
        assert np.linalg.norm(iSWAP, ord=2) <= D.subnormalization_
        assert np.linalg.norm(CH, ord=2) <= E.subnormalization_
        assert np.linalg.norm(RY_tensor, ord=2) <= F.subnormalization_

        sim = CircuitSimulator(
            circ,
            gate_matrices={
                "A": CNOT,
                "B": CZ,
                "C": SWAP,
                "D": iSWAP,
                "E": CH,
                "F": RY_tensor,
            },
        )

        matrix = sim.get_block_encoding_matrix()
        assert matrix.shape == (64, 64)  # 2^6 = 64

        sum1_alpha = 0.4 * A.subnormalization_ + 0.6 * B.subnormalization_
        sum2_alpha = 0.5 * C.subnormalization_ + 0.5 * D.subnormalization_
        prod_alpha = sum1_alpha * sum2_alpha
        expr_alpha = prod_alpha * E.subnormalization_ * F.subnormalization_
        assert abs(expr.subnormalization() - expr_alpha) < TOL

    def test_nested_sum_prod_8qubits_simulation(self):
        """Test simulation of complex nested expression on 8-qubit random state."""

        A = Basic("A", subnormalization_=1.6, qtype_=BitType(2))
        B = Basic("B", subnormalization_=1.5, qtype_=BitType(2))
        C = Basic("C", subnormalization_=1.7, qtype_=BitType(2))

        D = Basic("D", subnormalization_=1.8, qtype_=BitType(4))

        sum_expr = Sum([(0.6, A), (0.4, B)])
        prod_expr = Prod([sum_expr, C])
        expr = Tensor([prod_expr, D])

        circ = expr.circuit()
        assert circ.data_qubits == 6

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        CZ = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )
        SWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        assert np.linalg.norm(CNOT, ord=2) <= A.subnormalization_
        assert np.linalg.norm(CZ, ord=2) <= B.subnormalization_
        assert np.linalg.norm(SWAP, ord=2) <= C.subnormalization_

        D_matrix = np.kron(CNOT, SWAP)
        assert np.linalg.norm(D_matrix, ord=2) <= D.subnormalization_

        sim = CircuitSimulator(
            circ, gate_matrices={"A": CNOT, "B": CZ, "C": SWAP, "D": D_matrix}
        )

        np.random.seed(2000)
        initial_state = np.random.randn(64) + 1j * np.random.randn(64)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        block_matrix = sim.get_block_encoding_matrix()
        expected = block_matrix @ initial_state
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(result, expected)

        assert abs(np.linalg.norm(result) - 1.0) < TOL

    @pytest.mark.slow
    def test_deeply_nested_products_10qubits(self):
        """Test deeply nested product structure with 10 qubits."""

        A = Basic("A", subnormalization_=1.9, qtype_=BitType(2))
        B = Basic("B", subnormalization_=1.85, qtype_=BitType(2))
        C = Basic("C", subnormalization_=1.8, qtype_=BitType(2))
        D = Basic("D", subnormalization_=1.75, qtype_=BitType(2))
        E = Basic("E", subnormalization_=1.7, qtype_=BitType(2))

        expr = Tensor([A, B, C, D, E])
        circ = expr.circuit()

        assert circ.data_qubits == 10

        CNOT = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex
        )
        CZ = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex
        )
        SWAP = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex
        )
        X = np.array([[0, 1], [1, 0]], dtype=complex)
        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        XZ = np.kron(X, Z)

        H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
        HH = np.kron(H, H)

        assert np.linalg.norm(CNOT, ord=2) <= A.subnormalization_
        assert np.linalg.norm(CZ, ord=2) <= B.subnormalization_
        assert np.linalg.norm(SWAP, ord=2) <= C.subnormalization_
        assert np.linalg.norm(XZ, ord=2) <= D.subnormalization_
        assert np.linalg.norm(HH, ord=2) <= E.subnormalization_

        sim = CircuitSimulator(
            circ,
            gate_matrices={"A": CNOT, "B": CZ, "C": SWAP, "D": XZ, "E": HH},
        )

        matrix = sim.get_block_encoding_matrix()
        assert matrix.shape == (1024, 1024)  # 2^10

        np.random.seed(3000)
        initial_state = np.random.randn(1024) + 1j * np.random.randn(1024)
        initial_state = initial_state / np.linalg.norm(initial_state)

        result = sim.simulate(initial_state)

        expected = matrix @ initial_state
        expected = expected / np.linalg.norm(expected)

        assert np.allclose(result, expected)

        assert abs(np.linalg.norm(result) - 1.0) < TOL

        expected_alpha = (
            A.subnormalization_
            * B.subnormalization_
            * C.subnormalization_
            * D.subnormalization_
            * E.subnormalization_
        )
        assert abs(expr.subnormalization() - expected_alpha) < TOL

    def test_mixed_sum_prod_if_12qubits(self):
        """Test extremely complex nested expression with If, Sum, Prod, and Tensor."""

        A1 = Basic("A1", subnormalization_=1.8, qtype_=BitType(2))
        A2 = Basic("A2", subnormalization_=1.7, qtype_=BitType(2))
        B1 = Basic("B1", subnormalization_=1.6, qtype_=BitType(2))
        B2 = Basic("B2", subnormalization_=1.9, qtype_=BitType(2))

        sum_A = Sum([(0.5, A1), (0.5, A2)])
        sum_B = Sum([(0.4, B1), (0.6, B2)])

        prod = Prod([sum_A, sum_B])

        C1 = Basic("C1", subnormalization_=1.75, qtype_=BitType(2))
        C2 = Basic("C2", subnormalization_=1.75, qtype_=BitType(2))
        cond = Condition("x", active=True)
        if_expr = If(cond, C1, C2)

        D = Basic("D", subnormalization_=1.95, qtype_=BitType(3))
        E = Basic("E", subnormalization_=1.88, qtype_=BitType(2))

        expr = Tensor([prod, if_expr, D, E])
        circ = expr.circuit()

        assert circ.data_qubits == 10
        assert circ.allocation.total_qubits() >= 10

    def test_poly_identity_polynomial(self):
        """Test Poly with p(x) = x produces the identity transformation."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.0, 1.0])  # p(x) = x
        poly = Poly(A, p)

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        expected = A_matrix
        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_constant_polynomial(self):
        """Test Poly with p(x) = c produces c*I."""
        A = Basic("A", subnormalization_=1.5)
        p = Polynomial([0.3])  # p(x) = 0.3
        poly = Poly(A, p)

        poly_normalized = poly.optimize()
        assert isinstance(poly_normalized, Const)
        assert abs(poly_normalized.value - 0.3) < TOL

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix})
        actual = sim.get_block_encoding_matrix()

        expected = np.eye(2) * 0.3
        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_negative_identity(self):
        """Test that negative identity polynomial produces correct matrix."""

        A = Basic("A")
        prog = Poly(A, Polynomial([0, -1]))

        A_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.allclose(A_matrix @ A_matrix.conj().T, np.eye(2))

        circ = prog.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix})
        matrix = sim.get_block_encoding_matrix() * prog.subnormalization()
        expected = -A_matrix

        assert np.allclose(matrix, expected)

    def test_poly_square_polynomial(self):
        """Test Poly with p(x) = x^2 produces A^2."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.0, 0.0, 1.0])  # p(x) = x^2
        poly = Poly(A, p)

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))
        assert np.allclose(U_A, U_A.conj().T)

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        expected = A_matrix @ A_matrix

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_cubic_basic(self):
        """Test Poly with polynomial p(x) = x^3 where x is unitary."""
        A = Basic(
            "A",
        )
        p = Polynomial([0.0, 0.0, 0.0, 1.0])  # p(x) = x^3
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.ODD

        circ = poly.circuit()

        A_matrix = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix})
        actual = sim.get_block_encoding_matrix()

        expected = A_matrix @ A_matrix @ A_matrix

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_cube_polynomial(self):
        """Test Poly with polynomial p(x) = x^3."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.0, 0.0, 0.0, 1.0])  # p(x) = x^3
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.ODD

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        expected = A_matrix @ A_matrix @ A_matrix

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_fourth_polynomial(self):
        """Test Poly with polynomial p(x) = x^4."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.0, 0.0, 0.0, 0.0, 1.0])  # p(x) = x^4
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.EVEN

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        expected = A_matrix @ A_matrix @ A_matrix @ A_matrix

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_linear_mixed_basic_polynomial(self):
        """Test Poly with mixed parity polynomia where the base is unitary."""
        A = Basic(
            "A",
        )
        p = Polynomial([1, 1])  # p(x) = 0.3x + 0.2
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.MIXED

        circ = poly.circuit()

        A_matrix = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"A": A_matrix})
        actual = sim.get_block_encoding_matrix()

        I = np.eye(2, dtype=complex)
        expected = A_matrix + I

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_linear_mixed_polynomial(self):
        """Test Poly with mixed parity polynomial."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.2, 0.3])  # p(x) = 0.3x + 0.2
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.MIXED

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        # Expected: p(A) = 0.3*A + 0.2*I
        I = np.eye(2, dtype=complex)
        expected = 0.3 * A_matrix + 0.2 * I

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_quadratic_polynomial(self):
        """Test Poly with even quadratic polynomial."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.1, 0, 0.3])  # p(x) = 0.3x^2 + 0.1
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.EVEN

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        # Expected: p(A) = 0.3*A^2 + 0.1*I
        I = np.eye(2, dtype=complex)
        A_squared = A_matrix @ A_matrix
        expected = 0.3 * A_squared + 0.1 * I

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_quadratic_mixed_polynomial(self):
        """Test Poly with mixed parity quadratic polynomial."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([0.1, 0.2, 0.3])  # p(x) = 0.3x^2 + 0.2x + 0.1
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.MIXED

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        # Expected: p(A) = 0.3*A^2 + 0.1*I
        I = np.eye(2, dtype=complex)
        A_squared = A_matrix @ A_matrix
        expected = 0.3 * A_squared + 0.2 * A_matrix + 0.1 * I

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_with_ancillas(self):
        """Test Poly where sub-expression has ancillas."""
        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)
        sum_expr = Sum([(0.25, A), (0.75, B)])

        p = Polynomial([0.0, 0.0, 1.0])  # p(x) = x^2
        poly = Poly(sum_expr, p)

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.4, 0.8], [0.8, -0.4]], dtype=complex)
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_
        U_B = qsp_unitary_dilation(B_matrix)
        assert np.allclose(U_B @ U_B.conj().T, np.eye(4))

        sim = CircuitSimulator(
            circ,
            gate_matrices={
                "A": U_A,
                "B": U_B,
            },
        )
        actual = sim.get_block_encoding_matrix()

        S = 0.25 * A_matrix + 0.75 * B_matrix
        expected = S @ S

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_high_degree_polynomial(self):
        """Test Poly with higher degree polynomial."""
        A = Basic("A", ancilla_qubits_=1)
        # p(x) = x^5
        p = Polynomial([0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.ODD
        assert p.degree() == 5

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        # Expected: A^5
        A_power = A_matrix
        for _ in range(4):
            A_power = A_power @ A_matrix

        assert np.allclose(actual * poly.subnormalization(), A_power)

    def test_poly_negative_unitary(self):
        """Test Poly with a negative polynomial on a unitary."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([-1.0])
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.EVEN
        circ = poly.circuit()

        U_A = np.array([[0.6, 0.8], [0.8, -0.6]], dtype=complex)
        assert np.linalg.norm(U_A, ord=2) <= A.subnormalization_
        assert np.allclose(U_A @ U_A.conj().T, np.eye(2))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()
        expected = -np.eye(2, dtype=complex)

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_negative_polynomial(self):
        """Test Poly with a negative polynomial."""
        A = Basic("A", ancilla_qubits_=1)
        p = Polynomial([-1.0])
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.EVEN
        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()
        expected = -np.eye(2, dtype=complex)

        assert np.allclose(actual * poly.subnormalization(), expected)

    def test_poly_chebyshev_like_polynomial(self):
        """Test Poly with a Chebyshev-like polynomial."""
        A = Basic("A", ancilla_qubits_=1)
        # Approximate T_2(x) = 2x^2 - 1 (second Chebyshev polynomial)
        p = Polynomial([-1.0, 0.0, 2.0])
        poly = Poly(A, p)

        assert p.parity() == Polynomial.Parity.EVEN

        circ = poly.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        sim = CircuitSimulator(circ, gate_matrices={"A": U_A})
        actual = sim.get_block_encoding_matrix()

        I = np.eye(2, dtype=complex)
        A_squared = A_matrix @ A_matrix
        expected = 2.0 * A_squared - I

        assert np.allclose(actual * poly.subnormalization(), expected)
