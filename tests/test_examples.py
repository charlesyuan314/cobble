from dataclasses import replace
import math

from cobble.circuit import (
    BlackBox,
    Circuit,
    Control,
    Controlled,
    FlatGate,
    Hadamard,
    Ry,
    Z,
)
from cobble.expr import Basic, Const, Dagger, Expr, Poly, Prod, Sum, Tensor
from cobble.polynomial import Polynomial
from cobble.qtype import BitType
from cobble.simulator import CircuitSimulator
import numpy as np
from numpy.polynomial.chebyshev import cheb2poly
import pytest
from scipy.linalg import cosm, sinm
from scipy.special import jv

from utils import matrix_polyval, qsp_unitary_dilation


@pytest.fixture
def set_no_solver(monkeypatch):
    monkeypatch.setenv("NOSOLVER", "1")
    yield


class TestLCUExample:
    """Test concrete LCU example: H = A + 0.3*B where A = X⊗X + Y⊗Y, B = X⊗X - Y⊗Y.

    This tests the UNNORMALIZED nested Sum structure to match the exact circuit diagram.
    """

    def test_hamiltonian_sum_circuit(self):
        """Test exact 1-to-1 circuit structure for H = A + 0.3*B.

        Qubit allocation (4 qubits total):
        - Qubits 0-1: data (for 2-qubit operators)
        - Qubit 2: H ancilla (top-level LCU for A vs 0.3*B)
        - Qubit 3: A/B ancilla (shared between A and B, used for LCU within each)
        """

        X = Basic("X")
        Y = Basic("Y")

        A = Sum([(1.0, Tensor([X, X])), (1.0, Tensor([Y, Y]))])
        B = Sum([(1.0, Tensor([X, X])), (-1.0, Tensor([Y, Y]))])
        H = Sum([(1.0, A), (0.3, B)])

        theta_H = 2 * math.acos(math.sqrt(1.0 / 1.3))

        circ = H.circuit()

        assert circ.data_qubits == 2, "Should have 2 data qubits"
        assert circ.ancilla_qubits == 2, "Should have 2 ancillas"

        print(circ.to_ascii())

        actual_flat_gates = circ.to_list()

        expected_gates = [
            Ry(target=3, angle=theta_H),
            Hadamard(target=2),  # No control unlike in paper
            Controlled(
                BlackBox(name="X", data_qubits=1, start_qubit=0),
                controls=(Control.neg(3), Control.neg(2)),
            ),  # X⊗X, qubit 0
            Controlled(
                BlackBox(name="X", data_qubits=1, start_qubit=1),
                controls=(Control.neg(3), Control.neg(2)),
            ),  # X⊗X, qubit 1
            Controlled(
                BlackBox(name="Y", data_qubits=1, start_qubit=0),
                controls=(Control.neg(3), 2),
            ),  # Y⊗Y, qubit 0
            Controlled(
                BlackBox(name="Y", data_qubits=1, start_qubit=1),
                controls=(Control.neg(3), 2),
            ),  # Y⊗Y, qubit 1
            Hadamard(target=2),
            Hadamard(target=2),
            Controlled(Z(target=2), controls=(3,)),
            Controlled(
                BlackBox(name="X", data_qubits=1, start_qubit=0),
                controls=(3, Control.neg(2)),
            ),
            Controlled(
                BlackBox(name="X", data_qubits=1, start_qubit=1),
                controls=(3, Control.neg(2)),
            ),
            Controlled(
                BlackBox(name="Y", data_qubits=1, start_qubit=0), controls=(3, 2)
            ),
            Controlled(
                BlackBox(name="Y", data_qubits=1, start_qubit=1), controls=(3, 2)
            ),
            Hadamard(target=2),
            Ry(target=3, angle=-theta_H),
        ]

        expected_flat_gates: list[FlatGate] = []
        for gate in expected_gates:
            expected_flat_gates.extend(gate.to_flat_gates())

        assert len(actual_flat_gates) == len(
            expected_flat_gates
        ), f"Expected {len(expected_flat_gates)} gates, got {len(actual_flat_gates)}"

        for i, (actual, expected) in enumerate(
            zip(actual_flat_gates, expected_flat_gates)
        ):
            assert replace(actual, is_conjugate_pair=False) == replace(
                expected, is_conjugate_pair=False
            ), f"Gate {i} mismatch:\n  Expected: {expected}\n  Got:      {actual}"

    def test_hamiltonian_sum_matrix(self):
        """Test that unnormalized H produces the correct matrix via simulation."""

        X = Basic("X")
        Y = Basic("Y")

        A = Sum([(1.0, Tensor([X, X])), (1.0, Tensor([Y, Y]))])
        B = Sum([(1.0, Tensor([X, X])), (-1.0, Tensor([Y, Y]))])
        H = Sum([(1.0, A), (0.3, B)])

        circ = H.circuit()

        pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.linalg.norm(pauli_X, ord=2) <= X.subnormalization_
        assert np.linalg.norm(pauli_Y, ord=2) <= Y.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"X": pauli_X, "Y": pauli_Y})
        computed = sim.get_block_encoding_matrix() * H.subnormalization()

        XX = np.kron(pauli_X, pauli_X)
        YY = np.kron(pauli_Y, pauli_Y)
        expected = 1.3 * XX + 0.7 * YY

        assert np.allclose(computed, expected)

    def test_hamiltonian_normalized_circuit(self):
        """Test exact circuit structure for normalized H = 1.3*(X⊗X) + 0.7*(Y⊗Y).

        After normalization, the nested Sum structure is flattened to a single LCU.

        Qubit allocation (3 qubits total):
        - Qubits 0-1: data (for 2-qubit operators)
        - Qubit 2: ancilla (single LCU for X⊗X vs Y⊗Y)
        """

        X = Basic("X")
        Y = Basic("Y")

        A = Sum([(1.0, Tensor([X, X])), (1.0, Tensor([Y, Y]))])
        B = Sum([(1.0, Tensor([X, X])), (-1.0, Tensor([Y, Y]))])
        H = Sum([(1.0, A), (0.3, B)])
        H_normalized = H.optimize()

        theta_H = 2 * math.acos(math.sqrt(1.3 / 2.0))

        expected_gates = [
            Ry(target=2, angle=theta_H),
            Controlled(
                BlackBox(name="X", data_qubits=1, start_qubit=0),
                controls=(Control.neg(2),),
            ),
            Controlled(
                BlackBox(name="X", data_qubits=1, start_qubit=1),
                controls=(Control.neg(2),),
            ),
            Controlled(BlackBox(name="Y", data_qubits=1, start_qubit=0), controls=(2,)),
            Controlled(BlackBox(name="Y", data_qubits=1, start_qubit=1), controls=(2,)),
            Ry(target=2, angle=-theta_H),
        ]

        expected_flat_gates: list[FlatGate] = []
        for gate in expected_gates:
            expected_flat_gates.extend(gate.to_flat_gates())

        circ = H_normalized.circuit()

        assert circ.data_qubits == 2, "Should have 2 data qubits"
        assert circ.ancilla_qubits == 1, "Should have 1 ancilla (normalized structure)"

        actual_flat_gates = circ.to_list()

        assert len(actual_flat_gates) == len(
            expected_flat_gates
        ), f"Expected {len(expected_flat_gates)} gates, got {len(actual_flat_gates)}"

        for i, (actual, expected) in enumerate(
            zip(actual_flat_gates, expected_flat_gates)
        ):
            assert replace(actual, is_conjugate_pair=False) == replace(
                expected, is_conjugate_pair=False
            ), f"Gate {i} mismatch:\n  Expected: {expected}\n  Got:      {actual}"

    def test_hamiltonian_normalized_matrix(self):
        """Test that normalized H produces the correct matrix via simulation."""

        X = Basic("X")
        Y = Basic("Y")

        A = Sum([(1.0, Tensor([X, X])), (1.0, Tensor([Y, Y]))])
        B = Sum([(1.0, Tensor([X, X])), (-1.0, Tensor([Y, Y]))])
        H = Sum([(1.0, A), (0.3, B)])
        H_normalized = H.optimize()

        circ = H_normalized.circuit()

        pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.linalg.norm(pauli_X, ord=2) <= X.subnormalization_
        assert np.linalg.norm(pauli_Y, ord=2) <= Y.subnormalization_

        sim = CircuitSimulator(circ, gate_matrices={"X": pauli_X, "Y": pauli_Y})
        computed = sim.get_block_encoding_matrix() * H_normalized.subnormalization()

        XX = np.kron(pauli_X, pauli_X)
        YY = np.kron(pauli_Y, pauli_Y)
        expected = 1.3 * XX + 0.7 * YY

        assert np.allclose(computed, expected)

    def test_hamiltonian_matrices_equal(self):
        """Test that unnormalized and normalized H produce the same matrix.

        Both versions should encode the same Hamiltonian H = 1.3*(X⊗X) + 0.7*(Y⊗Y).
        """

        X = Basic("X")
        Y = Basic("Y")

        A = Sum([(1.0, Tensor([X, X])), (1.0, Tensor([Y, Y]))])
        B = Sum([(1.0, Tensor([X, X])), (-1.0, Tensor([Y, Y]))])
        H_unnormalized = Sum([(1.0, A), (0.3, B)])
        H_normalized = H_unnormalized.optimize()

        circ_unnorm = H_unnormalized.circuit()
        circ_norm = H_normalized.circuit()

        pauli_X = np.array([[0, 1], [1, 0]], dtype=complex)
        pauli_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
        gate_matrices = {"X": pauli_X, "Y": pauli_Y}
        assert np.linalg.norm(pauli_X, ord=2) <= X.subnormalization_
        assert np.linalg.norm(pauli_Y, ord=2) <= Y.subnormalization_

        sim_unnorm = CircuitSimulator(circ_unnorm, gate_matrices=gate_matrices)
        sim_norm = CircuitSimulator(circ_norm, gate_matrices=gate_matrices)

        matrix_unnorm = (
            sim_unnorm.get_block_encoding_matrix() * H_unnormalized.subnormalization()
        )
        matrix_norm = (
            sim_norm.get_block_encoding_matrix() * H_normalized.subnormalization()
        )

        assert np.allclose(matrix_norm, matrix_unnorm)


class TestQSPExample:
    """Test QSP example."""

    def test_hamiltonian_sum_matrix(self):
        """Test that unnormalized H = ((A - B) + 0.5 (A - B)**2) ((A - B) - 0.25 (A - B)**2)
        produces the correct matrix via simulation."""

        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)

        H = ((A - B) + 0.5 * (A - B) ** 2) * ((A - B) - 0.25 * (A - B) ** 2)
        circ = H.circuit()

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.3, -0.7], [-0.7, -0.3]], dtype=complex)
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
        matrix = sim.get_block_encoding_matrix() * H.subnormalization()

        expected = (
            (A_matrix - B_matrix) + 0.5 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        ) @ (
            (A_matrix - B_matrix) - 0.25 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        )
        assert np.allclose(matrix, expected)

    def test_hamiltonian_sum_alt_matrix(self):
        """Test that unnormalized H = ((A - B) + 0.5 (A - B)**2) ((A - B) - 0.5 (A - B)**2)
        produces the correct matrix via simulation."""

        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)

        H = ((A - B) + 0.5 * (A - B) ** 2) * ((A - B) - 0.5 * (A - B) ** 2)
        circ = H.circuit()

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.3, -0.7], [-0.7, -0.3]], dtype=complex)
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
        matrix = sim.get_block_encoding_matrix() * H.subnormalization()

        expected = (
            (A_matrix - B_matrix) + 0.5 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        ) @ (
            (A_matrix - B_matrix) - 0.5 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        )
        assert np.allclose(matrix, expected)

    def test_hamiltonian_sum_alt_matrix_normalized(self):
        """Test that normalized H = ((A - B) + 0.5 (A - B)**2) ((A - B) - 0.5 (A - B)**2)
        produces the correct matrix via simulation."""

        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)

        H = ((A - B) + 0.5 * (A - B) ** 2) * ((A - B) - 0.5 * (A - B) ** 2)
        H = H.optimize()
        circ = H.circuit()

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        B_matrix = np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        assert np.linalg.norm(B_matrix, ord=2) <= B.subnormalization_

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.3, -0.7], [-0.7, -0.3]], dtype=complex)
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
        matrix = sim.get_block_encoding_matrix() * H.subnormalization()

        expected = (
            (A_matrix - B_matrix) + 0.5 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        ) @ (
            (A_matrix - B_matrix) - 0.5 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        )
        assert np.allclose(matrix, expected)

    @pytest.mark.slow
    def test_normalized(self):
        """Test that the normalized version produces the correct matrix via simulation."""

        A = Basic("A", ancilla_qubits_=1)
        B = Basic("B", ancilla_qubits_=1)

        H = ((A - B) + 0.5 * (A - B) ** 2) * ((A - B) - 0.25 * (A - B) ** 2)
        H = H.optimize()

        circ = H.circuit()

        A_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        B_matrix = np.array([[0.3, -0.7], [-0.7, -0.3]], dtype=complex)
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
        matrix = sim.get_block_encoding_matrix() * H.subnormalization()

        expected = (
            (A_matrix - B_matrix) + 0.5 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        ) @ (
            (A_matrix - B_matrix) - 0.25 * (A_matrix - B_matrix) @ (A_matrix - B_matrix)
        )
        assert np.allclose(matrix, expected)


class TestSimpleExamples:
    """Test matrix simulation for simple examples from main.py."""

    def test_simple_1_matrix(self):
        """Test (A + A) * 0.5 => A produces correct matrix."""

        A = Basic("A")

        prog = Sum([(0.5, A), (0.5, A)])
        targ = prog.optimize()

        circ_prog = prog.circuit()
        circ_targ = targ.circuit()

        A_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_

        sim_prog = CircuitSimulator(circ_prog, gate_matrices={"A": A_matrix})
        sim_targ = CircuitSimulator(circ_targ, gate_matrices={"A": A_matrix})

        matrix_prog = sim_prog.get_block_encoding_matrix() * prog.subnormalization()
        matrix_targ = sim_targ.get_block_encoding_matrix() * targ.subnormalization()

        assert np.allclose(matrix_prog, A_matrix)
        assert np.allclose(matrix_targ, A_matrix)
        assert np.allclose(matrix_prog, matrix_targ)

    def test_simple_3_matrix(self):
        """Test I - A†A produces correct matrix via simulation."""

        A = Basic(
            "A",
            ancilla_qubits_=1,
        )
        I = Const(1.0)

        prog = Sum([(1.0, I), (-1.0, Prod([Dagger(A), A]))])
        prog_normalized = prog.optimize()

        A_matrix = 0.5 * np.array([[1, 0], [0, -1]], dtype=complex)
        assert np.linalg.norm(A_matrix, ord=2) <= A.subnormalization_
        U_A = qsp_unitary_dilation(A_matrix)
        assert np.allclose(U_A @ U_A.conj().T, np.eye(4))

        circ_prog = prog.circuit()
        sim_prog = CircuitSimulator(circ_prog, gate_matrices={"A": U_A, "A_dag": U_A})
        matrix_prog = sim_prog.get_block_encoding_matrix() * prog.subnormalization()

        circ_norm = prog_normalized.circuit()
        sim_norm = CircuitSimulator(circ_norm, gate_matrices={"A": U_A})
        matrix_norm = (
            sim_norm.get_block_encoding_matrix() * prog_normalized.subnormalization()
        )

        expected = np.eye(2, dtype=complex) - A_matrix @ A_matrix

        assert np.allclose(matrix_prog, expected)
        assert np.allclose(matrix_norm, expected)
        assert np.allclose(matrix_prog, matrix_norm)


class TestPenalizedCouplerExample:
    """Test penalized coupler example with cancellations."""

    def test_penalized_coupler_matrix(self):
        """Test that canceling terms produce correct matrix."""

        Z0 = Basic("Z0", qtype_=BitType(2))
        Z1 = Basic("Z1", qtype_=BitType(2))
        ZZ = Basic("ZZ", qtype_=BitType(2))
        I = Expr.kron(Const(1.0), Const(1.0))
        H_tot = 2 * Z0 + 2 * Z1 + 1.2 * ZZ - (Z0 - Z1 + I)
        H_normalized = H_tot.optimize()

        Z = np.array([[1, 0], [0, -1]], dtype=complex)
        I = np.eye(2, dtype=complex)

        Z0_matrix = np.kron(Z, I)
        Z1_matrix = np.kron(I, Z)
        ZZ_matrix = np.kron(Z, Z)
        assert np.linalg.norm(Z0_matrix, ord=2) <= Z0.subnormalization_
        assert np.linalg.norm(Z1_matrix, ord=2) <= Z1.subnormalization_
        assert np.linalg.norm(ZZ_matrix, ord=2) <= ZZ.subnormalization_

        gate_matrices = {"Z0": Z0_matrix, "Z1": Z1_matrix, "ZZ": ZZ_matrix}

        circ_unnorm = H_tot.circuit()
        sim_unnorm = CircuitSimulator(circ_unnorm, gate_matrices=gate_matrices)
        matrix_unnorm = (
            sim_unnorm.get_block_encoding_matrix() * H_tot.subnormalization()
        )

        circ_norm = H_normalized.circuit()
        sim_norm = CircuitSimulator(circ_norm, gate_matrices=gate_matrices)
        matrix_norm = (
            sim_norm.get_block_encoding_matrix() * H_normalized.subnormalization()
        )

        expected = (
            1.2 * ZZ_matrix + Z0_matrix + 3 * Z1_matrix - np.eye(4, dtype=complex)
        )

        assert np.allclose(matrix_unnorm, expected)
        assert np.allclose(matrix_norm, expected)


class TestLaplacianFilterExample:
    """Test Laplacian filter with tensor products and polynomial cancellations."""

    @pytest.mark.slow
    def test_laplacian_filter_matrix(self):
        """Test that polynomial tensor product produces correct matrix."""

        Hx = Basic("Hx")
        Hy = Basic("Hy")

        f = [1.0, 0.5]  # 1 + 0.5x
        g = [-2.9, 3.4]
        p = [1.1, 0.4]
        q = [-5.8, -0.3]

        term_fp = Expr.kron(Poly(Hx, Polynomial(f)), Poly(Hy, Polynomial(p)))
        term_fq = Expr.kron(Poly(Hx, Polynomial(f)), Poly(Hy, Polynomial(q)))
        term_gp = Expr.kron(Poly(Hx, Polynomial(g)), Poly(Hy, Polynomial(p)))
        term_gq = Expr.kron(Poly(Hx, Polynomial(g)), Poly(Hy, Polynomial(q)))
        prog = term_fp + term_gp + term_fq + term_gq
        prog_normalized = prog.optimize()

        X_matrix = np.array([[0, 1], [1, 0]], dtype=complex)
        Y_matrix = np.array([[0, -1j], [1j, 0]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= Hx.subnormalization_
        assert np.linalg.norm(Y_matrix, ord=2) <= Hy.subnormalization_

        gate_matrices = {"Hx": X_matrix, "Hy": Y_matrix}

        circ_unnorm = prog.circuit()
        sim_unnorm = CircuitSimulator(circ_unnorm, gate_matrices=gate_matrices)
        matrix_unnorm = sim_unnorm.get_block_encoding_matrix() * prog.subnormalization()

        circ_norm = prog_normalized.circuit()
        sim_norm = CircuitSimulator(circ_norm, gate_matrices=gate_matrices)
        matrix_norm = (
            sim_norm.get_block_encoding_matrix() * prog_normalized.subnormalization()
        )

        fx_plus_gx = (-1.9) * np.eye(2, dtype=complex) + 3.9 * X_matrix
        py_plus_qy = (-4.7) * np.eye(2, dtype=complex) + 0.1 * Y_matrix
        expected = np.kron(fx_plus_gx, py_plus_qy)

        assert np.allclose(matrix_unnorm, expected)
        assert np.allclose(matrix_norm, expected)


class TestOLSRidgeExample:
    """Test OLS-ridge example with Neumann series."""

    @pytest.mark.slow
    def test_ols_ridge_matrix(self):
        """Test OLS-ridge Neumann series produces correct matrix."""

        d = 2
        theta = 0.5
        lam = 0.5
        mu = 2.0

        X = Basic("X", ancilla_qubits_=1)
        I = Const(1.0)

        H = Dagger(X) * X
        prog = theta * X * 1 / mu * Sum.of(
            (I - H / mu) ** k for k in range(d + 1)
        ) * Dagger(X) + (1 - theta) * X * 1 / lam * Sum.of(
            (-H / lam) ** k for k in range(d + 1)
        ) * Dagger(
            X
        )
        prog_normalized = prog.optimize()

        X_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_
        U_X = qsp_unitary_dilation(X_matrix)
        assert np.allclose(U_X @ U_X.conj().T, np.eye(4))

        circ_unnorm = prog.circuit()
        sim_unnorm = CircuitSimulator(
            circ_unnorm, gate_matrices={"X": U_X, "X_dag": U_X}
        )
        matrix_unnorm = sim_unnorm.get_block_encoding_matrix() * prog.subnormalization()

        circ_norm = prog_normalized.circuit()
        sim_norm = CircuitSimulator(circ_norm, gate_matrices={"X": U_X})
        matrix_norm = (
            sim_norm.get_block_encoding_matrix() * prog_normalized.subnormalization()
        )

        H_matrix = X_matrix @ X_matrix
        I_matrix = np.eye(2, dtype=complex)

        base_L_matrix = I_matrix - H_matrix / mu
        expected = np.zeros((2, 2), dtype=complex)

        for k in range(d + 1):
            if k == 0:
                pow_L_matrix = I_matrix
            else:
                pow_L_matrix = np.linalg.matrix_power(base_L_matrix, k)
            expected += (theta / mu) * X_matrix @ pow_L_matrix @ X_matrix

        base_R_matrix = -H_matrix / lam

        for k in range(d + 1):
            if k == 0:
                pow_R_matrix = I_matrix
            else:
                pow_R_matrix = np.linalg.matrix_power(base_R_matrix, k)
            expected += ((1.0 - theta) / lam) * X_matrix @ pow_R_matrix @ X_matrix

        assert np.allclose(matrix_unnorm, expected)
        assert np.allclose(matrix_norm, expected)


class TestAlgorithmicExamples:
    """Test examples from quantum algorithms."""

    def optimal_matrix_inversion_polynomial(self, n, a):
        """
        Optimal polynomial to approximate X^(-1) from Sünderhauf et al. 2025.

        L_n(x) := 1/(2^n - 1)*(T_n(x) + (1-a)/(1+a)*T_{n-1}(x))
        P_{2n-1}(x) := 1/x - L_n((2x^2-(1+a^2))/(1-a^2)) / (x * L_n(-(1+a^2) / (1-a^2)))
        """

        def P_coeffs(n, a):
            Tn = np.polynomial.Chebyshev.basis(n)
            Tn1 = np.polynomial.Chebyshev.basis(n - 1)
            Ln = (Tn + ((1 - a) / (1 + a)) * Tn1) * (1 / 2 ** (n - 1))
            Ln_poly = Ln.convert(kind=np.polynomial.Polynomial)
            u = np.polynomial.Polynomial([-(1 + a**2) / (1 - a**2), 0, 2 / (1 - a**2)])
            Ln_u = Ln_poly(u)
            u0 = -(1 + a**2) / (1 - a**2)
            Ln_u0 = Ln_poly(u0)
            numerator = Ln_u0 - Ln_u  # even polynomial, divisible by x^2
            num_coef = numerator.coef / Ln_u0  # divide by scalar Ln_u0
            return num_coef[1:]  # drop constant term (since divisible by x)

        def T_n(n, X):
            return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0])))

        def L_n(n, X):
            return (T_n(n, X) + ((1 - a) / (1 + a)) * T_n(n - 1, X)) / Const(2**n - 1)

        X = Basic("X", ancilla_qubits_=1)
        P = Const(1.0) / X - L_n(
            n, (2 * X**2 - Const(1 + a**2)) / Const(1 - a**2)
        ) / (X * L_n(n, Const(-((1 + a**2)) / (1 - a**2))))

        return X, P, P_coeffs(n, a)

    def test_optimal_matrix_inversion_polynomial(self, set_no_solver):
        n = 7
        a = 1.0 / 6.0

        _, P, P_coeffs = self.optimal_matrix_inversion_polynomial(n, a)

        P_simplified = P.optimize()
        assert isinstance(P_simplified, Poly)
        assert P_simplified.p.parity() == Polynomial.Parity.ODD
        assert P_simplified.p.degree() == 2 * n - 1
        assert np.allclose(P_simplified.p.coeffs, P_coeffs)

        assert isinstance(P_simplified.circuit(), Circuit)

    @pytest.mark.slow
    def test_optimal_matrix_inversion_polynomial_matrix(self):
        n = 7
        a = 1.0 / 6.0

        X, P, P_coeffs = self.optimal_matrix_inversion_polynomial(n, a)
        P_simplified = P.optimize()

        X_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_
        U_X = qsp_unitary_dilation(X_matrix)
        assert np.allclose(U_X @ U_X.conj().T, np.eye(4))

        circ = P_simplified.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"X": U_X})
        matrix = sim.get_block_encoding_matrix() * P_simplified.subnormalization()
        assert np.allclose(matrix, matrix_polyval(P_coeffs, X_matrix))

    def sign_polynomial(self, n):
        """
        Polynomial to approximate the sign(X) function.

        sign(x) = 4 / pi * sum_{k=0}^{n} (1 / (2k + 1)) * T_{2k+1}(x)
        """

        def sign_coeffs(n):
            cheb = cheb2poly([0 if i % 2 == 0 else 1.0 / i for i in range(2 * n + 1)])
            return 4 / math.pi * cheb

        def T_n(n, X):
            return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0])))

        X = Basic("X", ancilla_qubits_=1)
        poly = Const(4 / math.pi) * Sum(
            [(1.0 / (2 * i + 1), T_n(2 * i + 1, X)) for i in range(n)]
        )
        return (
            X,
            poly,
            sign_coeffs(n),
        )

    def test_sign_function(self, set_no_solver):
        """Test the polynomial to approximate the sign(X) function.

        sign(x) = 4 / pi * sum_{k=0}^{n} (1 / (2k + 1)) * T_{2k+1}(x)
        """

        n = 9
        _, sign, sign_coeffs = self.sign_polynomial(n)
        sign_simplified = sign.optimize()
        assert isinstance(sign_simplified, Poly)
        assert sign_simplified.p.parity() == Polynomial.Parity.ODD
        assert sign_simplified.p.degree() == 2 * n - 1
        assert np.allclose(sign_simplified.p.coeffs, sign_coeffs)

        assert isinstance(sign_simplified.circuit(), Circuit)

    @pytest.mark.slow
    def test_sign_function_matrix(self):
        n = 9  # start getting numerical accuracy errors beyond this
        X, sign, sign_coeffs = self.sign_polynomial(n)
        sign_simplified = sign.optimize()

        X_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_
        U_X = qsp_unitary_dilation(X_matrix)
        assert np.allclose(U_X @ U_X.conj().T, np.eye(4))

        circ = sign_simplified.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"X": U_X})
        matrix = sim.get_block_encoding_matrix() * sign_simplified.subnormalization()
        assert np.allclose(matrix, matrix_polyval(sign_coeffs, X_matrix))

    def cosine_polynomial(self, n, t):
        """
        Polynomial to approximate the cos(tX) function.

        cos(x) = J_0(t) + 2 * sum_{k=1}^{n} (-1)^k * J_{2k}(t) * T_{2k}(x)
        """

        def jacobi_anger_cheb_coeffs(t, n):
            max_degree = 2 * n
            c = np.zeros(max_degree + 1, dtype=float)
            for k in range(1, n + 1):
                m = 2 * k
                c[m] = 2.0 * ((-1.0) ** k) * jv(m, t)
            c[0] += jv(0, t)
            return cheb2poly(c)

        def T_n(n, X):
            return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0])))

        X = Basic("X", ancilla_qubits_=1)
        poly = Const(jv(0, t)) + 2 * Sum(
            [((-1) ** k * jv(2 * k, t), T_n(2 * k, X)) for k in range(1, n + 1)]
        )
        return (
            X,
            poly,
            jacobi_anger_cheb_coeffs(t, n),
        )

    def test_cosine_function(self, set_no_solver):
        n = 10
        t = 1.0

        _, cos, cos_coeffs = self.cosine_polynomial(n, t)

        cos_simplified = cos.optimize()
        assert isinstance(cos_simplified, Poly)
        assert cos_simplified.p.parity() == Polynomial.Parity.EVEN
        assert cos_simplified.p.degree() == 2 * n
        assert np.allclose(cos_simplified.p.coeffs, cos_coeffs)

        assert isinstance(cos_simplified.circuit(), Circuit)

    @pytest.mark.slow
    def test_cosine_function_matrix(self):
        n = 10
        t = 1.0
        X, cos, cos_coeffs = self.cosine_polynomial(n, t)
        cos_simplified = cos.optimize()

        X_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_
        U_X = qsp_unitary_dilation(X_matrix)
        assert np.allclose(U_X @ U_X.conj().T, np.eye(4))

        circ = cos_simplified.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"X": U_X})
        matrix = sim.get_block_encoding_matrix() * cos_simplified.subnormalization()

        assert np.allclose(matrix, matrix_polyval(cos_coeffs, X_matrix))
        assert np.allclose(matrix, cosm(t * X_matrix))

    def sine_polynomial(self, n, t):
        """
        Polynomial to approximate the sin(tX) function.

        sin(x) = 2 * sum_{k=0}^{n} (-1)^k * J_{2k+1}(t) * T_{2k+1}(x)
        """

        def jacobi_anger_cheb_coeffs(t, n):
            max_degree = 2 * n + 1
            c = np.zeros(max_degree + 1, dtype=float)
            for k in range(n + 1):
                m = 2 * k + 1
                c[m] = 2.0 * ((-1.0) ** k) * jv(m, t)
            return cheb2poly(c)

        def T_n(n, X):
            return Poly(X, Polynomial(cheb2poly([0.0] * n + [1.0])))

        X = Basic("X", ancilla_qubits_=1)
        poly = 2 * Sum(
            [((-1) ** k * jv(2 * k + 1, t), T_n(2 * k + 1, X)) for k in range(n + 1)]
        )
        return (
            X,
            poly,
            jacobi_anger_cheb_coeffs(t, n),
        )

    def test_sine_function(self, set_no_solver):
        n = 10
        t = 1.0
        _, sin, sin_coeffs = self.sine_polynomial(n, t)

        sin_simplified = sin.optimize()
        assert isinstance(sin_simplified, Poly)
        assert sin_simplified.p.parity() == Polynomial.Parity.ODD
        assert sin_simplified.p.degree() == 2 * n + 1
        assert np.allclose(sin_simplified.p.coeffs, sin_coeffs)

        assert isinstance(sin_simplified.circuit(), Circuit)

    @pytest.mark.slow
    def test_sine_function_matrix(self):
        n = 10
        t = 1.0
        X, sin, sin_coeffs = self.sine_polynomial(n, t)
        sin_simplified = sin.optimize()

        X_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_
        U_X = qsp_unitary_dilation(X_matrix)
        assert np.allclose(U_X @ U_X.conj().T, np.eye(4))

        circ = sin_simplified.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"X": U_X})
        matrix = sim.get_block_encoding_matrix() * sin_simplified.subnormalization()

        assert np.allclose(matrix, matrix_polyval(sin_coeffs, X_matrix))
        assert np.allclose(matrix, sinm(t * X_matrix))

    def test_hamiltonian_simulation(self, set_no_solver):
        """Test the polynomial approximation for e^(iHt) = cos(Ht) + i sin(Ht)."""
        n = 10
        t = 1.0
        _, cos, _ = self.cosine_polynomial(n, t)
        _, sin, _ = self.sine_polynomial(n, t)
        P = cos + Const(1j) * sin
        P_simplified = P.optimize()

        assert isinstance(P_simplified, Sum)
        assert isinstance(P_simplified.circuit(), Circuit)

    @pytest.mark.slow
    def test_hamiltonian_simulation_matrix(self):
        n = 10
        t = 1.0

        X, cos, _ = self.cosine_polynomial(n, t)
        _, sin, _ = self.sine_polynomial(n, t)

        P = cos + Const(1j) * sin
        P_simplified = P.optimize()

        X_matrix = np.array([[0.5, 0.8], [0.8, -0.5]], dtype=complex)
        assert np.linalg.norm(X_matrix, ord=2) <= X.subnormalization_
        U_X = qsp_unitary_dilation(X_matrix)
        assert np.allclose(U_X @ U_X.conj().T, np.eye(4))

        circ = P_simplified.circuit()
        sim = CircuitSimulator(circ, gate_matrices={"X": U_X})
        matrix = sim.get_block_encoding_matrix() * P_simplified.subnormalization()

        assert np.allclose(matrix, cosm(t * X_matrix) + 1j * sinm(t * X_matrix))

    @pytest.mark.slow
    def test_chebyshev_polynomials(self, set_no_solver):
        """Test Chebyshev polynomials up to degree 100."""

        def cheb_coeffs(n):
            return cheb2poly([0.0] * n + [1.0])

        def T_n(n, X):
            return Sum([(coeff, X**i) for i, coeff in enumerate(cheb_coeffs(n))])

        X = Basic("X")
        for n in range(2, 100):
            T_n_simplified = T_n(n, X).optimize()
            circuit = T_n_simplified.circuit()
            assert circuit.data_qubits == 1
            assert circuit.ancilla_qubits == 1
            assert isinstance(T_n_simplified, Poly)
            assert (
                T_n_simplified.p.parity() == Polynomial.Parity.EVEN
                if n % 2 == 0
                else Polynomial.Parity.ODD
            )
            assert T_n_simplified.p.degree() == n
            assert np.allclose(T_n_simplified.p.coeffs, cheb_coeffs(n))
