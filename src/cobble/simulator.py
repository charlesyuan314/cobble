from __future__ import annotations

import numpy as np
import quimb.tensor as qtn

from cobble.circuit import Circuit, FlatGate, Gate, BlackBox, Controlled

# Standard quantum gates as matrices
GATE_MATRICES = {
    "NOT": np.array([[0, 1], [1, 0]], dtype=complex),
    "H": np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
    "I": np.eye(2, dtype=complex),
}


def _rx_matrix(theta: float) -> np.ndarray:
    """Rotation around X axis."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -1j * s], [-1j * s, c]], dtype=complex)


def _ry_matrix(theta: float) -> np.ndarray:
    """Rotation around Y axis."""
    c = np.cos(theta / 2)
    s = np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _rz_matrix(theta: float) -> np.ndarray:
    """Rotation around Z axis."""
    return np.array(
        [[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], dtype=complex
    )


def _swap_matrix() -> np.ndarray:
    """SWAP gate matrix."""
    swap = np.zeros((4, 4), dtype=complex)
    swap[0, 0] = 1
    swap[1, 2] = 1
    swap[2, 1] = 1
    swap[3, 3] = 1
    return swap


def _su2_rotation_matrix(theta: float, phi: float, lbd: float) -> np.ndarray:
    """SU(2) rotation gate matrix."""
    return np.array(
        [
            [
                np.exp(1j * (lbd + phi)) * np.cos(theta),
                np.exp(1j * phi) * np.sin(theta),
            ],
            [np.exp(1j * lbd) * np.sin(theta), -np.cos(theta)],
        ],
        dtype=complex,
    )


def _global_phase_matrix(phase: float) -> np.ndarray:
    """Global phase gate matrix."""
    return np.array([[np.exp(1j * phase), 0], [0, np.exp(1j * phase)]], dtype=complex)


def _get_gate_matrix(gate: FlatGate) -> np.ndarray:
    """Get the matrix representation of a flat gate."""
    if gate.name in GATE_MATRICES:
        return GATE_MATRICES[gate.name]
    elif gate.name == "Rx":
        return _rx_matrix(gate.params[0])
    elif gate.name == "Ry":
        return _ry_matrix(gate.params[0])
    elif gate.name == "Rz":
        return _rz_matrix(gate.params[0])
    elif gate.name == "SWAP":
        return _swap_matrix()
    elif gate.name == "SU2Rotation":
        return _su2_rotation_matrix(gate.params[0], gate.params[1], gate.params[2])
    elif gate.name == "GlobalPhase":
        return _global_phase_matrix(gate.params[0])
    else:
        # For custom gates (like BlackBox), return identity as placeholder
        num_qubits = len(gate.targets)
        dim = 2**num_qubits
        return np.eye(dim, dtype=complex)


def _build_controlled_gate_with_ordering(
    base_matrix: np.ndarray,
    control_qubits: tuple[int, ...],
    target_qubits: tuple[int, ...],
    all_qubits_sorted: tuple[int, ...],
) -> np.ndarray:
    """Build a controlled gate matrix respecting global qubit ordering.

    Args:
        base_matrix: The base gate matrix for target qubits (2^k x 2^k for k targets)
        control_qubits: Indices of control qubits
        target_qubits: Indices of target qubits
        all_qubits_sorted: All involved qubits in sorted order

    Returns:
        Full gate matrix in the computational basis ordered by all_qubits_sorted
    """
    n_qubits = len(all_qubits_sorted)
    dim = 2**n_qubits

    # Create mapping from global qubit index to position in all_qubits_sorted
    qubit_to_pos = {q: i for i, q in enumerate(all_qubits_sorted)}

    # Positions of control and target qubits in the sorted list
    control_positions = [qubit_to_pos[q] for q in control_qubits]
    target_positions = [qubit_to_pos[q] for q in target_qubits]

    result = np.zeros((dim, dim), dtype=complex)

    # Iterate over all input basis states (columns)
    for input_idx in range(dim):
        # Extract bits for each qubit in input state (MSB first in all_qubits_sorted)
        input_bits = [
            (input_idx >> (n_qubits - 1 - pos)) & 1 for pos in range(n_qubits)
        ]

        # Check if all control qubits are 1
        controls_active = all(input_bits[pos] == 1 for pos in control_positions)

        if controls_active:
            # Apply the base gate to target qubits
            # Extract input target qubit values
            target_in_bits = tuple(input_bits[pos] for pos in target_positions)
            target_in_idx = sum(
                b * (2 ** (len(target_positions) - 1 - k))
                for k, b in enumerate(target_in_bits)
            )

            # For each possible output target configuration
            for target_out_idx in range(2 ** len(target_positions)):
                # Get the amplitude from the base matrix
                amplitude = base_matrix[target_out_idx, target_in_idx]

                if abs(amplitude) < 1e-15:
                    continue

                # Build the output state by keeping control qubits the same
                # and setting target qubits according to target_out_idx
                output_bits = list(input_bits)
                target_out_bits = [
                    (target_out_idx >> (len(target_positions) - 1 - k)) & 1
                    for k in range(len(target_positions))
                ]

                for k, pos in enumerate(target_positions):
                    output_bits[pos] = target_out_bits[k]

                # Convert output_bits back to index
                output_idx = sum(
                    output_bits[pos] * (2 ** (n_qubits - 1 - pos))
                    for pos in range(n_qubits)
                )

                # Set matrix element: U[output_idx, input_idx]
                result[output_idx, input_idx] = amplitude
        else:
            # Identity: no gate applied, state unchanged
            result[input_idx, input_idx] = 1.0

    return result


def _build_single_qubit_gate(
    gate_matrix: np.ndarray, qubit: int, depth: int
) -> qtn.Tensor:
    """Build a tensor for a single-qubit gate.

    Args:
        gate_matrix: 2x2 gate matrix
        qubit: Qubit index
        depth: Depth in the circuit (for unique labeling)

    Returns:
        Tensor with indices [output, input]
    """
    return qtn.Tensor(
        data=gate_matrix.reshape(2, 2),  # type: ignore
        inds=[f"q{qubit}_{depth+1}", f"q{qubit}_{depth}"],
        tags={f"GATE_Q{qubit}_D{depth}"},
    )


def _build_multi_qubit_gate(
    gate_matrix: np.ndarray, qubits: tuple[int, ...], depth: int
) -> qtn.Tensor:
    """Build a tensor for a multi-qubit gate.

    Args:
        gate_matrix: Gate matrix (2^n x 2^n for n qubits)
        qubits: Tuple of qubit indices
        depth: Depth in the circuit

    Returns:
        Tensor with indices ordered by qubits
    """
    n_qubits = len(qubits)
    # Reshape to have separate indices for each qubit: out0, out1, ..., in0, in1, ...
    matrix_reshaped = gate_matrix.reshape([2] * (2 * n_qubits))

    # Create index names
    inds_out = [f"q{q}_{depth+1}" for q in qubits]
    inds_in = [f"q{q}_{depth}" for q in qubits]

    return qtn.Tensor(
        data=matrix_reshaped,  # type: ignore
        inds=inds_out + inds_in,
        tags={f"GATE_{'_'.join(map(str, qubits))}_D{depth}"},
    )


def _convert_ancilla_data_to_data_ancilla(
    matrix: np.ndarray, data_qubits: int, ancilla_qubits: int
) -> np.ndarray:
    """Convert a matrix from ancilla ⊗ data ordering to data ⊗ ancilla ordering.

    The input matrix has the data block in the upper left (ancilla ⊗ data ordering).
    This function converts it to data ⊗ ancilla ordering as expected by the simulator.

    Args:
        matrix: Matrix in ancilla ⊗ data ordering
        data_qubits: Number of data qubits
        ancilla_qubits: Number of ancilla qubits

    Returns:
        Matrix in data ⊗ ancilla ordering
    """
    dim_data = 2**data_qubits
    dim_ancilla = 2**ancilla_qubits

    # Reshape to separate ancilla and data indices
    # [out_anc, out_data, in_anc, in_data]
    reshaped = matrix.reshape(dim_ancilla, dim_data, dim_ancilla, dim_data)

    # Transpose to [out_data, out_anc, in_data, in_anc]
    transposed = reshaped.transpose(1, 0, 3, 2)

    # Reshape back to matrix form [data ⊗ ancilla]
    return transposed.reshape(dim_data * dim_ancilla, dim_data * dim_ancilla)


class CircuitSimulator:
    """Simulator for quantum circuits using tensor networks.

    This class provides methods to simulate quantum circuits with separate
    tracking of data and ancilla qubits, supporting postselection on ancillas.
    """

    def __init__(
        self,
        circuit: Circuit,
        gate_matrices: dict[str, np.ndarray] | None = None,
        ancilla_data_ordering: bool = True,
    ):
        """Initialize simulator with a circuit.

        Args:
            circuit: The quantum circuit to simulate
            gate_matrices: Optional dict mapping gate names to their matrix representations.
                          Use this to provide explicit matrices for BlackBox gates.
                          Keys are gate names (e.g., "U", "Oracle"), values are unitary matrices.
            ancilla_data_ordering: If True, provided matrices are assumed to be in ancilla ⊗ data
                                  ordering (data block in upper left) and will be converted to
                                  data ⊗ ancilla ordering. If False, matrices are assumed to
                                  already be in data ⊗ ancilla ordering. Default: True.

        Raises:
            ValueError: If circuit contains BlackBox gates without provided matrices
        """
        self.circuit = circuit
        self.data_qubits = circuit.data_qubits
        self.ancilla_qubits = circuit.ancilla_qubits
        self.total_qubits = circuit.allocation.total_qubits()
        self.flat_gates = circuit.to_list()
        self.gate_matrices = gate_matrices or {}
        self.ancilla_data_ordering = ancilla_data_ordering

        # Validate that all BlackBox gates have explicit matrices
        self._validate_blackbox_gates()

    def _validate_blackbox_gates(self) -> None:
        """Validate that all BlackBox gates have explicit matrices provided.

        Also converts matrices from ancilla ⊗ data ordering to data ⊗ ancilla ordering if requested.

        Raises:
            ValueError: If any BlackBox gate lacks a matrix in gate_matrices
        """

        # Track which gate names have been converted to avoid double conversion
        converted_gates = set()

        # Check all gates in the original circuit (not just flat gates)
        def check_gate(gate: Gate) -> None:
            if isinstance(gate, BlackBox):
                if gate.name not in self.gate_matrices:
                    raise ValueError(
                        f"BlackBox gate '{gate.name}' requires an explicit matrix. "
                        f"Provide it via gate_matrices parameter. "
                        f"Cannot simulate uninstantiated Base expressions."
                    )
                expected_dim = 2 ** (gate.data_qubits + gate.ancilla_qubits)
                if self.gate_matrices[gate.name].shape != (expected_dim, expected_dim):
                    raise ValueError(
                        f"BlackBox gate '{gate.name}' has matrix of shape {self.gate_matrices[gate.name].shape}, "
                        f"expected {expected_dim}x{expected_dim}"
                    )

                # Convert from ancilla ⊗ data to data ⊗ ancilla ordering if requested
                # and if the gate has both data and ancilla qubits (but only once per gate name)
                if (
                    self.ancilla_data_ordering
                    and gate.data_qubits > 0
                    and gate.ancilla_qubits > 0
                    and gate.name not in converted_gates
                ):
                    self.gate_matrices[gate.name] = (
                        _convert_ancilla_data_to_data_ancilla(
                            self.gate_matrices[gate.name],
                            gate.data_qubits,
                            gate.ancilla_qubits,
                        )
                    )
                    converted_gates.add(gate.name)
            elif isinstance(gate, Controlled):
                check_gate(gate.gate)

        for gate in self.circuit.gates:
            check_gate(gate)

    def to_tensor_network(self) -> qtn.TensorNetwork:
        """Convert the circuit to a tensor network.

        Returns:
            Tensor network representing the circuit unitary.
            The network has open indices for each qubit at depth 0 and final_depth.
        """
        tensors = []

        # Track the current depth for each qubit (starts at 0)
        qubit_depths = [0] * self.total_qubits

        # Process each gate
        for gate in self.flat_gates:
            # Get base gate matrix
            # Check if there is a custom matrix for this gate
            if gate.name in self.gate_matrices:
                base_matrix = self.gate_matrices[gate.name]
            else:
                base_matrix = _get_gate_matrix(gate)

            # Determine which qubits are affected and build the full gate matrix
            if gate.controls:
                # For controlled gates, need to build the matrix respecting qubit ordering
                all_qubits = tuple(sorted(gate.controls + gate.targets))
                # Build controlled gate in the computational basis ordered by qubit index
                full_matrix = _build_controlled_gate_with_ordering(
                    base_matrix, gate.controls, gate.targets, all_qubits
                )
                affected_qubits = all_qubits
            else:
                affected_qubits = tuple(sorted(gate.targets))
                full_matrix = base_matrix

            # Find the maximum depth among affected qubits
            max_depth = max(qubit_depths[q] for q in affected_qubits)

            # Add identity gates to bring all affected qubits to the same depth
            for q in affected_qubits:
                while qubit_depths[q] < max_depth:
                    tensors.append(
                        _build_single_qubit_gate(GATE_MATRICES["I"], q, qubit_depths[q])
                    )
                    qubit_depths[q] += 1

            # Add the gate
            if len(affected_qubits) == 1:
                tensors.append(
                    _build_single_qubit_gate(full_matrix, affected_qubits[0], max_depth)
                )
            else:
                tensors.append(
                    _build_multi_qubit_gate(full_matrix, affected_qubits, max_depth)
                )

            # Update depths for affected qubits
            for q in affected_qubits:
                qubit_depths[q] = max_depth + 1

        # Find final depth
        final_depth = max(qubit_depths) if qubit_depths else 0

        # Ensure at least depth 1 for proper input/output separation
        if final_depth == 0:
            final_depth = 1

        # Add identity gates to bring all qubits to final depth
        for q in range(self.total_qubits):
            while qubit_depths[q] < final_depth:
                tensors.append(
                    _build_single_qubit_gate(GATE_MATRICES["I"], q, qubit_depths[q])
                )
                qubit_depths[q] += 1

        # Create tensor network
        tn = qtn.TensorNetwork(tensors)

        # Relabel external indices for consistency
        for q in range(self.total_qubits):
            tn.reindex_({f"q{q}_0": f"q{q}_in"})
            tn.reindex_({f"q{q}_{final_depth}": f"q{q}_out"})

        return tn

    def simulate(
        self, initial_data_vector: np.ndarray, return_full: bool = False
    ) -> np.ndarray:
        """Simulate the circuit on an initial data vector with postselection.

        The ancilla qubits are initialized to |0⟩ and postselected to |0⟩.

        Args:
            initial_data_vector: Initial state vector for data qubits.
                                 Shape: (2^data_qubits,)
            return_full: If True, return full state before postselection.
                        If False, return data state after postselection and renormalization.

        Returns:
            Final state vector after applying circuit and postselection.
            If return_full=False: shape (2^data_qubits,) - normalized data state
            If return_full=True: shape (2^total_qubits,) - full state before postselection
        """
        # Validate input
        expected_data_dim = 2**self.data_qubits
        if initial_data_vector.shape[0] != expected_data_dim:
            raise ValueError(
                f"Initial vector has dimension {initial_data_vector.shape[0]}, "
                f"expected {expected_data_dim} for {self.data_qubits} data qubits"
            )

        # Build full initial state: data ⊗ |0...0⟩_ancilla
        ancilla_zero_state = np.zeros(2**self.ancilla_qubits, dtype=complex)
        ancilla_zero_state[0] = 1.0
        full_initial_state = np.kron(initial_data_vector, ancilla_zero_state)

        # Convert circuit to tensor network
        tn = self.to_tensor_network()

        # Create input state tensor
        initial_tensor = qtn.Tensor(
            data=full_initial_state.reshape([2] * self.total_qubits),  # type: ignore
            inds=[f"q{q}_in" for q in range(self.total_qubits)],
            tags={"INITIAL_STATE"},
        )

        # Add to network
        tn.add_tensor(initial_tensor)

        # Contract to get output state
        output_inds = [f"q{q}_out" for q in range(self.total_qubits)]
        final_tensor = tn.contract(output_inds=output_inds)
        final_state = final_tensor.data.flatten()  # type: ignore

        if return_full:
            return final_state

        # Postselect ancillas to |0⟩
        # Reshape to separate data and ancilla
        final_state_reshaped = final_state.reshape(
            (2**self.data_qubits, 2**self.ancilla_qubits)
        )

        # Extract amplitude for ancilla = |0...0⟩ (index 0)
        postselected_state = final_state_reshaped[:, 0]

        # Normalize
        norm = np.linalg.norm(postselected_state)
        if norm < 1e-15:
            raise ValueError(
                "Postselection resulted in zero state (success probability too low)"
            )

        return postselected_state / norm

    def get_block_encoding_matrix(self) -> np.ndarray:
        """Get the effective matrix of the circuit after postselection.

        This computes the block encoding matrix A such that when ancillas are
        postselected to |0⟩, the effective operation on data qubits is A.

        Mathematically: ⟨0|_anc U |0⟩_anc = A

        Returns:
            Matrix of shape (2^data_qubits, 2^data_qubits) representing the
            block-encoded operation.
        """
        # Get full unitary matrix
        full_unitary = self.get_full_unitary()

        # Determine which qubits are data vs ancilla
        data_qubit_indices = list(range(self.data_qubits))
        ancilla_qubit_indices = self.circuit.allocation.ancilla_qubits

        # Build the block encoding by projecting ancillas to |0⟩
        # Need to sum over all computational basis states where ancillas are |0⟩
        data_dim = 2**self.data_qubits
        total_dim = 2**self.total_qubits

        block_matrix = np.zeros((data_dim, data_dim), dtype=complex)

        # Iterate over all full basis states
        for in_idx in range(total_dim):
            for out_idx in range(total_dim):
                # Extract bits for each qubit
                in_bits = [
                    (in_idx >> (self.total_qubits - 1 - q)) & 1
                    for q in range(self.total_qubits)
                ]
                out_bits = [
                    (out_idx >> (self.total_qubits - 1 - q)) & 1
                    for q in range(self.total_qubits)
                ]

                # Check if all ancilla qubits are |0⟩
                ancillas_zero_in = all(in_bits[q] == 0 for q in ancilla_qubit_indices)
                ancillas_zero_out = all(out_bits[q] == 0 for q in ancilla_qubit_indices)

                if ancillas_zero_in and ancillas_zero_out:
                    # Extract data qubit values
                    data_in = sum(
                        in_bits[q] * (2 ** (self.data_qubits - 1 - q))
                        for q in data_qubit_indices
                    )
                    data_out = sum(
                        out_bits[q] * (2 ** (self.data_qubits - 1 - q))
                        for q in data_qubit_indices
                    )

                    # Add to block matrix
                    block_matrix[data_out, data_in] += full_unitary[out_idx, in_idx]

        return block_matrix

    def get_full_unitary(self) -> np.ndarray:
        """Get the full unitary matrix of the circuit.

        Returns:
            Full unitary matrix of shape (2^total_qubits, 2^total_qubits)
        """
        # Convert circuit to tensor network
        tn = self.to_tensor_network()

        # Contract to get full matrix
        output_inds = [f"q{q}_out" for q in range(self.total_qubits)]
        input_inds = [f"q{q}_in" for q in range(self.total_qubits)]

        # Contract everything
        matrix_tensor = tn.contract(output_inds=output_inds + input_inds)

        # Reshape to matrix form
        dim = 2**self.total_qubits
        matrix = matrix_tensor.data.reshape((dim, dim))  # type: ignore

        return matrix


def simulate_circuit(circuit: Circuit, initial_data_vector: np.ndarray) -> np.ndarray:
    """Convenience function to simulate a circuit.

    Args:
        circuit: Quantum circuit to simulate
        initial_data_vector: Initial state for data qubits

    Returns:
        Final data state after postselection and normalization
    """
    simulator = CircuitSimulator(circuit)
    return simulator.simulate(initial_data_vector)


def get_block_encoding(circuit: Circuit) -> np.ndarray:
    """Convenience function to get the block encoding matrix.

    Args:
        circuit: Quantum circuit

    Returns:
        Block encoding matrix after ancilla postselection
    """
    simulator = CircuitSimulator(circuit)
    return simulator.get_block_encoding_matrix()
