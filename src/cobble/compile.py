from __future__ import annotations
from dataclasses import replace
import math
import os
from typing import TYPE_CHECKING
import warnings

from cobble.circuit import (
    BlackBox,
    Circuit,
    Control,
    Controlled,
    Gate,
    GlobalPhase,
    Hadamard,
    Increment,
    NOT,
    QubitAllocation,
    Rz,
    XorInt,
    Z,
    sign_correction,
    state_preparation_tree,
)
from cobble.polynomial import Polynomial, TOL
import numpy as np
from numpy.polynomial.chebyshev import poly2cheb

if TYPE_CHECKING:
    from cobble.expr import Basic, Const, Expr, If, Poly, Prod, Sum, Tensor


def _compile_base(self: Basic) -> Circuit:
    data_qubits = self.qtype().width()
    ancilla_indices = list(range(data_qubits, data_qubits + self.ancilla_qubits_))
    allocation = QubitAllocation(data_qubits, ancilla_indices)
    return Circuit(
        [BlackBox(self.name, data_qubits, self.ancilla_qubits_)],
        allocation,
    )


def _compile_const(self: Const) -> Circuit:
    phase = float(np.angle(self.value))
    if phase - 0.0 < TOL:
        return Circuit.identity(self.qtype().width(), 0)
    else:
        gates = [GlobalPhase(self.qtype().width(), phase)]
        return Circuit(gates, QubitAllocation(self.qtype().width(), [0]))


def _compile_single_term_sum(term: tuple[float, Expr]) -> Circuit:
    coeff, expr = term
    circ = expr.circuit()
    if coeff > 0:
        return circ
    else:
        # Negative coefficient: add global phase
        gates = [GlobalPhase(0, math.pi)] + list(circ.gates)
        return Circuit(gates, circ.allocation)


def _compile_sum(self: Sum) -> Circuit:
    """Compile Sum using LCU: state prep, sign correction, SELECT oracle, uncompute."""
    if not self.terms:
        raise ValueError("Cannot compile empty sum")
    if len(self.terms) == 1:
        return _compile_single_term_sum(self.terms[0])

    data_qubits = self.qtype().width()
    max_term_ancillas = max(term.ancilla_qubits() for _, term in self.terms)
    num_lambda_ancillas = math.ceil(math.log2(len(self.terms)))

    ancilla_indices = list(range(data_qubits, data_qubits + self.ancilla_qubits()))
    allocation = QubitAllocation(data_qubits, ancilla_indices)

    term_ancillas = ancilla_indices[:max_term_ancillas]
    lambda_ancillas = ancilla_indices[
        max_term_ancillas : max_term_ancillas + num_lambda_ancillas
    ]

    # term.circuit() encodes term/alpha(term), so need coeff * alpha(term)
    coeffs = [coeff * term.subnormalization() for coeff, term in self.terms]

    gates: list[Gate] = []
    # State preparation on lambda register
    prep_gates = [
        replace(g, is_conjugate_pair=True)
        for g in state_preparation_tree(coeffs, lambda_ancillas)
    ]
    gates.extend(prep_gates)

    # Sign correction for negative coefficients
    gates.extend(sign_correction(coeffs, lambda_ancillas))

    # SELECT oracle: apply each term controlled on lambda register state |i>
    for i, (_, term) in enumerate(self.terms):
        term_circ = term.circuit()
        remapped_circ = term_circ.remap_qubits(term_ancillas, data_offset=0)

        # Control on lambda register being in state |i>
        binary = format(i, f"0{num_lambda_ancillas}b")
        controls: list[Control | int] = []
        for bit_idx, bit_val in enumerate(binary):
            qubit = lambda_ancillas[bit_idx]
            if bit_val == "1":
                controls.append(qubit)
            else:
                controls.append(Control.neg(qubit))

        for gate in remapped_circ.gates:
            gates.append(Controlled.add_controls(gate, tuple(controls)))

    # Uncompute state preparation (sign correction remains for phases)
    for gate in reversed(prep_gates):
        gates.append(gate.adjoint())

    return Circuit(gates, allocation)


def _compile_prod(self: Prod) -> Circuit:
    """Compile product: sequence factors in reverse, track with flag register."""
    if not self.factors:
        raise ValueError("Cannot compile empty product")
    if len(self.factors) == 1:
        return self.factors[0].circuit()

    data_qubits = self.qtype().width()
    num_factors = len(self.factors)
    max_factor_ancillas = max(f.ancilla_qubits() for f in self.factors)
    num_flag_qubits = math.ceil(math.log2(num_factors))

    ancilla_indices = list(range(data_qubits, data_qubits + self.ancilla_qubits()))
    allocation = QubitAllocation(data_qubits, ancilla_indices)

    working_ancillas = ancilla_indices[:max_factor_ancillas]
    flag_register = ancilla_indices[
        max_factor_ancillas : max_factor_ancillas + num_flag_qubits
    ]

    gates: list[Gate] = []
    # Process factors in reverse: UV encodes AB, so build V first, then U
    for factor_idx, factor in enumerate(reversed(self.factors)):
        circ = factor.circuit()
        remapped_circ = circ.remap_qubits(working_ancillas, data_offset=0)
        gates.extend(remapped_circ.gates)

        if factor_idx < len(self.factors) - 1 and flag_register:
            if working_ancillas:
                controls = tuple(Control.neg(anc) for anc in working_ancillas)
                gates.append(
                    Controlled.add_controls(Increment(tuple(flag_register)), controls)
                )
            else:
                gates.append(Increment(tuple(flag_register)))

    if flag_register:
        gates.append(XorInt(tuple(flag_register), num_factors - 1))

    return Circuit(gates, allocation)


def _compile_tensor(self: Tensor) -> Circuit:
    if not self.factors:
        raise ValueError("Cannot compile empty tensor product")

    if len(self.factors) == 1:
        return self.factors[0].circuit()

    data_qubits = self.qtype().width()
    ancilla_indices = list(range(data_qubits, data_qubits + self.ancilla_qubits()))
    allocation = QubitAllocation(data_qubits, ancilla_indices)

    gates: list[Gate] = []
    data_offset = 0
    ancilla_offset = 0

    for factor in self.factors:
        circ = factor.circuit()
        factor_data_width = factor.qtype().width()
        factor_ancilla_count = factor.ancilla_qubits()
        data_mapping = {i: data_offset + i for i in range(factor_data_width)}
        ancilla_targets = ancilla_indices[
            ancilla_offset : ancilla_offset + factor_ancilla_count
        ]
        remapped_circ = circ.remap_qubits(ancilla_targets, data_mapping=data_mapping)
        gates.extend(remapped_circ.gates)

        data_offset += factor_data_width
        ancilla_offset += factor_ancilla_count

    return Circuit(gates, allocation)


def _compile_if(self: If) -> Circuit:
    then_circ = self.then_branch.circuit()
    else_circ = self.else_branch.circuit()

    branch_data_qubits = self.then_branch.qtype().width()

    # Condition qubit is part of the data register
    cond_qubit = branch_data_qubits
    total_data_qubits = branch_data_qubits + 1

    ancilla_indices = list(
        range(total_data_qubits, total_data_qubits + self.ancilla_qubits())
    )
    allocation = QubitAllocation(total_data_qubits, ancilla_indices)

    then_circ_remapped = then_circ.remap_qubits(ancilla_indices, data_offset=0)
    else_circ_remapped = else_circ.remap_qubits(ancilla_indices, data_offset=0)

    gates: list[Gate] = []

    if self.cond.active:
        # if x then A else B: A controlled on x=|1>, B controlled on x=|0>
        for gate in then_circ_remapped.gates:
            gates.append(Controlled.add_controls(gate, (cond_qubit,)))
        for gate in else_circ_remapped.gates:
            gates.append(Controlled.add_controls(gate, (Control.neg(cond_qubit),)))
    else:
        # if not x then A else B: A controlled on x=|0>, B controlled on x=|1>
        for gate in then_circ_remapped.gates:
            gates.append(Controlled.add_controls(gate, (Control.neg(cond_qubit),)))
        for gate in else_circ_remapped.gates:
            gates.append(Controlled.add_controls(gate, (cond_qubit,)))

    return Circuit(gates, allocation)


def _compile_constant_polynomial(
    self: Poly, rot_ancilla: int, allocation: QubitAllocation
) -> Circuit:
    gates = []
    if self.p.coeffs[0] < 0.0:
        gates.append(NOT(rot_ancilla, is_conjugate_pair=True))
        gates.append(Z(rot_ancilla))
        gates.append(NOT(rot_ancilla, is_conjugate_pair=True))
    return Circuit(gates, allocation)


def _compile_poly(self: Poly) -> Circuit:
    """Compile polynomial using QSP: decompose mixed parity, compute angles, build QSP circuit."""
    from cobble.expr import Dagger, Poly

    if self.p.parity() == Polynomial.Parity.MIXED:
        return (
            Poly(self.expr, self.p.even_component())
            + Poly(self.expr, self.p.odd_component())
        ).circuit()

    expr_data_qubits = self.expr.qtype().width()
    n_total_ancillas = 1 + self.expr.ancilla_qubits()
    ancilla_indices = list(range(expr_data_qubits, expr_data_qubits + n_total_ancillas))
    allocation = QubitAllocation(expr_data_qubits, ancilla_indices)

    rot_ancilla = ancilla_indices[-1]
    expr_ancilla_indices = ancilla_indices[:-1]

    if self.p.is_constant():
        return _compile_constant_polynomial(self, rot_ancilla, allocation)

    remapped_circ = self.expr.circuit().remap_qubits(
        expr_ancilla_indices, data_offset=0
    )
    if self.expr.is_hermitian_block_encoding():
        remapped_adj_circ = remapped_circ
    else:
        remapped_adj_circ = (
            Dagger(self.expr)
            .circuit()
            .remap_qubits(expr_ancilla_indices, data_offset=0)
        )

    poly = self.scaled_p()
    normalized_coeffs = np.array(poly.coeffs) / self.subnormalization()
    angles = _compute_qsp_angles(normalized_coeffs, poly.degree())

    gates = []
    gates.append(Hadamard(rot_ancilla, is_conjugate_pair=True))

    anti_controls = tuple(Control.neg(anc) for anc in expr_ancilla_indices)
    projector_gate = replace(
        Controlled.add_controls(NOT(rot_ancilla), anti_controls),
        is_conjugate_pair=True,
    )
    d = poly.degree()

    for i in range(d + 1):
        gates.append(projector_gate)
        gates.append(Rz(rot_ancilla, 2 * angles[i]))
        gates.append(projector_gate)

        if i < d:
            circ = remapped_circ if i % 2 == 0 else remapped_adj_circ
            gates.extend(circ.gates)

    gates.append(Hadamard(rot_ancilla, is_conjugate_pair=True))

    return Circuit(gates, allocation)


def _compute_qsp_angles(normalized_coeffs: np.ndarray, degree: int) -> list[float]:
    """Compute QSP angles using configured solver."""
    from cobble.expr import QSP_TOL

    if "NOSOLVER" in os.environ:
        solver = "constant"
    else:
        solver = os.environ.get("SOLVER", "pennylane")

    if solver == "constant":
        return [1.23456789] * (degree + 1)
    elif solver == "pennylane":
        from pennylane import poly_to_angles

        with warnings.catch_warnings(action="ignore", category=DeprecationWarning):
            return poly_to_angles(normalized_coeffs, "QSVT", angle_solver="iterative")  # type: ignore
    elif solver == "pyqsp":
        from pennylane import transform_angles
        from pyqsp import angle_sequence

        pyqsp_out = angle_sequence.QuantumSignalProcessingPhases(  # type: ignore
            poly2cheb(normalized_coeffs),
            method="sym_qsp",
            chebyshev_basis=True,
            tolerance=QSP_TOL,
        )[0]
        pyqsp_out[0] -= math.pi / 4
        pyqsp_out[-1] -= math.pi / 4
        return transform_angles(pyqsp_out, "QSP", "QSVT")
    else:
        raise ValueError(f"Unknown SOLVER: {solver}")


def _get_lcu_horner(P: Poly | Sum) -> tuple[Expr, Expr]:
    """Get the LCU and Horner forms of a polynomial or sum of polynomials."""

    from cobble.expr import Poly, Sum

    if isinstance(P, Poly):
        P_lcu = P.lcu()
        P_horner = P.horner()
    else:
        lcu_terms = []
        horner_terms = []
        for coef, term in P.terms:
            if isinstance(term, Poly):
                lcu_terms.append((coef, term.lcu()))
                horner_terms.append((coef, term.horner()))
            else:
                lcu_terms.append((coef, term))
                horner_terms.append((coef, term))
        P_lcu = Sum(lcu_terms)
        P_horner = Sum(horner_terms)

    return P_lcu, P_horner
