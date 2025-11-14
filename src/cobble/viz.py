from __future__ import annotations
import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cobble.circuit import Circuit, FlatGate, Gate


def _to_visualization_gates(self: Circuit) -> list[FlatGate]:
    """Convert gates to flat format for visualization, preserving negative controls."""
    result: list[FlatGate] = []
    for gate in self.gates:
        result.extend(_gate_to_vis_flat(gate))
    return result


def _gate_to_vis_flat(gate: Gate) -> list[FlatGate]:
    """Convert a gate to flat format for visualization only (no NOT expansion)."""
    from cobble.circuit import Controlled, FlatGate

    if isinstance(gate, Controlled):
        # For controlled gates, preserve negative controls
        inner_gates = _gate_to_vis_flat(gate.gate)
        result: list[FlatGate] = []

        # Extract control qubits and polarities
        control_qubits = tuple(c.qubit for c in gate.controls)
        control_polarities = tuple(c.polarity for c in gate.controls)

        for fg in inner_gates:
            # Add controls with preserved polarities
            all_controls = control_qubits + fg.controls
            all_polarities = control_polarities + fg.control_polarities
            result.append(
                FlatGate(fg.name, fg.targets, all_controls, fg.params, all_polarities)
            )

        return result
    else:
        # For other gates, use standard flattening
        return gate.to_flat_gates()


def _to_ascii(self: Circuit, max_width: int = 120) -> str:
    """Generate an ASCII diagram of the quantum circuit.

    Args:
        max_width: Maximum width of the diagram before wrapping

    Returns:
        ASCII representation of the circuit
    """
    total_qubits = self.allocation.total_qubits()
    if total_qubits == 0:
        return "Empty circuit (0 qubits)"

    vis_gates = _to_visualization_gates(self)

    if not vis_gates:
        # Empty circuit
        lines = []
        for q in range(total_qubits):
            label = f"q{q}: "
            lines.append(label + "─" * 10)
        # Flip vertically: highest qubit at top, qubit 0 at bottom
        lines.reverse()
        return "\n".join(lines)

    # Build the circuit in segments that fit within max_width
    segments = _build_ascii_segments(vis_gates, total_qubits, max_width)

    # Join segments with continuation markers
    result_lines = []
    for seg_idx, segment in enumerate(segments):
        if seg_idx > 0:
            result_lines.append("")  # Blank line between segments
        # Flip vertically: highest qubit at top, qubit 0 at bottom
        result_lines.extend(reversed(segment))

    return "\n".join(result_lines)


def _build_ascii_segments(
    flat_gates: list[FlatGate], total_qubits: int, max_width: int
) -> list[list[str]]:
    """Build ASCII circuit diagram, potentially split into multiple segments."""
    # Initialize qubit lines
    label_width = max(len(f"q{q}") for q in range(total_qubits)) + 2
    qubit_lines = [f"q{q}: ".ljust(label_width) for q in range(total_qubits)]

    segments = []
    current_segment_width = label_width

    for gate in flat_gates:
        gate_diagram = _render_gate_ascii(gate, total_qubits)
        gate_width = len(gate_diagram[0])

        # Check if new segment
        if (
            current_segment_width + gate_width > max_width
            and current_segment_width > label_width
        ):
            # Save current segment
            segments.append(qubit_lines.copy())
            # Start new segment
            qubit_lines = [f"q{q}: ".ljust(label_width) for q in range(total_qubits)]
            current_segment_width = label_width

        # Append gate to current lines
        for q in range(total_qubits):
            qubit_lines[q] += gate_diagram[q]

        current_segment_width += gate_width

    # Add final segment
    if segments or any(len(line) > label_width for line in qubit_lines):
        segments.append(qubit_lines)

    return segments if segments else [qubit_lines]


def _render_gate_ascii(gate: FlatGate, total_qubits: int) -> list[str]:
    """Render a single gate as ASCII art.

    Returns a list of strings, one per qubit line.
    """
    from cobble.circuit import ControlPolarity

    lines = ["─" * 3 for _ in range(total_qubits)]  # Default: wire continuation

    all_qubits = set(gate.targets) | set(gate.controls)
    if not all_qubits:
        return lines

    min_qubit = min(all_qubits)
    max_qubit = max(all_qubits)

    # Format gate name with parameters
    gate_label = gate.name
    if gate.params:
        # Format parameters (shorten if needed)
        param_strs = []
        for p in gate.params:
            if abs(p - math.pi) < 0.001:
                param_strs.append("π")
            elif abs(p + math.pi) < 0.001:
                param_strs.append("-π")
            elif abs(p - math.pi / 2) < 0.001:
                param_strs.append("π/2")
            elif abs(p + math.pi / 2) < 0.001:
                param_strs.append("-π/2")
            else:
                param_strs.append(f"{p:.2f}")
        gate_label = f"{gate.name}({','.join(param_strs)})"

    if gate.controls:
        # Controlled gate
        padding = "│"

        # Build control symbol map (qubit -> symbol)
        control_symbols: dict[int, str] = {}
        for i, ctrl_qubit in enumerate(gate.controls):
            if i < len(gate.control_polarities):
                # Use polarity information if available
                polarity = gate.control_polarities[i]
                control_symbols[ctrl_qubit] = (
                    "●" if polarity == ControlPolarity.POSITIVE else "○"
                )
            else:
                # Default to positive control
                control_symbols[ctrl_qubit] = "●"

        # Determine gate symbol for target
        if gate.name == "NOT":
            target_symbol = "⊕"
            use_simple_symbol = True
        elif gate.name == "SWAP":
            target_symbol = "×"
            use_simple_symbol = True
        else:
            # Generic controlled gate, use box (including Z)
            target_symbol = f"[{gate_label}]"
            use_simple_symbol = False

        # Calculate width needed
        if use_simple_symbol:
            gate_width = 3  # Standard width: ─X─
        else:
            gate_width = len(target_symbol) + 2  # Add padding on sides

        # Build the gate representation
        for q in range(total_qubits):
            if q in gate.controls:
                # Control qubit - use the appropriate symbol
                ctrl_symbol = control_symbols[q]
                half_width = gate_width // 2
                lines[q] = (
                    "─" * half_width + ctrl_symbol + "─" * (gate_width - half_width - 1)
                )
            elif q in gate.targets:
                # Target qubit
                if use_simple_symbol:
                    half_width = gate_width // 2
                    lines[q] = (
                        "─" * half_width
                        + target_symbol
                        + "─" * (gate_width - half_width - 1)
                    )
                else:
                    # Box notation
                    lines[q] = "─" + target_symbol + "─"
            elif min_qubit < q < max_qubit:
                # Between control and target, vertical line
                half_width = gate_width // 2
                lines[q] = (
                    " " * half_width + padding + " " * (gate_width - half_width - 1)
                )
            else:
                lines[q] = "─" * gate_width

        # Normalize all line widths
        actual_max_width = max(len(line) for line in lines)
        for q in range(total_qubits):
            if len(lines[q]) < actual_max_width:
                if "│" in lines[q] or target_symbol in lines[q]:
                    # For lines with special symbols, pad with spaces
                    lines[q] = lines[q].ljust(actual_max_width, " ")
                else:
                    # For wire lines, pad with dashes
                    lines[q] = lines[q].ljust(actual_max_width, "─")

    elif len(gate.targets) == 1:
        # Single qubit gate
        target = gate.targets[0]

        # Choose representation
        if gate.name in ("NOT", "H", "Z"):
            symbol = {"NOT": "X", "H": "H", "Z": "Z"}[gate.name]
            lines[target] = f"─[{symbol}]─"
        else:
            # Generic single qubit gate
            lines[target] = f"─[{gate_label}]─"

        # Normalize width
        max_width = max(len(line) for line in lines)
        lines = [
            line.ljust(max_width, "─") if i == target else "─" * max_width
            for i, line in enumerate(lines)
        ]

    elif len(gate.targets) == 2 and gate.name == "SWAP":
        # SWAP gate
        t1, t2 = sorted(gate.targets)
        swap_symbol = "×"

        gate_width = 3
        for q in range(total_qubits):
            if q == t1 or q == t2:
                lines[q] = "─" + swap_symbol + "─"
            elif t1 < q < t2:
                lines[q] = " │ "
            else:
                lines[q] = "─" * gate_width

    else:
        # Multi-target gate or black box
        gate_width = max(len(gate_label) + 4, 7)
        box_line = "─" + "┌" + "─" * (gate_width - 4) + "┐" + "─"
        mid_line = " │" + gate_label.center(gate_width - 4) + "│ "
        end_line = "─" + "└" + "─" * (gate_width - 4) + "┘" + "─"

        target_list = sorted(gate.targets)
        min_target = target_list[0]
        max_target = target_list[-1]

        for q in range(total_qubits):
            if q == min_target:
                lines[q] = end_line
            elif q == max_target:
                lines[q] = box_line
            elif min_target < q < max_target:
                if q in target_list:
                    lines[q] = mid_line
                else:
                    lines[q] = " │" + " " * (gate_width - 4) + "│ "
            else:
                lines[q] = "─" * gate_width

    # Ensure all lines have the same width
    max_width = max(len(line) for line in lines)
    lines = [
        (
            line.ljust(max_width, " ")
            if "│" in line or "┌" in line or "└" in line
            else line.ljust(max_width, "─")
        )
        for line in lines
    ]

    return lines


def _format_gate_label_tex(name: str, params: tuple[float, ...]) -> str:
    """Helper to format a gate's name and parameters for LaTeX."""
    is_adjoint = name.endswith("_dag")
    if is_adjoint:
        name = name[:-4]

    # Map common gate names to their LaTeX symbols
    tex_map = {
        "NOT": "X",
        "H": "H",
        "Z": "Z",
        "Rx": "R_x",
        "Ry": "R_y",
        "Rz": "R_z",
        "SU2Rotation": "U_3",
    }
    # Use the mapped name or the original, escaping underscores
    label = tex_map.get(name, name.replace("_", "\\_"))

    if params:
        param_strs = []
        for p in params:
            # Pretty-print multiples of pi
            if abs(p - math.pi) < 0.001:
                param_strs.append("\\pi")
            elif abs(p + math.pi) < 0.001:
                param_strs.append("-\\pi")
            elif abs(p - math.pi / 2) < 0.001:
                param_strs.append("\\pi/2")
            elif abs(p + math.pi / 2) < 0.001:
                param_strs.append("-\\pi/2")
            else:
                param_strs.append(f"{p:.2f}")
        param_str = ",".join(param_strs)
        label = f"{label}({param_str})"

    if is_adjoint:
        label += "^\\dagger"

    return label


def _render_gate_quantikz(gate: FlatGate, total_qubits: int) -> list[str]:
    from cobble.circuit import ControlPolarity

    lines = ["\\qw"] * total_qubits
    if not (gate.targets or gate.controls):
        return lines

    label = _format_gate_label_tex(gate.name, gate.params)
    target_qubits = sorted(gate.targets)

    # Place controls first
    for i, ctrl_qubit in enumerate(gate.controls):
        polarity = gate.control_polarities[i]
        if target_qubits:
            nearest_target = min(target_qubits, key=lambda t: abs(t - ctrl_qubit))
            ctrl_pos = total_qubits - 1 - ctrl_qubit
            target_pos = total_qubits - 1 - nearest_target
            wire_distance = target_pos - ctrl_pos
            lines[ctrl_qubit] = (
                f"\\ctrl{{{wire_distance}}}"
                if polarity == ControlPolarity.POSITIVE
                else f"\\octrl{{{wire_distance}}}"
            )
        else:
            # Single-qubit control with no explicit target
            lines[ctrl_qubit] = (
                "\\ctrl{0}" if polarity == ControlPolarity.POSITIVE else "\\octrl{0}"
            )

    # Place targets
    if gate.name == "SWAP":
        t1, t2 = target_qubits
        lines[t1] = "\\swap"
        lines[t2] = "\\targX"
        for q in range(t1 + 1, t2):
            lines[q] = ""
    elif len(target_qubits) == 1:
        if label == "X" and gate.controls:
            lines[target_qubits[0]] = "\\targ{}"
        else:
            lines[target_qubits[0]] = f"\\gate{{{label}}}"
    else:
        # Multi-qubit gate
        min_q, max_q = target_qubits[0], target_qubits[-1]
        is_contiguous = len(target_qubits) == max_q - min_q + 1
        style = "" if is_contiguous else ", style={dashed}"

        # Anchor at visually top-most wire
        top_wire = max_q
        lines[top_wire] = f"\\gate[wires={max_q - min_q + 1}{style}]{{{label}}}"

        # Clear other wires in the box
        for q in range(min_q, top_wire):
            lines[q] = ""

    return lines


def _to_tex(self: Circuit) -> str:
    """Generate a TeX file using the quantikz2 package to visualize the circuit."""
    total_qubits = self.allocation.total_qubits()
    if total_qubits == 0:
        return "% Empty circuit"

    vis_gates = _to_visualization_gates(self)

    # LaTeX preamble
    header = [
        "\\documentclass{standalone}",
        "\\usepackage{tikz}",
        "\\usetikzlibrary{quantikz2}",
        "\\begin{document}",
        "\\begin{quantikz}",
    ]

    # Render all gates
    columns = [_render_gate_quantikz(g, total_qubits) for g in vis_gates]

    # Construct rows, top-to-bottom inversion
    body = []
    for q in reversed(range(total_qubits)):
        row_cmds = [f"\\lstick{{$q_{{{q}}}$}}"]
        row_cmds.extend(columns[i][q] for i in range(len(vis_gates)))
        row_cmds.append("\\qw")
        body.append(" & ".join(row_cmds) + " \\\\")

    footer = [
        "\\end{quantikz}",
        "\\end{document}",
    ]

    return "\n".join(header + body + footer)
