from argparse import ArgumentParser
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import os
import sys
import time

from cobble.compile import _get_lcu_horner
from cobble.expr import Basic, Expr, Poly, Sum
import numpy as np

from examples.hamiltonian_simulation import hamiltonian_simulation
from examples.laplacian_filter import laplacian_filter
from examples.matrix_inversion import matrix_inversion
from examples.ols_ridge import ols_ridge
from examples.penalized_coupler import penalized_coupler
from examples.regression_example import regression_example
from examples.simple_1 import simple_1
from examples.simple_2 import simple_2
from examples.simple_3 import simple_3
from examples.simulation_example import simulation_example
from examples.spectral_thresholding import spectral_thresholding


@contextmanager
def suppress_stdout_stderr():
    """A context manager that redirects stdout and stderr to devnull"""
    with open(os.devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


# ANSI color codes
GREEN = "\033[92m"
BOLD = "\033[1m"
ORANGE = "\033[33m"
RESET = "\033[0m"


def colorize(value: str, condition: bool, color: str) -> str:
    """Apply color to value if condition is met."""
    return f"{color}{value}{RESET}" if condition else value


def colorize_comparison(old, new, fmt: str = "") -> str:
    """Color new value: green if improved (less), orange if worse (more)."""
    new_str = f"{new:{fmt}}" if fmt else str(new)
    if new < old:
        return colorize(new_str, True, BOLD + GREEN)
    elif new > old:
        return colorize(new_str, True, ORANGE)
    return new_str


def assert_program(name: str, expr: Expr, target: Expr):
    _ = expr.qtype()
    e_norm = expr.optimize()
    t_norm = target.optimize()
    if not Expr.structural_eq(e_norm, target):
        print(f"[FAIL] {name}", file=sys.stderr)
        print(f"  got   : {e_norm}", file=sys.stderr)
        print(f"  expect: {t_norm}", file=sys.stderr)
        return False

    # Show costs before and after normalization
    a0, g0, n0, c0 = (
        expr.subnormalization(),
        expr.queries(),
        expr.ancilla_qubits(),
        expr.total_cost(),
    )
    a1, g1, n1, c1 = (
        e_norm.subnormalization(),
        e_norm.queries(),
        e_norm.ancilla_qubits(),
        e_norm.total_cost(),
    )
    e_norm_str = str(e_norm)
    if len(e_norm_str + name) > 72:
        e_norm_str = e_norm_str[: (72 - len(name))] + "..."
    print(f"{BOLD}{name}{RESET} -> {e_norm_str}", file=sys.stderr)

    speedup = c0 / c1
    print(
        f"α={a0:.4g}→{colorize_comparison(a0, a1, '.4g')}, "
        f"g={g0}→{colorize_comparison(g0, g1)}, "
        f"n={n0}→{colorize_comparison(n0, n1)}, "
        f"cost={c0:.1f}->{colorize_comparison(c0, c1, '.1f')}, "
        f"speedup={colorize(f'{speedup:.1f}x', speedup > 1, BOLD + GREEN)}\n",
        file=sys.stderr,
    )
    return True


def process_basic_examples() -> bool:
    ok_all = True
    ok_all &= assert_program(*simple_1(A=Basic("A")))
    ok_all &= assert_program(*simple_2(A=Basic("A")))
    ok_all &= assert_program(*simple_3(A=Basic("A"), B=Basic("B")))
    return ok_all


def process_paper_examples() -> bool:
    ok_all = True
    ok_all &= assert_program(*simulation_example(X=Basic("X"), Y=Basic("Y")))
    ok_all &= assert_program(*regression_example(A=Basic("A"), B=Basic("B")))
    ok_all &= assert_program(*penalized_coupler())
    ok_all &= assert_program(*laplacian_filter(Hx=Basic("Hx"), Hy=Basic("Hy")))
    ok_all &= assert_program(*ols_ridge(X=Basic("X")))
    return ok_all


def process_algorithmic_example(name: str, P: Poly | Sum) -> None:
    def print_costs(name: str, e: Expr, baseline: float | None = None):
        a, g, n, c = (
            e.subnormalization(),
            e.queries(),
            e.ancilla_qubits(),
            e.total_cost(),
        )
        baseline_str = ""
        if baseline is not None:
            slowdown = c / baseline
            baseline_str = f", slowdown={colorize(f'{slowdown:.1f}x', slowdown > 1, BOLD + ORANGE)}"
        print(
            f"{BOLD}{name}{RESET}: α={a:.4g}, g={g}, n={n}, cost={c:.1f}{baseline_str}",
            file=sys.stderr,
        )

    P_lcu, P_horner = _get_lcu_horner(P)
    print_costs(name, P)
    print_costs("LCU", P_lcu, P.total_cost())
    print_costs("Horner", P_horner, P.total_cost())
    print(file=sys.stderr)


def process_algorithmic_examples() -> None:
    process_algorithmic_example(*matrix_inversion(X=Basic("X")))
    process_algorithmic_example(*hamiltonian_simulation(X=Basic("X"), real_phase=True))
    process_algorithmic_example(*spectral_thresholding(X=Basic("X")))


def process_timing_examples() -> None:
    from cobble.qasm import circuit_to_gate_count
    from examples.chebyshev import T_n

    iters = 10
    N = 30
    X = Basic("X")
    os.environ["SOLVER"] = "constant"
    gate_counts = {}
    print("Counting gates for unoptimized Chebyshev polynomials...", file=sys.stderr)
    for n in range(2, N):
        T_n_expr = T_n(n, X)[1]
        circ = T_n_expr.circuit()
        num_gates = circuit_to_gate_count(circ)
        print(f"T_{n}(X): {num_gates} unoptimized gates.", file=sys.stderr, flush=True)
        gate_counts[n] = num_gates
    del os.environ["SOLVER"]

    print(
        "Optimizing + compiling Chebyshev polynomials...",
        file=sys.stderr,
    )

    for n in range(2, N):
        print(f"T_{n}(X)", file=sys.stderr)
        solver_times = np.zeros(iters)
        non_solver_times = np.zeros(iters)
        for i in range(iters):
            print(f"{i} ", end="", file=sys.stderr, flush=True)
            start_time = time.time_ns()
            T_n_simplified = T_n(n, X)[1].optimize()
            optimize_time = time.time_ns() - start_time

            os.environ["SOLVER"] = "constant"
            constant_solver_start_time = time.time_ns()
            _ = T_n_simplified.circuit()
            constant_solver_end_time = time.time_ns()
            constant_solver_time = constant_solver_end_time - constant_solver_start_time

            os.environ["SOLVER"] = "pyqsp"
            # Suppress PyQSP logging
            with suppress_stdout_stderr():
                solver_start_time = time.time_ns()
                circ = T_n_simplified.circuit()
                solver_end_time = time.time_ns()
            del os.environ["SOLVER"]
            net_solver_time = solver_end_time - solver_start_time - constant_solver_time

            qasm_start_time = time.time_ns()
            _ = circ.to_qasm()
            qasm_time = time.time_ns() - qasm_start_time

            non_solver_time = qasm_time + constant_solver_time + optimize_time

            solver_times[i] = net_solver_time
            non_solver_times[i] = non_solver_time
        print("", file=sys.stderr)

        solver_times /= 1e9
        non_solver_times /= 1e9
        print(
            f"Solver time: {np.mean(solver_times):.4f} s +/- {np.std(solver_times, ddof=1) / np.sqrt(iters):.4f} s"
        )
        print(
            f"Non-solver time: {np.mean(non_solver_times):.4f} s +/- {np.std(non_solver_times, ddof=1) / np.sqrt(iters):.4f} s"
        )


def process_qasm_example(example: str, optimized: bool = False) -> None:
    from examples.chebyshev import T_n

    basic_example_dict = {
        "simple-1": (simple_1, Basic("$A")),
        "simple-2": (simple_2, Basic("$A")),
        "simple-3": (simple_3, Basic("$A"), Basic("$B")),
        "simulation-example": (
            simulation_example,
            Basic("X"),
            Basic("Z"),
        ),  # use actual unitaries, and use Z for broader support
        "regression-example": (regression_example, Basic("$A"), Basic("$B")),
        "penalized-coupler": (penalized_coupler,),
        "laplacian-filter": (laplacian_filter, Basic("$Hx"), Basic("$Hy")),
        "ols-ridge": (ols_ridge, Basic("$X")),
    }

    algorithmic_example_dict = {
        "matrix-inversion": (matrix_inversion, Basic("$X")),
        "hamiltonian-simulation": (hamiltonian_simulation, Basic("$X")),
        "spectral-thresholding": (spectral_thresholding, Basic("$X")),
        "chebyshev": (T_n, 10, Basic("$X")),
    }

    if example == "all":
        for example in basic_example_dict:
            process_qasm_example(example, optimized)
        for example in algorithmic_example_dict:
            process_qasm_example(example, optimized)
        return

    if (x := basic_example_dict.get(example)) is not None:
        example_func, *args = x
        _, expr, _ = example_func(*args)
        expr = expr.optimize() if optimized else expr
    elif (x := algorithmic_example_dict.get(example)) is not None:
        example_func, *args = x
        _, expr = example_func(*args)
        expr = expr if optimized else _get_lcu_horner(expr)[0]
    else:
        raise ValueError(f"Unknown example: {example}")

    print(f"Processing {example}...", file=sys.stderr)

    circ = expr.circuit()
    qasm = circ.to_qasm()

    suffix = "-opt" if optimized else ""
    with open(example + suffix + ".qasm", "w") as f:
        print("// " + example + suffix + ".qasm", file=f)
        print(qasm, file=f)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--qasm", help="Generate QASM output for example (or 'all')", action="store"
    )
    parser.add_argument(
        "--opt",
        help="Optimize examples before generating QASM",
        action="store_true",
    )
    parser.add_argument("--timing", action="store_true", help="Compute timing results")
    args = parser.parse_args()

    if args.qasm and args.timing:
        raise ValueError("Cannot specify both --qasm and --timing")

    if args.qasm:
        process_qasm_example(args.qasm, optimized=args.opt)
        raise SystemExit(0)

    if args.timing:
        print(BOLD + "=" * 80 + RESET, file=sys.stderr)
        print(BOLD + "Processing timing examples...\n" + RESET, file=sys.stderr)
        process_timing_examples()
        raise SystemExit(0)

    print(BOLD + "=" * 80 + RESET, file=sys.stderr)
    print(BOLD + "Processing basic examples...\n" + RESET, file=sys.stderr)
    success = process_basic_examples()
    if not success:
        raise SystemExit(1)

    print(BOLD + "=" * 80 + RESET, file=sys.stderr)
    print(BOLD + "Processing paper examples...\n" + RESET, file=sys.stderr)
    success = process_paper_examples()
    if not success:
        raise SystemExit(1)

    print(BOLD + "=" * 80 + RESET, file=sys.stderr)
    print(BOLD + "Processing algorithmic examples...\n" + RESET, file=sys.stderr)
    process_algorithmic_examples()
    print(BOLD + "=" * 80 + RESET, file=sys.stderr)
