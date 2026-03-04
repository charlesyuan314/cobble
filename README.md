# Cobble: Compiling Block Encodings for Quantum Computational Linear Algebra

This repository contains a research prototype implementation of the Cobble language from the paper ["Cobble: Compiling Block Encodings for Quantum Computational Linear Algebra"](https://arxiv.org/abs/2511.01736). This code is provided as-is to aid in reproducibility. It is not yet intended for production use, and a more complete reproduction package is under development.

## Source organization

- **`src/cobble/expr.py`** — Expression AST: `Basic`, `Const`, `Sum`, `Prod`, `Tensor`, `Poly`, `If`, `Division`, `Dagger`, and the base `Expr` class. Expressions are built in Python and compiled to circuits.
- **`src/cobble/compile.py`** — Compilation from expressions to circuits (LCU, QSP, etc.).
- **`src/cobble/circuit.py`** — Circuit representation (gates, allocations, controlled operations).
- **`src/cobble/qasm.py`** — Export of circuits to OpenQASM.
- **`src/cobble/optimize.py`** — Expression-level optimization (normalization, fusion, rewriting).
- **`src/cobble/polynomial.py`** — Polynomial types used by `Poly(expr, polynomial)`.
- **`src/cobble/qtype.py`** — Types for expressions (e.g. `BitType`, tensor types).
- **`src/cobble/simulator.py`** — Simulator for testing block encodings.
- **`examples/`** — Example Cobble programs (paper benchmarks and small demos).
- **`main.py`** — Entry point: runs the paper examples, optional QASM generation and timing.

## To Run

```shell
pip install -r requirements.txt
pip install -e .
pytest  # run unit tests
python3 -i main.py  # run paper examples and enter interpreter
```
