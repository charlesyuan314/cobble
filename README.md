# Cobble: Compiling Block Encodings for Quantum Computational Linear Algebra

This repository contains a research prototype implementation of the Cobble language from the paper ["Cobble: Compiling Block Encodings for Quantum Computational Linear Algebra"](https://arxiv.org/abs/2511.01736). This code is provided as-is to aid in reproducibility. It is not yet intended for production use, and a more complete reproduction package is under development.

To run:

```shell
pip install -r requirements.txt
pip install -e .
pytest  # run unit tests
python3 -i main.py  # run paper examples and enter interpreter
```
