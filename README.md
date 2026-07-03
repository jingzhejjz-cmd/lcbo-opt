# lcbo-opt

`lcbo-opt` packages the LCBO optimizer and a small set of constrained benchmark utilities.

## Install

```bash
pip install -e .
```

Use the optional RL benchmark dependencies with:

```bash
pip install -e ".[rl]"
```

## Quick example

Open `examples/toy_example.ipynb` for a minimal LCBO run that passes Python
objects directly to the optimizer.

The public optimizer API is:

```python
from lcbo_opt import LCBO, make_benchmark_problem
```
