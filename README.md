# lcbo-opt

Official implementation of **Local Constrained Bayesian Optimization (LCBO)**:

```bibtex
@misc{jingzhe2026localconstrainedbayesianoptimization,
  title={Local Constrained Bayesian Optimization},
  author={Jing Jingzhe and Fan Zheyi and Szu Hui Ng and Qingpei Hu},
  year={2026},
  eprint={2603.07965},
  archivePrefix={arXiv},
  primaryClass={stat.ML},
  url={https://arxiv.org/abs/2603.07965},
}
```

`lcbo-opt` packages the LCBO optimizer, exact GP surrogate utilities, and
benchmark wrappers used by the paper experiments.

## Install

Use Python 3.10 or newer.

```bash
pip install -e .
```

The HalfCheetah benchmark requires the optional RL dependencies:

```bash
pip install -e ".[rl]"
```

## Examples

Run the known-GP synthetic setting, where the true GP hyperparameters are passed
to LCBO and fixed throughout the run:

```bash
python examples/synthetic_fixed_hyperparameters.py --iterations 3
```

Run the unknown-hyperparameter setting, where LCBO receives priors, fixes only
the noise variance, and optimizes the other GP hyperparameters by MLE every
fifth iteration:

```bash
python examples/unknown_mle_hyperparameters.py --iterations 6
```

After installing `.[rl]`, the same MLE example can be run on HalfCheetah:

```bash
python examples/unknown_mle_hyperparameters.py --benchmark halfcheetah --iterations 6 --horizon 50 --n-repeats 1
```

All bundled examples use the default isotropic RBF kernel. To experiment with a
custom kernel, pass `kernel_factory` to `LCBO`; leaving it unset preserves the
isotropic behavior of the original implementation.

## Public API

```python
from lcbo_opt import LCBO, make_benchmark_problem, run_lcbo_experiment
```

For synthetic benchmarks generated from known GP priors, use:

```python
from lcbo_opt import build_prior_config_from_problem
```
