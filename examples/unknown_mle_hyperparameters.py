"""Small LCBO run with GP priors and periodic MLE hyperparameter updates.

This is the lightweight analogue of the ``lcbo_origin/halfcheetah.py`` setup:
the optimizer does not receive true GP hyperparameters. It starts from priors,
keeps the observation noise fixed, and optimizes lengthscale/outputscale by MLE
every fifth LCBO iteration. Pass ``--benchmark halfcheetah`` after installing
the optional RL dependencies to run the same pattern on HalfCheetah.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch
from gpytorch.constraints import GreaterThan
from gpytorch.priors import GammaPrior, NormalPrior

from lcbo_opt import LCBO, make_benchmark_problem, run_lcbo_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["synthetic", "halfcheetah"], default="synthetic")
    parser.add_argument("--dim", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--n0", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=50)
    parser.add_argument("--n-repeats", type=int, default=1)
    return parser.parse_args()


def make_problem(args: argparse.Namespace, device: torch.device, dtype: torch.dtype):
    if args.benchmark == "halfcheetah":
        return make_benchmark_problem(
            "halfcheetah",
            seed=args.seed,
            horizon=args.horizon,
            n_repeats=args.n_repeats,
            noise_variance=1e-2,
            device=device,
            dtype=dtype,
        )

    return make_benchmark_problem(
        "within_model",
        dim=args.dim,
        n_constraints=1,
        seed=args.seed,
        noise_variance=1e-2,
        offset=0.25,
        device=device,
        dtype=dtype,
    )


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    problem = make_problem(args, device, dtype)
    common_model_config = {
        "lengthscale": {
            "prior": GammaPrior(concentration=3.0, rate=3.0),
            "base_init": 1.0,
        },
        "outputscale": {
            "constraint": GreaterThan(0.001),
            "prior": NormalPrior(loc=2.0, scale=1.0),
        },
        "noise": {"fixed": 0.01},
    }
    prior_config = {"obj": common_model_config}
    for i in range(len(problem.constraints)):
        prior_config[f"cons_{i}"] = common_model_config

    optimizer = LCBO(
        problem=problem,
        max_objective_calls=1000,
        n_repeats_gradient=1,
        max_active_points=1,
        temp_lse=1.0,
        rho_fn=lambda k: 10.0 * (k + 1) ** 0.25,
        step_size_fn=lambda k: 0.5,
        prior_config=prior_config,
        N_max=max(10, min(problem.dim * 2, 30)),
        set_hyper=False,
        device=device,
        dtype=dtype,
    )

    result = run_lcbo_experiment(
        problem,
        optimizer,
        n_runs=1,
        base_seed=args.seed,
        max_iterations=args.iterations,
        n0=args.n0,
    )

    logs = result["logs"]
    last = logs[-1] if logs else {}
    best = result["best_objective"]
    best_text = "none" if not np.isfinite(best) else f"{best:.6g}"
    print(f"benchmark={args.benchmark}")
    print(f"completed_iterations={len(logs)}")
    print(f"last_call_counter={last.get('call_counter')}")
    print(f"best_feasible_objective={best_text}")
    print("kernel=RBF isotropic, hyperparameters=MLE every 5 iterations")


if __name__ == "__main__":
    main()
