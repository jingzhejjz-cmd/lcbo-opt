"""Small LCBO run with fixed GP hyperparameters from the synthetic benchmark.

This mirrors the ``lcbo_origin/synthetic_25.py`` setting: the target and
constraints are generated from known GP priors, so LCBO receives those true
kernel hyperparameters and keeps them fixed during optimization.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from lcbo_opt import LCBO, build_prior_config_from_problem, make_benchmark_problem, run_lcbo_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=25)
    parser.add_argument("--constraints", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--n0", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double
    noise_variance = 1e-2

    problem = make_benchmark_problem(
        "within_model",
        dim=args.dim,
        n_constraints=args.constraints,
        seed=args.seed,
        noise_variance=noise_variance,
        offset=0.5,
        device=device,
        dtype=dtype,
    )
    prior_config = build_prior_config_from_problem(problem, noise_variance=noise_variance)

    optimizer = LCBO(
        problem=problem,
        max_objective_calls=1000,
        n_repeats_gradient=2,
        max_active_points=2,
        temp_lse=1.0,
        rho_fn=lambda k: 10.0 * (k + 1) ** 0.25,
        step_size_fn=lambda k: 0.5,
        prior_config=prior_config,
        N_max=args.dim * 2,
        set_hyper=True,
        device=device,
        dtype=dtype,
    )

    result = run_lcbo_experiment(
        problem,
        optimizer,
        n_runs=1,
        base_seed=args.seed,
        max_iterations=args.iterations,
        n0=args.n0 or args.dim,
    )

    logs = result["logs"]
    last = logs[-1] if logs else {}
    best = result["best_objective"]
    best_text = "none" if not np.isfinite(best) else f"{best:.6g}"
    print(f"completed_iterations={len(logs)}")
    print(f"last_call_counter={last.get('call_counter')}")
    print(f"best_feasible_objective={best_text}")
    print("kernel=RBF isotropic, hyperparameters=fixed")


if __name__ == "__main__":
    main()
