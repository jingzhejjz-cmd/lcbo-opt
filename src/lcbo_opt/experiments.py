"""Experiment helpers for LCBO examples and smoke tests."""

import json
import os
import time
import copy

import numpy as np
import torch

from .base import DEFAULT_DEVICE, DEFAULT_DTYPE


def generate_initial_data(
    optimizer,
    objective_func,
    constraint_funcs,
    bounds,
    n0: int = 50,
    seed: int = 42,
):
    """Generate random initial observations and return a starting point."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    bounds = bounds.to(device=optimizer.device, dtype=optimizer.dtype)
    dim = bounds.shape[1]
    X_init_norm = torch.rand(n0, dim, dtype=optimizer.dtype, device=optimizer.device)
    X_init = X_init_norm * (bounds[1] - bounds[0]) + bounds[0]

    Y_obj_init = torch.zeros(n0, 1, dtype=optimizer.dtype, device=optimizer.device)
    Y_cons_init_list = [
        torch.zeros(n0, 1, dtype=optimizer.dtype, device=optimizer.device)
        for _ in constraint_funcs
    ]

    for i in range(n0):
        theta = X_init[i]
        Y_obj_init[i, 0] = objective_func(theta, obs_noise=False)
        for j, constraint_func in enumerate(constraint_funcs):
            Y_cons_init_list[j][i, 0] = constraint_func(theta, obs_noise=False)

    optimizer.add_initial_data(X_init, Y_obj_init, Y_cons_init_list)

    is_feasible = torch.ones(n0, dtype=torch.bool, device=optimizer.device)
    for Y_cons in Y_cons_init_list:
        is_feasible &= Y_cons.squeeze() <= 0

    feasible_indices = torch.where(is_feasible)[0]
    if len(feasible_indices) > 0:
        feasible_obj = Y_obj_init[feasible_indices]
        best_idx = feasible_indices[torch.argmin(feasible_obj)]
    else:
        total_violation = torch.zeros(n0, dtype=optimizer.dtype, device=optimizer.device)
        for Y_cons in Y_cons_init_list:
            total_violation += torch.clamp(Y_cons.squeeze(), min=0)
        best_idx = torch.argmin(total_violation)

    return X_init[best_idx]


def run_lcbo_experiment(
    problem,
    optimizer,
    *,
    n_runs: int = 1,
    base_seed: int = 0,
    max_iterations: int = 20,
    n0: int = 5,
    log_path: str | None = None,
):
    """Run LCBO repeatedly and optionally save JSON logs."""

    iteration_logs = []
    best_overall_obj = float("inf")
    best_overall_theta = None
    optimizer_prototype = copy.deepcopy(optimizer)

    for run_idx in range(n_runs):
        run_optimizer = optimizer if run_idx == 0 else copy.deepcopy(optimizer_prototype)

        theta = generate_initial_data(
            optimizer=run_optimizer,
            objective_func=problem.objective,
            constraint_funcs=problem.constraints,
            bounds=problem.bounds,
            n0=n0,
            seed=base_seed + run_idx,
        )

        best_obj = float("inf")
        best_theta = None
        for iteration in range(max_iterations):
            start = time.time()
            theta, stop = run_optimizer.run_step(theta, iteration)
            elapsed = time.time() - start

            obj_val = problem.objective(theta, obs_noise=False).item()
            cons_vals = [constraint(theta, obs_noise=False).item() for constraint in problem.constraints]
            feasible = all(value <= 0 for value in cons_vals)

            if feasible and obj_val < best_obj:
                best_obj = obj_val
                best_theta = theta.clone()

            iteration_logs.append(
                {
                    "run": run_idx,
                    "iteration": iteration + 1,
                    "call_counter": run_optimizer.call_counter,
                    "iter_time": elapsed,
                    "obj_val": obj_val,
                    "cons_vals": cons_vals,
                    "feasible": feasible,
                }
            )

            if stop:
                break

        if best_theta is not None and best_obj < best_overall_obj:
            best_overall_obj = best_obj
            best_overall_theta = best_theta

    if log_path:
        log_dir = os.path.dirname(log_path)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        with open(log_path, "w", encoding="utf-8") as fp:
            json.dump(iteration_logs, fp, indent=2)

    return {
        "logs": iteration_logs,
        "best_objective": best_overall_obj,
        "best_theta": best_overall_theta,
    }


__all__ = ["generate_initial_data", "run_lcbo_experiment", "DEFAULT_DEVICE", "DEFAULT_DTYPE"]
