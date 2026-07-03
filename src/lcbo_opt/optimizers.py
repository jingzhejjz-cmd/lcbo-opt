"""LCBO optimizer implementation."""

from collections.abc import Callable

import gpytorch
import numpy as np
import torch
from gpytorch.kernels import RBFKernel, ScaleKernel

from .base import (
    ConstrainedBOBase,
    DEFAULT_DEVICE,
    DEFAULT_DTYPE,
    build_prior_config_from_problem,
)


class LCBO(ConstrainedBOBase):
    """Local constrained Bayesian optimizer."""

    def __init__(
        self,
        obj_func=None,
        constraint_funcs=None,
        input_dim: int | None = None,
        bounds: torch.Tensor | None = None,
        max_objective_calls: int = 100,
        n_repeats_gradient=3,
        max_active_points=3,
        N_max: int = 50,
        step_size_fn=lambda k: 0.05,
        rho_fn=lambda k: np.sqrt(k + 1),
        temp_lse: float = 1.0,
        prior_config: dict | None = None,
        set_hyper: bool = False,
        grad_normalize: bool = True,
        delta_norm: float | None = 0.1,
        kernel_factory: Callable[[int], gpytorch.kernels.Kernel] | None = None,
        problem=None,
        pull_gp_param_from_problem: bool = False,
        noise_variance: float = 1e-4,
        device: torch.device | None = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        if problem is not None:
            obj_func = problem.objective if obj_func is None else obj_func
            constraint_funcs = problem.constraints if constraint_funcs is None else constraint_funcs
            input_dim = problem.dim if input_dim is None else input_dim
            bounds = problem.bounds if bounds is None else bounds
            if pull_gp_param_from_problem and prior_config is None:
                prior_config = build_prior_config_from_problem(problem, noise_variance)
        elif pull_gp_param_from_problem:
            raise ValueError("pull_gp_param_from_problem=True requires a problem.")

        if obj_func is None:
            raise ValueError("obj_func is required when problem is not provided.")
        if constraint_funcs is None:
            raise ValueError("constraint_funcs is required when problem is not provided.")
        if input_dim is None:
            raise ValueError("input_dim is required when problem is not provided.")
        if bounds is None:
            raise ValueError("bounds is required when problem is not provided.")

        super().__init__(
            obj_func=obj_func,
            constraint_funcs=constraint_funcs,
            input_dim=input_dim,
            bounds=bounds,
            max_objective_calls=max_objective_calls,
            N_max=N_max,
            prior_config=prior_config,
            set_hyper=set_hyper,
            kernel_factory=kernel_factory,
            device=device,
            dtype=dtype,
        )
        self.n_repeats_gradient_fn = (
            n_repeats_gradient
            if callable(n_repeats_gradient)
            else (lambda k, v=n_repeats_gradient: v)
        )
        self.max_active_points_fn = (
            max_active_points
            if callable(max_active_points)
            else (lambda k, v=max_active_points: v)
        )
        self.n_repeats_gradient = int(max(1, self.n_repeats_gradient_fn(0)))
        self.max_active_points = int(max(0, self.max_active_points_fn(0)))
        self.step_size_fn = step_size_fn
        self.rho_fn = rho_fn
        self.temp = temp_lse
        self.grad_normalize = grad_normalize
        self.delta_norm = delta_norm

    def _get_trace_term(self, model, theta_norm, candidate_norm):
        X_train = model.train_inputs[0]
        X_hat = torch.cat([X_train, candidate_norm.view(1, -1)], dim=0)

        sigma_n = model.likelihood.noise

        K_hat = model.covar_module(X_hat).evaluate()
        K_hat_noise = K_hat + torch.eye(X_hat.shape[0], dtype=self.dtype, device=self.device) * sigma_n

        grad_k = self._kernel_gradient_matrix(model, theta_norm, X_hat)

        try:
            L = torch.linalg.cholesky(K_hat_noise)
            alpha = torch.linalg.solve_triangular(L, grad_k, upper=False)
            trace_val = torch.sum(alpha**2)
        except RuntimeError:
            temp = torch.linalg.solve(K_hat_noise, grad_k)
            trace_val = torch.sum(temp * grad_k)

        return trace_val

    def _kernel_gradient_matrix(self, model, theta_norm, X_hat):
        """Return gradients of k(theta, X_hat) with respect to theta."""

        covar_module = model.covar_module
        base_kernel = covar_module.base_kernel if isinstance(covar_module, ScaleKernel) else covar_module

        if isinstance(base_kernel, RBFKernel):
            ls = base_kernel.lengthscale
            sigma_f = covar_module.outputscale if isinstance(covar_module, ScaleKernel) else 1.0
            diff = theta_norm.view(1, -1) - X_hat
            scaled_diff = diff / ls
            dist_sq = torch.sum(scaled_diff**2, dim=-1)
            k_val = sigma_f * torch.exp(-0.5 * dist_sq)
            return -(diff / (ls**2)) * k_val.unsqueeze(-1)

        with torch.enable_grad():
            theta_var = theta_norm.view(1, -1).detach().clone().requires_grad_(True)
            cross_cov = covar_module(theta_var, X_hat).evaluate().view(-1)
            return torch.stack(
                [
                    torch.autograd.grad(
                        cross_cov[i],
                        theta_var,
                        retain_graph=True,
                        create_graph=True,
                    )[0].view(-1)
                    for i in range(cross_cov.numel())
                ],
                dim=0,
            )

    def _get_prior_trace(self, model):
        """Compute the prior gradient trace for the model kernel."""

        lengthscale_module = getattr(model.covar_module, "base_kernel", model.covar_module)
        ls = lengthscale_module.lengthscale
        sigma_f = model.covar_module.outputscale
        return torch.sum((sigma_f / ls) ** 2)

    def acquisition_function(self, candidate_norm, theta_norm):
        """Score active samples by smoothed posterior trace reduction."""

        vals = []
        for model in [self.model_obj] + self.models_cons:
            reduction_trace = self._get_trace_term(model, theta_norm, candidate_norm)
            vals.append(-reduction_trace)

        vals = torch.stack(vals)
        max_val = torch.max(vals)
        return max_val + self.temp * torch.log(torch.sum(torch.exp((vals - max_val) / self.temp)))

    def optimize_acquisition(
        self,
        current_theta_norm,
        delta_norm: float | None = 0.1,
        n_steps: int = 50,
        lr: float = 0.01,
        n_restarts: int = 5,
    ):
        """Optimize the active-sampling acquisition in normalized coordinates."""

        if delta_norm is None:
            lower_bounds = torch.zeros_like(current_theta_norm)
            upper_bounds = torch.ones_like(current_theta_norm)
        else:
            lower_bounds = torch.clamp(current_theta_norm - delta_norm, 0.0, 1.0)
            upper_bounds = torch.clamp(current_theta_norm + delta_norm, 0.0, 1.0)

        candidates_list = [current_theta_norm.view(1, -1)]
        if n_restarts > 1:
            rand_samples = torch.rand(n_restarts - 1, self.dim, dtype=self.dtype, device=self.device)
            candidates_list.append(lower_bounds + (upper_bounds - lower_bounds) * rand_samples)

        candidates = torch.cat(candidates_list, dim=0).detach().requires_grad_(True)
        optimizer = torch.optim.Adam([candidates], lr=lr)
        curr_norm_detached = current_theta_norm.detach()

        for _ in range(n_steps):
            optimizer.zero_grad()
            loss = torch.sum(
                torch.stack(
                    [
                        self.acquisition_function(candidates[i], curr_norm_detached)
                        for i in range(candidates.shape[0])
                    ]
                )
            )
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                candidates.data = torch.max(torch.min(candidates.data, upper_bounds), lower_bounds)

        final_vals = torch.stack(
            [
                self.acquisition_function(candidates[i], curr_norm_detached)
                for i in range(candidates.shape[0])
            ]
        )
        return candidates[torch.argmin(final_vals)].detach()

    def _get_mean_and_grad_normalized(self, model, theta_norm_tensor):
        model.eval()
        theta_norm_tensor = theta_norm_tensor.detach().clone().requires_grad_(True)
        output = model(theta_norm_tensor)
        mu_norm = output.mean.sum()
        grad_norm = torch.autograd.grad(mu_norm, theta_norm_tensor)[0]
        model.train()
        return mu_norm, grad_norm.squeeze()

    def run_step(self, current_theta, iteration: int):
        """Run one LCBO iteration and return the next iterate."""

        self.iteration = iteration
        self.n_repeats_gradient = int(max(1, self.n_repeats_gradient_fn(iteration)))
        self.max_active_points = int(max(0, self.max_active_points_fn(iteration)))

        try:
            self.observe(current_theta, repeats=self.n_repeats_gradient)
        except StopIteration:
            return current_theta, True

        self.update_models()

        for _ in range(self.max_active_points):
            curr_norm = self.scaler_x.transform(current_theta.view(1, -1)).view(-1)
            cand_norm = self.optimize_acquisition(
                curr_norm,
                delta_norm=self.delta_norm,
                n_steps=50,
                lr=0.05,
            )
            cand_raw = self.scaler_x.inverse_transform(cand_norm.view(1, -1)).view(-1)
            try:
                self.observe(cand_raw, repeats=1)
            except StopIteration:
                pass
            self.update_models()

        curr_norm = self.scaler_x.transform(current_theta.view(1, -1)).detach()
        _, grad_obj_norm = self._get_mean_and_grad_normalized(self.model_obj, curr_norm)

        grad_penalty_sum_norm = torch.zeros_like(grad_obj_norm)
        for i, model_c in enumerate(self.models_cons):
            mu_c_norm, grad_c_norm = self._get_mean_and_grad_normalized(model_c, curr_norm)
            scaler_c = self.scalers_cons[i]
            threshold_shift = scaler_c.mean.item() / scaler_c.std.item()
            violation_term = mu_c_norm + threshold_shift
            if violation_term > 0:
                grad_penalty_sum_norm += violation_term * grad_c_norm

        total_grad_norm = grad_obj_norm + self.rho_fn(iteration) * grad_penalty_sum_norm
        if self.grad_normalize:
            grad_norm_value = torch.norm(total_grad_norm)
            update_norm = total_grad_norm / grad_norm_value if grad_norm_value > 1e-6 else torch.zeros_like(total_grad_norm)
        else:
            update_norm = total_grad_norm

        new_theta_norm = curr_norm - self.step_size_fn(iteration) * update_norm
        new_theta_norm = torch.clamp(new_theta_norm, 0.0, 1.0)
        new_theta = self.scaler_x.inverse_transform(new_theta_norm).view(-1)

        return new_theta, False
