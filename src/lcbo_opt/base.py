"""Shared constrained Bayesian optimization state and model updates."""

from collections.abc import Callable

import gpytorch
import numpy as np
import torch
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

from .gp import ExactGPModel
from .scalers import MinMaxScaler, StandardScaler

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_DTYPE = torch.double


def build_prior_config_from_problem(problem, noise_variance: float) -> dict:
    """Build fixed GP hyperparameter config when benchmark metadata provides it."""

    gp_parameters = problem.metadata.get("gp_parameters")
    if not gp_parameters:
        return {}

    prior_config = {
        "obj": {
            "lengthscale": {"fixed": gp_parameters["objective"]["lengthscales"]},
            "outputscale": {"fixed": gp_parameters["objective"]["signal_variance"]},
            "noise": {"fixed": noise_variance},
        }
    }
    for item in gp_parameters["constraints"]:
        prior_config[f"cons_{item['id']}"] = {
            "lengthscale": {"fixed": item["lengthscales"]},
            "outputscale": {"fixed": item["signal_variance"]},
            "noise": {"fixed": noise_variance},
        }
    return prior_config


class ConstrainedBOBase:
    """Base class for constrained Bayesian optimizers with GP surrogates."""

    def __init__(
        self,
        obj_func,
        constraint_funcs,
        input_dim: int,
        bounds: torch.Tensor,
        max_objective_calls: int = 100,
        N_max: int = 50,
        prior_config: dict | None = None,
        set_hyper: bool = False,
        kernel_factory: Callable[[int], gpytorch.kernels.Kernel] | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype = DEFAULT_DTYPE,
    ):
        self.device = device or DEFAULT_DEVICE
        self.dtype = dtype
        self.obj_func = obj_func
        self.constraint_funcs = constraint_funcs
        self.dim = input_dim
        self.bounds = bounds.to(device=self.device, dtype=self.dtype)
        self.max_objective_calls = max_objective_calls
        self.call_counter = 0
        self.N_max = N_max
        self.prior_config = prior_config if prior_config else {}
        self.set_hyper = set_hyper
        self.kernel_factory = kernel_factory

        self.X_history = torch.empty(0, self.dim, dtype=self.dtype, device=self.device)
        self.Y_obj_history = torch.empty(0, 1, dtype=self.dtype, device=self.device)
        self.Y_cons_history = [
            torch.empty(0, 1, dtype=self.dtype, device=self.device)
            for _ in constraint_funcs
        ]

        self.scaler_x = MinMaxScaler(self.bounds)
        self.scaler_obj = StandardScaler()
        self.scalers_cons = [StandardScaler() for _ in constraint_funcs]

        self.model_obj = None
        self.models_cons = [None] * len(constraint_funcs)
        self.iteration = 0

    def add_initial_data(self, X_init, Y_obj_init, Y_cons_init_list) -> None:
        """Append pre-evaluated observations to the optimizer history."""

        X_init = X_init.to(device=self.device, dtype=self.dtype)
        Y_obj_init = Y_obj_init.to(device=self.device, dtype=self.dtype)
        Y_cons_init_list = [y.to(device=self.device, dtype=self.dtype) for y in Y_cons_init_list]

        self.X_history = torch.cat([self.X_history, X_init], dim=0)
        self.Y_obj_history = torch.cat([self.Y_obj_history, Y_obj_init], dim=0)
        for i, y_cons in enumerate(Y_cons_init_list):
            self.Y_cons_history[i] = torch.cat([self.Y_cons_history[i], y_cons], dim=0)
        self.call_counter += len(X_init)

    def random_initialize(self, n_points: int = 5) -> None:
        """Evaluate random points before starting model-based optimization."""

        for _ in range(n_points):
            rand_norm = torch.rand(1, self.dim, dtype=self.dtype, device=self.device)
            rand_raw = self.scaler_x.inverse_transform(rand_norm).view(-1)
            try:
                self.observe(rand_raw, repeats=1)
            except StopIteration:
                break
        self.update_models()

    def _update_single_gp(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        config_key: str = "obj",
        optimize_hyperparams: bool = False,
    ):
        likelihood = GaussianLikelihood().to(device=self.device, dtype=self.dtype)
        p_conf = self.prior_config.get(config_key, {})

        model = ExactGPModel(
            train_x,
            train_y,
            likelihood,
            lengthscale_prior=p_conf.get("lengthscale"),
            noise_prior=p_conf.get("noise"),
            outputscale_prior=p_conf.get("outputscale"),
            kernel_factory=self.kernel_factory,
        ).to(device=self.device, dtype=self.dtype)

        model.train()
        likelihood.train()

        if optimize_hyperparams:
            mll = ExactMarginalLogLikelihood(likelihood, model)
            optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
            for _ in range(50):
                optimizer.zero_grad()
                output = model(train_x)
                loss = -mll(output, train_y)
                loss.backward()
                optimizer.step()

        return model

    def update_models(self) -> None:
        """Refit objective and constraint GP models from recent observations."""

        if len(self.X_history) == 0:
            return

        indices = slice(-self.N_max, None) if len(self.X_history) > self.N_max else slice(None)
        X_active = self.X_history[indices]
        Y_obj_active = self.Y_obj_history[indices]
        Y_cons_active_list = [y[indices] for y in self.Y_cons_history]
        optimize_hypers = self.iteration > 0 and self.iteration % 5 == 0 and not self.set_hyper

        with torch.no_grad():
            self.scaler_obj.fit(Y_obj_active)
            for i, scaler in enumerate(self.scalers_cons):
                scaler.fit(Y_cons_active_list[i])

            train_x_norm = self.scaler_x.transform(X_active).detach()
            train_y_obj_norm = self.scaler_obj.transform(Y_obj_active).squeeze(-1).detach()
            train_y_cons_norm_list = [
                self.scalers_cons[i].transform(y_cons).squeeze(-1).detach()
                for i, y_cons in enumerate(Y_cons_active_list)
            ]

        self.model_obj = self._update_single_gp(
            train_x_norm,
            train_y_obj_norm,
            "obj",
            optimize_hyperparams=optimize_hypers,
        )
        for i, train_y_c_norm in enumerate(train_y_cons_norm_list):
            self.models_cons[i] = self._update_single_gp(
                train_x_norm,
                train_y_c_norm,
                f"cons_{i}",
                optimize_hyperparams=optimize_hypers,
            )

    def observe(self, theta: torch.Tensor, repeats: int = 1) -> float:
        """Evaluate the objective and constraints at one point."""

        if self.call_counter >= self.max_objective_calls:
            raise StopIteration("Max objective calls reached")

        theta = theta.to(device=self.device, dtype=self.dtype)
        y_obj_sum = 0.0
        y_cons_sums = [0.0] * len(self.constraint_funcs)

        for _ in range(repeats):
            y_obj_sum += self.obj_func(theta).item()
            self.call_counter += 1
            for i, constraint_func in enumerate(self.constraint_funcs):
                y_cons_sums[i] += constraint_func(theta).item()

        y_obj = y_obj_sum / repeats
        y_cons = [value / repeats for value in y_cons_sums]

        self.X_history = torch.cat([self.X_history, theta.view(1, -1)], dim=0)
        self.Y_obj_history = torch.cat(
            [self.Y_obj_history, torch.tensor([[y_obj]], dtype=self.dtype, device=self.device)],
            dim=0,
        )
        for i, value in enumerate(y_cons):
            self.Y_cons_history[i] = torch.cat(
                [
                    self.Y_cons_history[i],
                    torch.tensor([[value]], dtype=self.dtype, device=self.device),
                ],
                dim=0,
            )

        return y_obj
