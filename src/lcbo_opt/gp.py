"""Exact Gaussian process model used by LCBO."""

from collections.abc import Callable

import gpytorch
import torch
from gpytorch.constraints import Interval
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel


def default_kernel_factory(input_dim: int):
    """Return the default isotropic RBF kernel."""

    return RBFKernel(ard_num_dims=None)


class ExactGPModel(gpytorch.models.ExactGP):
    """Exact GP with configurable kernel and optional fixed/prior hyperparameters."""

    def __init__(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        likelihood,
        lengthscale_prior: dict | None = None,
        noise_prior: dict | None = None,
        outputscale_prior: dict | None = None,
        kernel_factory: Callable[[int], gpytorch.kernels.Kernel] | None = None,
    ):
        super().__init__(train_x, train_y, likelihood)
        input_dim = train_x.shape[-1]
        kernel = (kernel_factory or default_kernel_factory)(input_dim)

        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel if isinstance(kernel, ScaleKernel) else ScaleKernel(kernel)

        self._configure_lengthscale(lengthscale_prior)
        self._configure_outputscale(outputscale_prior)
        self._configure_noise(likelihood, noise_prior)

    @property
    def lengthscale_module(self):
        """Return the kernel module that owns the lengthscale parameter."""

        return getattr(self.covar_module, "base_kernel", self.covar_module)

    def _configure_lengthscale(self, config: dict | None) -> None:
        if not config or not hasattr(self.lengthscale_module, "lengthscale"):
            return

        module = self.lengthscale_module
        if "fixed" in config:
            module.lengthscale = config["fixed"]
            module.raw_lengthscale.requires_grad_(False)
            return

        if "constraint" in config:
            module.lengthscale_constraint = config["constraint"]
            if isinstance(config["constraint"], Interval):
                lb = config["constraint"].lower_bound
                ub = config["constraint"].upper_bound
                module.lengthscale = (lb * ub) ** 0.5

        if "prior" in config:
            module.register_prior(
                "lengthscale_prior",
                config["prior"],
                lambda m: m.lengthscale,
                lambda m, v: m._set_lengthscale(v),
            )

        if "base_init" in config:
            module.lengthscale = float(config["base_init"])

    def _configure_outputscale(self, config: dict | None) -> None:
        if not config or not hasattr(self.covar_module, "outputscale"):
            return

        if "fixed" in config:
            self.covar_module.outputscale = float(config["fixed"])
            self.covar_module.raw_outputscale.requires_grad_(False)
            return

        if "constraint" in config:
            self.covar_module.outputscale_constraint = config["constraint"]
        if "prior" in config:
            self.covar_module.register_prior(
                "outputscale_prior",
                config["prior"],
                lambda m: m.outputscale,
                lambda m, v: m._set_outputscale(v),
            )

    def _configure_noise(self, likelihood, config: dict | None) -> None:
        if not config:
            return

        if "fixed" in config:
            likelihood.noise = float(config["fixed"])
            likelihood.noise_covar.raw_noise.requires_grad_(False)
            return

        if "constraint" in config:
            likelihood.noise_constraint = config["constraint"]
        if "prior" in config:
            likelihood.register_prior(
                "noise_prior",
                config["prior"],
                lambda m: m.noise,
                lambda m, v: m._set_noise(v),
            )

    def forward(self, x: torch.Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
