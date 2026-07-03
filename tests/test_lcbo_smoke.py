import pytest

torch = pytest.importorskip("torch")
gpytorch = pytest.importorskip("gpytorch")

from gpytorch.kernels import RBFKernel

from lcbo_opt import ExactGPModel, LCBO, generate_initial_data, make_benchmark_problem


def test_benchmark_factory_and_initial_data():
    problem = make_benchmark_problem(
        "within_model",
        dim=2,
        n_constraints=1,
        seed=0,
        noise_variance=1e-4,
        device=torch.device("cpu"),
    )
    optimizer = LCBO(
        problem.objective,
        problem.constraints,
        problem.dim,
        problem.bounds,
        max_objective_calls=20,
        max_active_points=1,
        N_max=5,
        device=torch.device("cpu"),
    )

    theta = generate_initial_data(
        optimizer,
        problem.objective,
        problem.constraints,
        problem.bounds,
        n0=3,
        seed=0,
    )

    assert theta.shape == (2,)
    assert optimizer.X_history.shape[0] == 3


def test_lcbo_one_step_smoke():
    problem = make_benchmark_problem(
        "within_model",
        dim=2,
        n_constraints=1,
        seed=1,
        noise_variance=1e-4,
        device=torch.device("cpu"),
    )
    optimizer = LCBO(
        problem.objective,
        problem.constraints,
        problem.dim,
        problem.bounds,
        max_objective_calls=30,
        n_repeats_gradient=1,
        max_active_points=1,
        N_max=5,
        device=torch.device("cpu"),
    )
    theta = generate_initial_data(
        optimizer,
        problem.objective,
        problem.constraints,
        problem.bounds,
        n0=3,
        seed=0,
    )
    next_theta, stop = optimizer.run_step(theta, 0)

    assert next_theta.shape == theta.shape
    assert stop is False


def test_exact_gp_kernel_factory_supports_ard():
    train_x = torch.rand(4, 3, dtype=torch.double)
    train_y = torch.rand(4, dtype=torch.double)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    default_model = ExactGPModel(train_x, train_y, likelihood)
    assert default_model.covar_module.base_kernel.ard_num_dims is None

    ard_model = ExactGPModel(
        train_x,
        train_y,
        gpytorch.likelihoods.GaussianLikelihood(),
        kernel_factory=lambda dim: RBFKernel(ard_num_dims=dim),
    )
    assert ard_model.covar_module.base_kernel.ard_num_dims == 3
