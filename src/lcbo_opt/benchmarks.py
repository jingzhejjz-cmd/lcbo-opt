"""Benchmark problem definitions and a unified benchmark factory."""

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch
from scipy.optimize import minimize

from .base import DEFAULT_DEVICE, DEFAULT_DTYPE


@dataclass
class BenchmarkProblem:
    """Callable constrained benchmark packaged for LCBO experiments."""

    name: str
    dim: int
    bounds: torch.Tensor
    objective: Callable
    constraints: list[Callable]
    metadata: dict = field(default_factory=dict)


class GPSamplerRFF:
    """Approximate a GP prior sample with random Fourier features."""

    def __init__(self, dim, lengthscale, signal_var=1.0, num_features=2000, seed=None):
        self.rng = np.random.RandomState(seed)
        self.dim = dim
        self.variance = signal_var
        self.lengthscale = np.array(lengthscale)
        self.num_features = num_features
        self.w = self.rng.normal(size=(num_features, dim)) / self.lengthscale.reshape(1, -1)
        self.b = self.rng.uniform(0, 2 * np.pi, size=num_features)
        self.feat_weights = self.rng.normal(size=num_features)
        self.scale = np.sqrt(self.variance) * np.sqrt(2.0 / num_features)

    def __call__(self, x):
        x = np.atleast_2d(x)
        features = np.cos(np.dot(x, self.w.T) + self.b)
        y = np.dot(features, self.feat_weights) * self.scale
        return y


class ConstrainedWithinModelBenchmark:
    """Synthetic constrained objective sampled from GP priors."""

    def __init__(self, dim, n_constraints=1, seed=0, offset=0.0):
        self.dim = dim
        self.n_constraints = n_constraints
        self.seed = seed
        self.offset = offset
        self.rng = np.random.RandomState(seed)
        self.ls_target, self.ls_constraints = self._generate_lengthscales()
        self.objective_func = GPSamplerRFF(
            dim=dim,
            lengthscale=self.ls_target,
            signal_var=1.0,
            seed=self.rng.randint(0, 10000),
        )
        self.constraint_funcs = [
            GPSamplerRFF(
                dim=dim,
                lengthscale=self.ls_constraints[i],
                signal_var=1.0,
                seed=self.rng.randint(0, 10000),
            )
            for i in range(n_constraints)
        ]

    @property
    def bounds(self):
        return np.column_stack((np.zeros(self.dim), np.ones(self.dim)))

    def _generate_lengthscales(self):
        d = self.dim
        term_inner = 1 + 2 * np.sqrt(1 - 3 / (5 * d)) if d >= 1 else 1
        term_outer = np.sqrt(1 / 3 + term_inner)
        delta_upper_bound = np.sqrt(d / 6) * term_outer
        delta_sub = delta_upper_bound * 0.2
        gamma = 0.3
        low = 2 * delta_sub * (1 - gamma)
        high = 2 * delta_sub * (1 + gamma)
        return self.rng.uniform(low, high), [
            self.rng.uniform(low, high) for _ in range(self.n_constraints)
        ]

    def objective(self, x):
        val = self.objective_func(x)
        return float(val) if np.ndim(val) == 0 else val.flatten()

    def constraints(self, x):
        vals = [constraint(x) for constraint in self.constraint_funcs]
        return np.array(vals).flatten() + self.offset

    def get_gp_parameters(self):
        return {
            "objective": {
                "signal_variance": 1.0,
                "lengthscales": self.ls_target,
            },
            "constraints": [
                {
                    "id": i,
                    "signal_variance": 1.0,
                    "lengthscales": ls,
                }
                for i, ls in enumerate(self.ls_constraints)
            ],
        }

    def solve_ground_truth(self, n_restarts=10, method="SLSQP"):
        n_samples = 1000
        X_sample = self.rng.uniform(0, 1, size=(n_samples, self.dim))
        Y_sample = self.objective_func(X_sample)
        C_sample = np.vstack(
            [constraint(X_sample) + self.offset for constraint in self.constraint_funcs]
        ).T

        feasible_mask = np.all(C_sample <= 0, axis=1)
        if not np.any(feasible_mask):
            violation = np.sum(np.maximum(0, C_sample), axis=1)
            start_X = X_sample[np.argsort(violation)[:n_restarts]]
        else:
            feasible_X = X_sample[feasible_mask]
            feasible_Y = Y_sample[feasible_mask]
            start_X = feasible_X[np.argsort(feasible_Y)[:n_restarts]]

        best_x = None
        best_fun = np.inf
        bounds = [(0.0, 1.0) for _ in range(self.dim)]
        constraints = [
            {"type": "ineq", "fun": lambda x, i=i: -self.constraints(x)[i]}
            for i in range(self.n_constraints)
        ]

        for x0 in start_X:
            res = minimize(
                fun=lambda x: self.objective(x),
                x0=x0,
                bounds=bounds,
                constraints=constraints,
                method=method,
                options={"ftol": 1e-6, "disp": False},
            )
            if np.all(self.constraints(res.x) <= 1e-5) and res.fun < best_fun:
                best_fun = res.fun
                best_x = res.x

        return best_x, best_fun


class Truss25Benchmark:
    """Scaled 25-bar truss design benchmark with stress and displacement constraints."""

    def __init__(self, smooth_factor=10.0):
        self.smooth_factor = smooth_factor
        self.E = 1e7
        self.rho = 0.1
        self.stress_lim = 40000.0
        self.disp_lim = 0.35
        self.obj_scale = 1.0 / 100.0
        self.nodes = np.array(
            [
                [-37.5, 0.0, 200.0],
                [37.5, 0.0, 200.0],
                [-37.5, 37.5, 100.0],
                [37.5, 37.5, 100.0],
                [37.5, -37.5, 100.0],
                [-37.5, -37.5, 100.0],
                [-100.0, 100.0, 0.0],
                [100.0, 100.0, 0.0],
                [100.0, -100.0, 0.0],
                [-100.0, -100.0, 0.0],
            ]
        )
        self.connectivity = np.array(
            [
                [0, 1],
                [0, 3],
                [0, 5],
                [1, 2],
                [1, 4],
                [1, 5],
                [0, 2],
                [0, 4],
                [1, 3],
                [2, 3],
                [2, 5],
                [3, 4],
                [4, 5],
                [2, 6],
                [2, 7],
                [3, 7],
                [3, 8],
                [4, 8],
                [4, 9],
                [5, 9],
                [5, 6],
                [2, 9],
                [3, 6],
                [4, 7],
                [5, 8],
            ]
        )
        self.num_nodes = len(self.nodes)
        self.num_bars = len(self.connectivity)
        self.fixed_nodes = [6, 7, 8, 9]
        self.dof_map = np.arange(self.num_nodes * 3).reshape(-1, 3)
        self.fixed_dofs = self.dof_map[self.fixed_nodes, :].flatten()
        self.free_dofs = np.delete(np.arange(self.num_nodes * 3), self.fixed_dofs)
        self.forces = np.zeros(self.num_nodes * 3)
        self.forces[0] = 1000.0
        self.forces[1] = 10000.0
        self.forces[2] = -5000.0
        self.lengths = np.linalg.norm(
            self.nodes[self.connectivity[:, 1]] - self.nodes[self.connectivity[:, 0]],
            axis=1,
        )

    @property
    def bounds(self):
        return np.column_stack((0.01 * np.ones(self.num_bars), 5.0 * np.ones(self.num_bars)))

    @property
    def dim(self):
        return self.num_bars

    def _logsumexp_smooth_max(self, values):
        max_val = np.max(values)
        alpha = self.smooth_factor
        return max_val + (1.0 / alpha) * np.log(np.sum(np.exp(alpha * (values - max_val))))

    def _normalize_design_vector(self, x):
        x_vec = np.asarray(x, dtype=float).reshape(-1)
        if x_vec.size != self.num_bars:
            raise ValueError(f"Expected {self.num_bars} design variables, got {x_vec.size}.")
        return x_vec

    def _solve_fea(self, areas):
        areas = self._normalize_design_vector(areas)
        num_dof = self.num_nodes * 3
        K_global = np.zeros((num_dof, num_dof))

        for i in range(self.num_bars):
            n1, n2 = self.connectivity[i]
            length = self.lengths[i]
            direction = (self.nodes[n2] - self.nodes[n1]) / length
            k_sub = (self.E * areas[i] / length) * np.outer(direction, direction)
            dofs_n1 = self.dof_map[n1]
            dofs_n2 = self.dof_map[n2]
            for r in range(3):
                for c in range(3):
                    K_global[dofs_n1[r], dofs_n1[c]] += k_sub[r, c]
                    K_global[dofs_n2[r], dofs_n2[c]] += k_sub[r, c]
                    K_global[dofs_n1[r], dofs_n2[c]] -= k_sub[r, c]
                    K_global[dofs_n2[r], dofs_n1[c]] -= k_sub[r, c]

        try:
            u_free = np.linalg.solve(
                K_global[np.ix_(self.free_dofs, self.free_dofs)],
                self.forces[self.free_dofs],
            )
        except np.linalg.LinAlgError:
            return None, None

        u_full = np.zeros(num_dof)
        u_full[self.free_dofs] = u_free
        stresses = np.zeros(self.num_bars)
        for i in range(self.num_bars):
            n1, n2 = self.connectivity[i]
            length = self.lengths[i]
            direction = (self.nodes[n2] - self.nodes[n1]) / length
            delta_L = np.dot(u_full[self.dof_map[n2]] - u_full[self.dof_map[n1]], direction)
            stresses[i] = self.E * delta_L / length

        return u_full, stresses

    def objective(self, x):
        x = self._normalize_design_vector(x)
        return np.dot(self.lengths, x) * self.rho * self.obj_scale

    def constraints(self, x):
        x = self._normalize_design_vector(x)
        u_full, stresses = self._solve_fea(x)
        if u_full is None:
            return np.array([100.0, 100.0])
        stress_constraint = self._logsumexp_smooth_max(np.abs(stresses) / self.stress_lim - 1.0)
        disp_constraint = self._logsumexp_smooth_max(np.abs(u_full[self.free_dofs]) / self.disp_lim - 1.0)
        return np.array([stress_constraint, disp_constraint])


class SteppedCantileverBenchmark:
    """Stepped cantilever beam volume minimization benchmark."""

    def __init__(self, n_segments=25):
        self.n_segments = n_segments
        self.dim = self.n_segments * 2
        self.L_total = 100.0
        self.E = 2.9e7
        self.P = 500.0
        self.l_seg = self.L_total / self.n_segments
        self.stress_limit = 40000.0
        self.disp_limit = 2.5
        self.obj_scale = 1.0 / 100.0
        self.lse_alpha = 20.0

    @property
    def bounds(self):
        return np.column_stack((0.5 * np.ones(self.dim), 5.0 * np.ones(self.dim)))

    def _calculate_physics(self, x):
        vars_reshaped = np.asarray(x, dtype=float).reshape(self.n_segments, 2)
        width = vars_reshaped[:, 0]
        height = vars_reshaped[:, 1]
        areas = width * height
        inertias = width * (height**3) / 12.0
        y_max = height / 2.0
        volume = np.sum(areas * self.l_seg)

        x_lefts = np.arange(self.n_segments) * self.l_seg
        moments = self.P * (self.L_total - x_lefts)
        stresses = moments * y_max / inertias

        def integ_moment_sq(x_pos):
            return -self.P * (self.L_total - x_pos) ** 3 / 3.0

        disp = 0.0
        for i in range(self.n_segments):
            x_a = i * self.l_seg
            x_b = (i + 1) * self.l_seg
            disp += -(integ_moment_sq(x_b) - integ_moment_sq(x_a)) / (self.E * inertias[i])

        return volume, stresses, disp

    def objective(self, x):
        volume, _, _ = self._calculate_physics(x)
        return volume * self.obj_scale

    def constraints(self, x):
        _, stresses, disp = self._calculate_physics(x)
        g_stress_raw = stresses / self.stress_limit - 1.0
        max_val = np.max(g_stress_raw)
        smooth_stress = max_val + (1.0 / self.lse_alpha) * np.log(
            np.sum(np.exp(self.lse_alpha * (g_stress_raw - max_val)))
        )
        return np.array([smooth_stress, disp / self.disp_limit - 1.0])


class HalfCheetahBenchmark:
    """HalfCheetah policy-search benchmark loaded only when RL extras are installed."""

    def __init__(self, seed=0, horizon=1000, n_repeats=5):
        try:
            import gymnasium as gym
        except ImportError as exc:
            raise ImportError(
                "HalfCheetahBenchmark requires the optional 'rl' dependencies: "
                "pip install lcbo-opt[rl]"
            ) from exc

        self.gym = gym
        self.env_name = "HalfCheetah-v4"
        self.master_seed = seed
        self.horizon = horizon
        self.n_repeats = n_repeats
        self.env = gym.make(self.env_name)
        self.obs_dim = self.env.observation_space.shape[0]
        self.act_dim = self.env.action_space.shape[0]
        self.dim = self.obs_dim * self.act_dim
        self.ctrl_cost_limit = 1500.0
        self.max_step_cost = 6.0
        self.ctrl_cost_weight = 0.1
        self.obs_mean, self.obs_std = self._compute_static_normalization()
        self._cache = {}

    def __del__(self):
        if hasattr(self, "env"):
            self.env.close()

    @property
    def bounds(self):
        limit = 0.2
        return np.column_stack((-limit * np.ones(self.dim), limit * np.ones(self.dim)))

    def _compute_static_normalization(self):
        temp_env = self.gym.make(self.env_name)
        obs, _ = temp_env.reset(seed=42)
        obs_buffer = []
        for _ in range(2000):
            obs_buffer.append(obs)
            obs, _, terminated, truncated, _ = temp_env.step(temp_env.action_space.sample())
            if terminated or truncated:
                obs, _ = temp_env.reset()
        temp_env.close()
        obs_matrix = np.array(obs_buffer)
        mean = np.mean(obs_matrix, axis=0)
        std = np.std(obs_matrix, axis=0)
        std[std < 1e-6] = 1.0
        return mean, std

    def _run_episode(self, x):
        x = np.asarray(x, dtype=np.float64)
        key = np.round(x, 5).tobytes()
        if key in self._cache:
            return self._cache[key]

        W = x.reshape(self.act_dim, self.obs_dim)
        sum_obj = 0.0
        sum_constraint = 0.0
        for i in range(self.n_repeats):
            obs, _ = self.env.reset(seed=self.master_seed + i * 100)
            velocity_reward = 0.0
            ctrl_cost_total = 0.0
            steps = 0
            for _ in range(self.horizon):
                norm_obs = (obs - self.obs_mean) / self.obs_std
                action = np.tanh(np.dot(W, norm_obs))
                obs, reward, terminated, truncated, _ = self.env.step(action)
                ctrl_cost = np.sum(np.square(action))
                ctrl_cost_total += ctrl_cost
                velocity_reward += reward + self.ctrl_cost_weight * ctrl_cost
                steps += 1
                if terminated or truncated:
                    break
            if steps < self.horizon:
                ctrl_cost_total += (self.horizon - steps) * self.max_step_cost
            sum_obj += -(velocity_reward / self.horizon)
            sum_constraint += ctrl_cost_total / self.ctrl_cost_limit - 1.0

        result = (sum_obj / self.n_repeats, np.array([sum_constraint / self.n_repeats]))
        self._cache[key] = result
        return result

    def objective(self, x):
        obj, _ = self._run_episode(x)
        return obj

    def constraints(self, x):
        _, constraints = self._run_episode(x)
        return constraints


def _as_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _wrap_problem(name, task, noise_variance=1e-4, device=None, dtype=DEFAULT_DTYPE, metadata=None):
    device = device or DEFAULT_DEVICE
    bounds = torch.tensor(task.bounds.T, dtype=dtype, device=device)
    noise_std = float(np.sqrt(noise_variance))

    def objective_func(x, obs_noise=True):
        x_np = np.asarray(_as_numpy(x), dtype=float).reshape(-1)
        result = torch.tensor(task.objective(x_np), dtype=dtype, device=device)
        if obs_noise:
            result = result + torch.randn_like(result) * noise_std
        return result

    constraints = []
    n_constraints = len(np.asarray(task.constraints(np.mean(task.bounds, axis=1))).reshape(-1))
    for idx in range(n_constraints):
        def make_constraint(constraint_idx):
            def constraint_func(x, obs_noise=True):
                x_np = np.asarray(_as_numpy(x), dtype=float).reshape(-1)
                val = np.asarray(task.constraints(x_np)).reshape(-1)[constraint_idx]
                result = torch.tensor(val, dtype=dtype, device=device)
                if obs_noise:
                    result = result + torch.randn_like(result) * noise_std
                return result

            return constraint_func

        constraints.append(make_constraint(idx))

    return BenchmarkProblem(
        name=name,
        dim=task.dim,
        bounds=bounds,
        objective=objective_func,
        constraints=constraints,
        metadata=metadata or {},
    )


def make_benchmark_problem(name: str, *, device=None, dtype=DEFAULT_DTYPE, **kwargs) -> BenchmarkProblem:
    """Create a benchmark with a unified callable interface."""

    normalized_name = name.lower().replace("-", "_")
    noise_variance = kwargs.pop("noise_variance", 1e-4)

    if normalized_name in {"within_model", "synthetic"}:
        compute_ground_truth = kwargs.pop("compute_ground_truth", False)
        task = ConstrainedWithinModelBenchmark(
            dim=kwargs.pop("dim", 2),
            n_constraints=kwargs.pop("n_constraints", 1),
            seed=kwargs.pop("seed", 0),
            offset=kwargs.pop("offset", kwargs.pop("off_set", 0.0)),
        )
        metadata = {"gp_parameters": task.get_gp_parameters()}
        if compute_ground_truth:
            metadata["ground_truth"] = task.solve_ground_truth()
        return _wrap_problem(
            "within_model",
            task,
            noise_variance=noise_variance,
            device=device,
            dtype=dtype,
            metadata=metadata,
        )

    if normalized_name == "truss25":
        task = Truss25Benchmark(smooth_factor=kwargs.pop("smooth_factor", 10.0))
        return _wrap_problem("truss25", task, noise_variance=noise_variance, device=device, dtype=dtype)

    if normalized_name in {"cantilever", "stepped_cantilever"}:
        task = SteppedCantileverBenchmark(n_segments=kwargs.pop("n_segments", 25))
        return _wrap_problem("cantilever", task, noise_variance=noise_variance, device=device, dtype=dtype)

    if normalized_name == "halfcheetah":
        task = HalfCheetahBenchmark(
            seed=kwargs.pop("seed", 0),
            horizon=kwargs.pop("horizon", 1000),
            n_repeats=kwargs.pop("n_repeats", 5),
        )
        return _wrap_problem("halfcheetah", task, noise_variance=noise_variance, device=device, dtype=dtype)

    raise ValueError(f"Unknown benchmark problem: {name}")
