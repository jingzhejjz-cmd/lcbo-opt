"""Tensor scalers used by LCBO models."""

import torch


class MinMaxScaler:
    """Scale box-constrained inputs to and from the unit hypercube."""

    def __init__(self, bounds: torch.Tensor):
        self.min_val = bounds[0]
        self.max_val = bounds[1]
        self.range_val = self.max_val - self.min_val
        one = torch.tensor(1.0, dtype=bounds.dtype, device=bounds.device)
        self.range_val = torch.where(self.range_val < 1e-6, one, self.range_val)

    def transform(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.min_val) / self.range_val

    def inverse_transform(self, x_norm: torch.Tensor) -> torch.Tensor:
        return x_norm * self.range_val + self.min_val


class StandardScaler:
    """Standardize scalar observations with a numerically safe variance."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data: torch.Tensor) -> None:
        self.mean = torch.mean(data, dim=0, keepdim=True)
        self.std = torch.std(data, dim=0, keepdim=True)
        one = torch.tensor(1.0, dtype=data.dtype, device=data.device)
        self.std = torch.where(self.std < 1e-6, one, self.std)

    def transform(self, data: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return data
        return (data - self.mean) / self.std

    def inverse_transform(self, data_norm: torch.Tensor) -> torch.Tensor:
        if self.mean is None:
            return data_norm
        original_shape = data_norm.shape
        if data_norm.dim() == 1:
            data_norm = data_norm.unsqueeze(-1)
        result = data_norm * self.std + self.mean
        if len(original_shape) == 1:
            result = result.squeeze(-1)
        return result
