"""Ordinary residual MLP used only for POD coefficient regression."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class ResidualMLPConfig:
    input_dimension: int
    output_dimension: int
    width: int = 256
    blocks: int = 4
    activation: str = "silu"

    def __post_init__(self) -> None:
        if int(self.input_dimension) < 1 or int(self.output_dimension) < 1:
            raise ValueError("network input/output dimensions must be positive")
        if int(self.width) < 4 or int(self.blocks) < 1:
            raise ValueError("network width/blocks are too small")
        if self.activation not in {"silu", "softplus", "tanh"}:
            raise ValueError("unsupported activation")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=float).reshape(-1)
        scale = np.asarray(self.scale, dtype=float).reshape(-1)
        if mean.shape != scale.shape or mean.size < 1:
            raise ValueError("normalizer dimensions do not match")
        if np.any(~np.isfinite(mean)) or np.any(~np.isfinite(scale)) or np.any(scale <= 0.0):
            raise ValueError("normalizer mean/scale must be finite with positive scale")
        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "scale", scale)

    @classmethod
    def fit(cls, inputs: np.ndarray, minimum_scale: float = 1e-12) -> "FeatureNormalizer":
        value = np.asarray(inputs, dtype=float)
        if value.ndim != 2 or value.shape[0] < 1 or np.any(~np.isfinite(value)):
            raise ValueError("normalizer inputs must be a finite matrix")
        mean = np.mean(value, axis=0)
        scale = np.std(value, axis=0)
        scale = np.maximum(scale, float(minimum_scale))
        return cls(mean, scale)

    def transform(self, inputs: np.ndarray) -> np.ndarray:
        value = np.asarray(inputs, dtype=float)
        if value.shape[-1] != self.mean.size:
            raise ValueError("normalizer input width mismatch")
        return (value - self.mean) / self.scale

    def inverse(self, normalized: np.ndarray) -> np.ndarray:
        value = np.asarray(normalized, dtype=float)
        if value.shape[-1] != self.mean.size:
            raise ValueError("normalizer width mismatch")
        return self.mean + self.scale * value


def _torch_activation(name: str):
    import torch.nn as nn

    if name == "silu":
        return nn.SiLU
    if name == "softplus":
        return nn.Softplus
    if name == "tanh":
        return nn.Tanh
    raise ValueError("unsupported activation")


def build_residual_mlp(config: ResidualMLPConfig, normalizer: FeatureNormalizer):
    """Build the PyTorch network lazily so core SDFMPNEO does not require torch."""
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("install sdfmpneo[neural] to use the neural electrothermal ROM") from exc

    if normalizer.mean.size != int(config.input_dimension):
        raise ValueError("normalizer and network input dimensions do not match")
    activation = _torch_activation(config.activation)

    class ResidualBlock(nn.Module):
        def __init__(self, width: int):
            super().__init__()
            self.first = nn.Linear(width, width)
            self.second = nn.Linear(width, width)
            self.act1 = activation()
            self.act2 = activation()

        def forward(self, x):
            return x + self.second(self.act2(self.first(self.act1(x))))

    class ResidualMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.register_buffer("input_mean", torch.as_tensor(normalizer.mean, dtype=torch.float64))
            self.register_buffer("input_scale", torch.as_tensor(normalizer.scale, dtype=torch.float64))
            self.input = nn.Linear(config.input_dimension, config.width)
            self.activation = activation()
            self.blocks = nn.ModuleList([ResidualBlock(config.width) for _ in range(config.blocks)])
            self.output = nn.Linear(config.width, config.output_dimension)
            self.double()

        def forward(self, x):
            if x.shape[-1] != self.config.input_dimension:
                raise ValueError("network input width mismatch")
            y = (x - self.input_mean) / self.input_scale
            y = self.activation(self.input(y))
            for block in self.blocks:
                y = block(y)
            return self.output(y)

    return ResidualMLP()


__all__ = ["FeatureNormalizer", "ResidualMLPConfig", "build_residual_mlp"]
