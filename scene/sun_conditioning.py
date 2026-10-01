"""Small shared decoder for sun-dependent, per-Gaussian SH residuals."""

import torch
from torch import nn
from utils.sh_utils import C0, C1, C2


def sun_sh_encoding(direction, degree=2):
    """Real SH in the same ordering as utils.sh_utils.eval_sh.

    Directions are validated on the CPU when cameras/CLI inputs are loaded.
    Keep this hot path free of GPU-to-CPU synchronization.
    """
    if degree not in (1, 2):
        raise ValueError("Sun SH degree must be 1 or 2")
    if direction.shape != (3,):
        raise ValueError("Expected one sun direction with shape (3,)")
    s = direction / direction.norm().clamp_min(1e-8)
    x, y, z = s.unbind()
    basis = [torch.ones_like(x) * C0, -C1 * y, C1 * z, -C1 * x]
    if degree == 2:
        basis += [C2[0] * x * y, C2[1] * y * z,
                  C2[2] * (2 * z * z - x * x - y * y),
                  C2[3] * x * z, C2[4] * (x * x - y * y)]
    return torch.stack(basis)


class SunSHDecoder(nn.Module):
    def __init__(self, feature_dim=16, hidden_dim=32, sun_sh_degree=2,
                 view_sh_degree=3):
        super().__init__()
        if feature_dim < 1 or hidden_dim < 1 or sun_sh_degree not in (1, 2):
            raise ValueError("Invalid sun decoder dimensions or SH degree")
        if not 0 <= view_sh_degree <= 3:
            raise ValueError("View SH degree must be between 0 and 3")
        self.coefficient_count = (view_sh_degree + 1) ** 2
        self.layers = nn.Sequential(
            nn.Linear(feature_dim + (sun_sh_degree + 1) ** 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.coefficient_count * 3),
        )
        self.sun_sh_degree = sun_sh_degree
        nn.init.zeros_(self.layers[-1].weight)
        nn.init.zeros_(self.layers[-1].bias)

    def forward(self, features, direction):
        sun = sun_sh_encoding(direction, self.sun_sh_degree)
        inputs = torch.cat((features, sun.expand(features.shape[0], -1)), dim=1)
        return self.layers(inputs).reshape(-1, self.coefficient_count, 3)
