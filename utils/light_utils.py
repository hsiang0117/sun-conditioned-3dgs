"""Input validation shared by camera loading and sun-direction CLI tools."""

import numpy as np


def normalized_sun_direction(value):
    direction = np.asarray(value, dtype=np.float32)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("sun_direction must contain three finite numbers")
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-8:
        raise ValueError("sun_direction must be nonzero")
    return direction / norm
