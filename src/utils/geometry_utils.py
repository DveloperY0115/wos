"""
geometry_utils.py

A collection of geometry utility functions.
"""

from typing import Literal

from jaxtyping import Shaped, jaxtyped
import numpy as np
from typeguard import typechecked

from .constants import EPS


@jaxtyped(typechecker=typechecked)
def normalize_mesh(
    v: Shaped[np.ndarray, "V 3"],
    method: Literal["unit_cube", "unit_sphere"] = "unit_cube",
) -> Shaped[np.ndarray, "V 3"]:
    """
    Normalizes the vertices of a mesh to a unit cube or a unit sphere.
    """
    if method == "unit_cube":
        v_min = np.min(v, axis=0, keepdims=True)
        v_max = np.max(v, axis=0, keepdims=True)
        v = (v - v_min) / ((v_max - v_min) + EPS)
        v = v - 0.5

    elif method == "unit_sphere":
        v_mean = np.mean(v, axis=0, keepdims=True)
        v = v - v_mean
        v_max = np.max(np.linalg.norm(v, axis=1))
        v = v / (v_max + EPS)

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return v
