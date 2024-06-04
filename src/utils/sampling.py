"""
sampling.py

A collection of utility functions related to random sampling.
"""

from jaxtyping import Shaped, jaxtyped
import numpy as np
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def uniform_sphere(
    n_sample: int,
    radius: float,
) -> Shaped[np.ndarray, "N 3"]:
    """
    Implements uniform sampling from a sphere in 3D space.
    """
    eps1 = np.random.rand(n_sample)
    eps2 = np.random.rand(n_sample)

    theta = 2 * np.pi * eps1

    # Uniform sampling on the unit sphere
    z = 1 - 2 * eps2
    x = np.cos(theta) * np.sqrt(1 - z * z)
    y = np.sin(theta) * np.sqrt(1 - z * z)

    # Scale the samples to the desired radius
    samples = radius * np.stack([x, y, z], axis=1)

    return samples

@jaxtyped(typechecker=typechecked)
def uniform_ball(
    n_sample: int,
    radius: float,
) -> Shaped[np.ndarray, "N 3"]:
    """
    Implements uniform sampling from a ball in 3D space.
    """
    eps1 = np.random.rand(n_sample)
    eps2 = np.random.rand(n_sample)
    eps3 = np.random.rand(n_sample)

    # Uniform sampling inside the unit ball
    z = (eps1 ** (1 / 3)) * (1 - 2 * eps2)
    x = np.sqrt(eps1 ** (2 / 3) - z * z) * np.cos(2 * np.pi * eps3)
    y = np.sqrt(eps1 ** (2 / 3) - z * z) * np.sin(2 * np.pi * eps3)

    # Scale the samples to the desired radius
    samples = radius * np.stack([x, y, z], axis=1)

    return samples