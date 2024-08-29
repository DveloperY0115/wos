"""
sampling.py

A collection of utility functions related to random sampling.
"""

import numpy as np
import taichi as ti
import taichi.math as tm


@ti.func
def safe_rand(eps: float = 1e-4):
    """
    A safe random number generator that avoids generating too small numbers.
    """
    rand = ti.random(float)
    return ti.max(rand, eps)

@ti.func
def uniform_sphere(radius: float, center: tm.vec3):
    """
    Implements uniform sampling from a sphere in 3D space.
    """
    assert radius > 0, (
        f"The argument `radius` must be positive. Got {radius}."
    )

    eps1 = safe_rand()
    eps2 = safe_rand()

    theta = 2 * np.pi * eps1

    # Uniform sampling on the unit sphere
    z = 1 - 2 * eps2
    sqrt_in = ti.max(1 - z * z, 0.0)  # Avoid negative number inside square root
    x = tm.cos(theta) * tm.sqrt(sqrt_in)
    y = tm.sin(theta) * tm.sqrt(sqrt_in)

    # Scale the sample to the desired radius
    sample = radius * tm.vec3([x, y, z])
    sample = center + sample

    return sample

@ti.func
def uniform_ball(radius: float, center: tm.vec3):
    """
    Implements uniform sampling from a ball in 3D space.
    """
    assert radius > 0, (
        f"The argument `radius` must be positive. Got {radius}."
    )

    eps1 = safe_rand()
    eps2 = safe_rand()
    eps3 = safe_rand()

    # Uniform sampling inside the unit ball
    z = (eps1 ** (1 / 3)) * (1 - 2 * eps2)
    sqrt_in = ti.max(eps1 ** (2 / 3) - z * z, 0.0)  # Avoid negative number inside square root
    x = tm.sqrt(sqrt_in) * tm.cos(2 * np.pi * eps3)
    y = tm.sqrt(sqrt_in) * tm.sin(2 * np.pi * eps3)
    if tm.sqrt(x * x + y * y + z * z) >= 1:
        x, y, z = x / 2.0, y / 2.0, z / 2.0

    # Scale the sample to the desired radius
    sample = radius * tm.vec3([x, y, z])
    sample = center + sample

    return sample
