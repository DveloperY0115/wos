"""
sampling.py

A collection of utility functions related to random sampling.
"""

from jaxtyping import jaxtyped
import numpy as np
import taichi as ti
import taichi.math as tm
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
@ti.func
def uniform_sphere(radius: float):
    """
    Implements uniform sampling from a sphere in 3D space.
    """
    eps1 = ti.random(float)
    eps2 = ti.random(float)

    theta = 2 * np.pi * eps1

    # Uniform sampling on the unit sphere
    z = 1 - 2 * eps2
    x = tm.cos(theta) * tm.sqrt(1 - z * z)
    y = tm.sin(theta) * tm.sqrt(1 - z * z)

    # Scale the sample to the desired radius
    sample = radius * tm.vec3([x, y, z])

    return sample

@jaxtyped(typechecker=typechecked)
@ti.func
def uniform_ball(radius: float):
    """
    Implements uniform sampling from a ball in 3D space.
    """
    eps1 = ti.random(float)
    eps2 = ti.random(float)
    eps3 = ti.random(float)

    # Uniform sampling inside the unit ball
    z = (eps1 ** (1 / 3)) * (1 - 2 * eps2)
    x = tm.sqrt(eps1 ** (2 / 3) - z * z) * tm.cos(2 * np.pi * eps3)
    y = tm.sqrt(eps1 ** (2 / 3) - z * z) * tm.sin(2 * np.pi * eps3)

    # Scale the sample to the desired radius
    sample = radius * tm.vec3([x, y, z])

    return sample