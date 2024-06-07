"""
math_funcs.py
"""

import numpy as np
import taichi as ti
import taichi.math as tm


@ti.func
def harmonic_green_2d(x: tm.vec2, y: tm.vec2, R: float):
    """
    Computes the harmonic Green's function defined in 2D space.

    This function is required for Monte-Carlo estimation of the
    source terms involved in the system.

    The function is defined as follows [Sawhney and Crane, 2020]:
    G(x, y) = (1 / 2 * pi) * log(R / || x - y ||)
    """
    return (1 / (2 * np.pi)) * tm.log(R / tm.length(x - y))

@ti.func
def harmonic_green_3d(x: tm.vec3, y: tm.vec3, R: float):
    """
    Computes the harmonic Green's function defined in 3D space.

    This function is required for Monte-Carlo estimation of the
    source terms involved in the system.

    The function is defined as follows [Sawhney and Crane, 2020]:
    G(x, y) = (1 / 4 * pi) * ((R - || x - y ||) / || x - y || * R)
    """
    return (1 / (4 * np.pi)) * ((R - tm.length(x - y)) / (tm.length(x - y) * R))
