"""
math_funcs.py
"""

import numpy as np
import taichi as ti
import taichi.math as tm

from .constants import EPS


@ti.func
def volume_ball(radius: float):
    """
    Computes the volume of a ball in 3D space.
    """
    return (4 / 3) * np.pi * radius ** 3

@ti.func
def harmonic_green_2d(x: tm.vec2, y: tm.vec2, R: float):
    """
    Computes the harmonic Green's function defined in 2D space.

    This function is required for Monte-Carlo estimation of the
    source terms involved in the system.

    The function is defined as follows [Sawhney and Crane, 2020]:
    G(x, y) = (1 / 2 * pi) * log(R / || x - y ||)
    """
    R_ = R + EPS
    val = (1 / (2 * np.pi)) * tm.log(R_ / (tm.length(x - y) + EPS))
    if tm.isnan(val) or tm.isinf(val):
        val = 0.0
    return val

@ti.func
def harmonic_green_3d(x: tm.vec3, y: tm.vec3, R: float):
    """
    Computes the harmonic Green's function defined in 3D space.

    This function is required for Monte-Carlo estimation of the
    source terms involved in the system.

    The function is defined as follows [Sawhney and Crane, 2020]:
    G(x, y) = (1 / 4 * pi) * ((R - || x - y ||) / || x - y || * R)
    """
    R_ = R + EPS
    val = (1 / (4 * np.pi)) * ((R_ - tm.length(x - y)) / ((tm.length(x - y) * R_) + EPS))
    if tm.isnan(val) or tm.isinf(val):  # TODO: Why NaNs..?
        val = 0.0
    return val
