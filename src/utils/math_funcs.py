"""
math_funcs.py
"""

import numpy as np
import taichi as ti
import taichi.math as tm

from .constants import EPS

@ti.func
def safe_length(v: tm.vec3, eps: float = EPS):
    """
    Computes the length of a vector with a small epsilon.
    """
    return tm.max(tm.length(v), eps)

@ti.func
def sinh(x: float):
    """
    Computes the hyperbolic sine of the input value.
    """
    return 0.5 * (tm.exp(x) - tm.exp(-x))

@ti.func
def cosh(x: float):
    """
    Computes the hyperbolic cosine of the input value.
    """
    return 0.5 * (tm.exp(x) + tm.exp(-x))

@ti.func
def volume_ball(radius: float):
    """
    Computes the volume of a ball in 3D space.
    """
    return (4 / 3) * np.pi * (radius ** 3)

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
def harmonic_green_3d(
    x: tm.vec3,
    y: tm.vec3,
    R: float,
    r_clamp: float = 1e-4,
):
    """
    Computes the harmonic Green's function defined in 3D space.

    This function is required for Monte-Carlo estimation of the
    source terms involved in the system.

    The function is defined as follows [Sawhney and Crane, 2020]:
    G(x, y) = (1 / 4 * pi) * ((R - || x - y ||) / || x - y || * R)
    """
    r = safe_length(y - x, r_clamp)
    if r >= R:
        r = R / 2.0
    assert r < R, f"'r' must be less than 'R'. Got {r} and {R}."
    val = (1 / (4 * np.pi)) * (1 / r - 1 / R)
    return val

@ti.func
def yukawa_potential_2d(x: tm.vec2, y: tm.vec2, R: float, c: float):
    """
    Computes the Yukawa potential defined in 2D space and the normalizer.
    """
    raise NotImplementedError("TODO")

@ti.func
def yukawa_potential_3d(x: tm.vec3, y: tm.vec3, R: float, c: float):
    """
    Computes the Yukawa potential defined in 3D space and the normalizer.
    """
    assert R > 0.0, f"The parameter 'R' must be positive. Got {R}."
    assert c > 0.0, f"The parameter 'c' must be positive. Got {c}."

    # Compute potential
    R_ = R + EPS
    r = tm.length(x - y) + EPS
    val = (1 / (4 * np.pi)) * ((sinh((R_ - r) * tm.sqrt(c))) / (r * sinh(R_ * tm.sqrt(c))))
    if tm.isnan(val) or tm.isinf(val):  # TODO: Why NaNs..?
        val = 0.0

    # Compute normalizer
    normalizer = (R_ * tm.sqrt(c)) / (sinh(R_ * tm.sqrt(c)) + EPS)

    return val, normalizer