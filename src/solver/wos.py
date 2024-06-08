"""
wos.py

A Taichi implementation of the Walk-on-Spheres algorithm.
"""

import taichi as ti
import taichi.math as tm

from ..utils.math_funcs import (
    harmonic_green_3d,
    volume_ball,
)
from ..utils.sampling import (
    uniform_ball,
    uniform_sphere,
)
from ..utils.types import (
    Float1DArray,
    Scene,
    VectorField,
)


@ti.func
def wos_walk(
    query_pt: tm.vec3,
    scene: Scene,
    n_step: int,
    eps: float,
):
    """
    Takes a single step of the Walk-on-Spheres algorithm.
    """
    curr_pt = query_pt

    sol = 0.0
    for _ in range(n_step):

        # Compute the distance to the closest boundary point
        dist = scene.query_dist(curr_pt)
        dist_abs = ti.abs(dist)

        # Terminate the walk when reached boundary
        if dist_abs < eps:
            break

        # Accumulate source term
        src_pt = uniform_ball(dist_abs, curr_pt)
        src_val = scene.query_source(src_pt)
        v_ball = volume_ball(dist_abs)
        sol += v_ball * src_val * harmonic_green_3d(curr_pt, src_pt, dist_abs)

        # Sample a random point on a sphere whose
        # radius is the distance to the closest boundary point
        curr_pt = uniform_sphere(dist_abs, curr_pt)

    # Retrieve the boundary value
    bd_val = scene.query_boundary(curr_pt)
    sol += bd_val

    return sol


@ti.kernel
def wos(
    query_pts: VectorField,
    scene: Scene,
    eps: float,
    n_walk: int,
    n_step: int,
    sol_: Float1DArray,
):
    """
    Simulates the Walk-on-Spheres algorithm.

    Args:
    - n_step: The number of maximum steps for each random walk
    """
    for i in range(query_pts.shape[0]):  # Parallelized
        for _ in range(n_walk):  # Sequential
            sol = wos_walk(
                query_pts[i], scene, n_step, eps

            )
            sol_[i] += sol
        sol_[i] /= n_walk