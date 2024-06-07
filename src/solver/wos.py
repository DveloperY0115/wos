"""
wos.py

A Taichi implementation of the Walk-on-Spheres algorithm.
"""

import taichi as ti
import taichi.math as tm

from src.utils.sampling import (
    uniform_ball,
)
from src.utils.types import (
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

    for _ in range(n_step):

        # Compute the distance to the closest boundary point
        dist = scene.query_dist(curr_pt)

        # If distance < eps, terminate the walk
        if ti.abs(dist) < eps:
            break

        # Sample a random point on the sphere whose
        # radius is the distance to the closest boundary point
        curr_pt = uniform_ball(ti.abs(dist), curr_pt)

    # Retrieve the boundary value
    bd_val = scene.query_boundary(curr_pt)

    return bd_val


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
            bd_val = wos_walk(
                query_pts[i], scene, n_step, eps

            )
            sol_[i] += bd_val
        sol_[i] /= n_walk