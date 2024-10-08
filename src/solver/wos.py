"""
wos.py

A Taichi implementation of the Walk-on-Spheres algorithm.
"""

import taichi as ti
import taichi.math as tm

from ..utils.constants import EquationType
from ..utils.math_funcs import (
    harmonic_green_3d,
    volume_ball,
    yukawa_potential_3d,
)
from ..utils.sampling import (
    uniform_ball,
    uniform_sphere,
)
from ..utils.taichi_types import (
    Int1DArray,
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
    eqn_type: ti.template(),  # FIXME: Need to add a proper type hint
    fetch_clst_bd: bool = True,
):
    """
    Takes a single step of the Walk-on-Spheres algorithm.
    """
    curr_pt = query_pt

    sol = 0.0
    norm = 1.0  # Normalization constant used in Screened Poisson's equation
    outside = False
    terminated = False
    for step_idx in range(n_step):

        # Compute the distance to the closest boundary point
        dist = scene.query_dist(curr_pt)
        dist_abs = ti.abs(dist)

        # Terminate the walk when outside the domain
        if step_idx == 0 and dist > 0.0:
            outside = True
            break

        # Terminate the walk when reached boundary
        if dist_abs < eps:
            terminated = True
            break

        # Accumulate source term if necessary
        if eqn_type == EquationType.laplace:
            pass  # Do nothing

        elif eqn_type == EquationType.poisson:  # FIXME: Numerical issue
            src_pt = uniform_ball(dist_abs, curr_pt)
            src_val = scene.query_source(src_pt)
            v_ball = volume_ball(dist_abs)
            sol += v_ball * src_val * harmonic_green_3d(curr_pt, src_pt, dist_abs)

        elif eqn_type == EquationType.screened_poisson:  # FIXME: Numerical issue
            src_pt = uniform_ball(dist_abs, curr_pt)
            src_val = scene.query_source(src_pt)
            v_ball = volume_ball(dist_abs)

            c = scene.query_screen_constant(curr_pt)
            yukawa, norm_curr = yukawa_potential_3d(curr_pt, src_pt, dist_abs, c)
            sol += norm * v_ball * src_val * yukawa
            norm = norm * norm_curr  # Accumulate normalizer

        else:
            pass  # FIXME: Need to raise error, but Taichi does not support it

        # Sample a random point on a sphere whose
        # radius is the distance to the closest boundary point
        curr_pt = uniform_sphere(dist_abs, curr_pt)

    # Retrieve the boundary value
    if outside:
        bd_val = 0.0
    else:
        # NOTE: Fetch the boundary value if either:
        # - The walk reached the boundary and terminated
        # - The walk did not reach the boundary but we want to fetch the closest boundary value
        if terminated or fetch_clst_bd:
            bd_val = scene.query_boundary(curr_pt)
            if eqn_type == EquationType.screened_poisson:
                sol += norm * bd_val
            else:
                sol += bd_val

    return sol, terminated

@ti.kernel
def wos(
    query_pts: VectorField,
    scene: Scene,
    eps: float,
    n_step: int,
    eqn_type: ti.template(),  # FIXME: Need to add a proper type hint
    sol_: Float1DArray,
    term_mask_: Int1DArray,
):
    """
    Simulates the Walk-on-Spheres algorithm.

    Args:
    - n_step: The number of maximum steps for each random walk
    """
    for i in range(query_pts.shape[0]):  # Parallelized
        curr_sol, terminated = wos_walk(
            query_pts[i], scene, n_step, eps, eqn_type
        )
        sol_[i] = curr_sol
        term_mask_[i] = 1 if terminated else 0