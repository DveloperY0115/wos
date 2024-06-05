"""
wos.py

A Taichi implementation of the Walk-on-Spheres algorithm.
"""

from typing import Any

import taichi as ti

from src.utils.types import (
    Scene,
    VectorField,
)


@ti.func
def wos_step(
    query_pts: VectorField,
    scene: Scene,
    eps: float,
):
    """
    Takes a single step of the Walk-on-Spheres algorithm.
    """
    pass

@ti.kernel
def wos(
    query_pts: VectorField,
    scene: Scene,
    eps: float,
    n_walk: int,
):
    """
    Simulates the Walk-on-Spheres algorithm.
    """
    for i in range(query_pts.shape[0]):  # Parallelized
        for _ in range(n_walk):  # Sequential
            wos_step(query_pts, scene, eps)