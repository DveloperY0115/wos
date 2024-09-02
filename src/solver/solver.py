"""
solver.py

A wrapper around variants of the Walk-on-Spheres algorithms.
"""

from pathlib import Path
from typing import Any, Literal

from jaxtyping import jaxtyped
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import torch
from tqdm import tqdm

from .wos import wos


class Solver:

    type: Literal["wos"]
    """Type of the solver algorithm."""
    cache: Any
    """Cache for the solver algorithm."""

    def __init__(
        self,
        type: Literal["wos"] = "wos",
        cache: Any = None
    ) -> None:
        """Initializes the solver."""

        self.type = type
        self.cache = cache

    def solve(
        self,
        query_pts: ti.template(),
        scene: ti.template(),
        img_height: int,
        img_width: int,
        eps: float,
        n_step: int,
        eqn_type: ti.template(),
        n_walk: int,
        vis_every: int,
        out_dir: Path,
    ) -> Any:
        """Invokes the solver algorithm."""
        
        total_sol = np.zeros((img_height, img_width))
        for walk_idx in tqdm(range(n_walk)):
            
            # Allocate memory for the solution obtained from this walk
            curr_sol = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
            curr_mask = ti.ndarray(dtype=ti.i32, shape=(query_pts.shape[0]))

            # Solve PDE via random walk
            if self.type == "wos":
                wos(
                    query_pts,
                    scene,
                    eps,
                    n_step,
                    eqn_type,
                    curr_sol,
                    curr_mask,
                )
            elif self.type == "wost":
                raise NotImplementedError("TODO")

            else:
                raise ValueError(f"Unknown solver type: {self.type}")
        
            curr_sol = curr_sol.to_numpy()
            curr_sol = curr_sol.reshape(img_height, img_width)
            curr_mask = curr_mask.to_numpy()
            curr_mask = curr_mask.reshape(img_height, img_width)

            # Compute the cumulative average
            total_sol = (curr_sol + (walk_idx + 1) * total_sol) / (walk_idx + 2)
            assert not np.any(np.isnan(total_sol)), "NaN detected in the solution"
            assert not np.any(np.isinf(total_sol)), "Inf detected in the solution"

            if (walk_idx + 1) % vis_every == 0:
                sol_vis = plt.cm.coolwarm(plt.Normalize()(total_sol))
                ti.tools.imwrite(sol_vis, str(out_dir / f"sol_{walk_idx+1:04d}.png"))

        return query_pts, total_sol
