"""
heat_sphere.py

A simple toy experiment for solving heat equation inside a sphere.
"""

from dataclasses import dataclass
from pathlib import Path
import time

import igl
from jaxtyping import jaxtyped
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from typeguard import typechecked
import tyro

from src.geometry.mesh import Mesh
from src.solver.wos import wos


# Initialize Taichi
ti.init(arch=ti.gpu)

@dataclass
class Args:
    mesh_path: Path
    """Path to the mesh file"""

    z: float = 0.0
    """The z-coordinate of the plane (i.e., slice) to visualize the heat map"""
    eps: float = 1e-6
    """Threshold to determine whether a walk has reached the domain boundary"""
    n_walk: int = 100000
    """Maximum number of random walks for each query point to simulate"""
    n_step: int = 50
    """Maximum number of steps for each random walk"""
    vis_every: float = 0.0
    """Time interval between subsequent visualizations"""

    img_height: int = 512
    """Height of the image"""
    img_width: int = 512
    """Width of the image"""


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # Initialize problem domain
    v, f = igl.read_triangle_mesh(str(args.mesh_path))
    v_ = ti.ndarray(dtype=ti.f32, shape=(v.shape[0], v.shape[1]))
    v_.from_numpy(v.astype(np.float32))
    v = v_
    f_ = ti.ndarray(dtype=ti.i32, shape=(f.shape[0], f.shape[1]))
    f_.from_numpy(f.astype(np.int32))
    f = f_
    mesh = Mesh(v, f)

    # Initialize query points
    xs = np.linspace(-2.0, 2.0, args.img_width)
    ys = np.linspace(-2.0, 2.0, args.img_height)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # Flatten the query points
    query_pts = np.stack(
        [xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), args.z)],
        axis=1,
    )
    query_pts_ = ti.Vector.field(3, dtype=ti.f32, shape=query_pts.shape[0])
    query_pts_.from_numpy(query_pts.astype(np.float32))
    query_pts = query_pts_
    
    # Initialize GUI
    print("Launching walk...")
    gui = ti.GUI("Heat Sphere", (args.img_width, args.img_height))

    while gui.running:
        sol = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
        for walk_idx in range(args.n_walk):
            wos(query_pts, mesh, args.eps, args.n_step, sol)
            sol = sol.to_numpy()
            sol = sol.reshape(args.img_height, args.img_width)

            # Visualize the solution
            sol_vis = sol.copy() / (walk_idx + 1)
            sol_vis = plt.cm.coolwarm(plt.Normalize()(sol_vis))
            gui.set_image(sol_vis)
            gui.show()

            sol = sol.reshape(args.img_height * args.img_width)
            sol_ = ti.ndarray(dtype=ti.f32, shape=(sol.shape[0]))
            sol_.from_numpy(sol)
            sol = sol_

            time.sleep(max(args.vis_every, 0.0))


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )