"""
heat_sphere.py

A simple toy experiment for solving heat equation inside a sphere.
"""

from dataclasses import dataclass
from pathlib import Path
import time

from jaxtyping import jaxtyped
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from tqdm import tqdm
from typeguard import typechecked
import tyro

from src.geometry.sphere import Sphere
from src.solver.wos import wos


# Initialize Taichi
ti.init(arch=ti.gpu)

@dataclass
class Args:

    out_dir: Path
    """Output directory"""
    use_gui: bool = False
    """Flag to enable GUI visualization"""

    radius: float = 1.0
    """Radius of the sphere"""
    src_only: bool = True
    """Flag to enable only the source term"""

    z: float = 0.0
    """The z-coordinate of the plane (i.e., slice) to visualize the heat map"""
    eps: float = 1e-6
    """Threshold to determine whether a walk has reached the domain boundary"""
    n_walk: int = 10000
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
    sphere = Sphere(
        center=tm.vec3([0.0, 0.0, 0.0]),
        radius=args.radius,
        src_only=args.src_only,
    )

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
    
    # Create output directory
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {args.out_dir}")

    print("Launching walk...")
    if args.use_gui:
        gui = ti.GUI("Heat Sphere", (args.img_width, args.img_height))

        while gui.running:
            sol = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
            for walk_idx in range(args.n_walk):
                wos(query_pts, sphere, args.eps, args.n_step, sol)
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
    else:
        sol = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
        for walk_idx in tqdm(range(args.n_walk)):
            wos(query_pts, sphere, args.eps, args.n_step, sol)
            sol = sol.to_numpy()
            sol = sol.reshape(args.img_height, args.img_width)

            # Visualize the solution
            sol_vis = sol.copy() / (walk_idx + 1)
            sol_vis = plt.cm.coolwarm(plt.Normalize()(sol_vis))
            ti.tools.imwrite(sol_vis, str(args.out_dir / f"sol_{walk_idx:04d}.png"))

            sol = sol.reshape(args.img_height * args.img_width)
            sol_ = ti.ndarray(dtype=ti.f32, shape=(sol.shape[0]))
            sol_.from_numpy(sol)
            sol = sol_


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
