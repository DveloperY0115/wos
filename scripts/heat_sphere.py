"""
heat_sphere.py

A simple toy experiment for solving heat equation inside a sphere.
"""

from dataclasses import dataclass

from jaxtyping import jaxtyped
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
from typeguard import typechecked
import tyro

from src.geometry.sphere import Sphere
from src.solver.wos import wos
from src.utils.types import (
    VectorField,
    Float1DArray,
)

# Initialize Taichi
ti.init(arch=ti.gpu)

@dataclass
class Args:
    radius: float = 1.0
    """Radius of the sphere"""
    src_only: bool = True
    """Flag to enable only the source term"""

    z: float = 0.0
    """The z-coordinate of the plane (i.e., slice) to visualize the heat map"""
    eps: float = 1e-6
    """Threshold to determine whether a walk has reached the domain boundary"""
    n_walk: int = 2000
    """Maximum number of random walks for each query point to simulate"""
    n_step: int = 50
    """Maximum number of steps for each random walk"""

    img_height: int = 256
    """Height of the image"""
    img_width: int = 256
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

    # Recursive call into the walk
    print("Launching walk...")
    sol = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
    wos(query_pts, sphere, args.eps, args.n_walk, args.n_step, sol)
    sol = sol.to_numpy()
    sol = sol.reshape(args.img_height, args.img_width)
    # print(f"NaN count: {np.isnan(sol).sum()}")

    print("Visualizing solution of scene. Close the window to continue...")
    plt.imshow(sol, cmap="coolwarm")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
