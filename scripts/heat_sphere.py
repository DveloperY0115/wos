"""
heat_sphere.py

A simple toy experiment for solving heat equation inside a sphere.
"""

from dataclasses import dataclass

from jaxtyping import jaxtyped
from typeguard import typechecked
import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import taichi.math as tm
import tyro

from src.geometry.sphere import Sphere
from src.utils.types import (
    VectorField,
    Float1DArray,
)

# Initialize Taichi
ti.init(arch=ti.gpu)

@dataclass
class Args:
    n_sample: int = 1000
    radius: float = 1.0
    device: str = "cpu"

    z: float = 0.0
    """The z-coordinate of the plane (i.e., slice) to visualize the heat map"""
    eps: float = 1e-3
    """Threshold to determine whether a walk has reached the domain boundary"""
    n_walk: int = 1000
    """Maximum number of random walks to simulate"""

    img_height: int = 256
    img_width: int = 256


@jaxtyped(typechecker=typechecked)
def main(args: Args) -> None:

    # Initialize problem domain
    sphere = Sphere(center=tm.vec3([0.0, 0.0, 0.0]), radius=args.radius)

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

    # Query SDF of sphere
    dists = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
    query_sphere(sphere, query_pts, dists)
    dists = dists.to_numpy()

    # Visualize Signed Distance Field
    dists = dists.reshape(args.img_height, args.img_width)
    print("Visualizing SDF of scene. Close the window to continue...")
    plt.imshow(dists, cmap="coolwarm")
    plt.colorbar()
    plt.show()

    # Recursive call into the walk
    # sol = 

    # for walk_idx in range(args.n_walk):

    #     # Query the unsigned distance of the query points to the sphere
    #     dists = sphere.query(query_pts)



@ti.kernel
def query_sphere(
    sphere: Sphere,
    query_pts: VectorField,
    dists_: Float1DArray,
):
    """
    Query the unsigned distance of a set of points to the sphere
    """
    for i in query_pts:
        dists_[i] = sphere.query(query_pts[i])


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
