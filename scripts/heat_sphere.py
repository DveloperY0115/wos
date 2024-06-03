"""
heat_sphere.py

A simple toy experiment for solving heat equation inside a sphere.
"""

from dataclasses import dataclass

from jaxtyping import Shaped, jaxtyped
from typeguard import typechecked
import torch
import tyro

from src.geometry.sphere import Sphere


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
    sphere = Sphere(center=torch.zeros(3), radius=args.radius)

    # Initialize query points
    xs = torch.linspace(-2.0, 2.0, args.img_width)
    ys = torch.linspace(-2.0, 2.0, args.img_height)
    xx, yy = torch.meshgrid(xs, ys, indexing="xy")

    # Flatten the query points
    query_pts = torch.stack(
        [xx.flatten(), yy.flatten(), torch.full_like(xx.flatten(), args.z)],
        dim=1,
    )

    # Recursive call into the walk
    sol = 


    # for walk_idx in range(args.n_walk):

    #     # Query the unsigned distance of the query points to the sphere
    #     dists = sphere.query(query_pts)




if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )