"""
sphere.py
"""

from jaxtyping import Shaped, jaxtyped
import torch
from torch import Tensor
from typeguard import typechecked


class Sphere:

    center: Shaped[Tensor, "3"] 
    """Center of the sphere"""
    radius: float
    """Radius of the sphere"""

    @jaxtyped(typechecker=typechecked)
    def __init__(
            self,
            center: Shaped[Tensor, "3"],
            radius: float,
        ):
        self.center = center
        self.radius = radius

    @jaxtyped(typechecker=typechecked)
    def query(self, x: Shaped[Tensor, "N 3"]) -> Shaped[Tensor, "N"]:
        """
        Query the unsigned distance of a point to the sphere
        """
        d_c2x = torch.norm(x - self.center[None], dim=-1)
        return torch.abs(d_c2x - self.radius)
