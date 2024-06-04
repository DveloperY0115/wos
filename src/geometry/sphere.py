"""
sphere.py
"""

from jaxtyping import Shaped, jaxtyped
import numpy as np
from typeguard import typechecked


class Sphere:

    center: Shaped[np.ndarray, "3"] 
    """Center of the sphere"""
    radius: float
    """Radius of the sphere"""

    @jaxtyped(typechecker=typechecked)
    def __init__(
            self,
            center: Shaped[np.ndarray, "3"],
            radius: float,
        ):
        self.center = center
        self.radius = radius

    @jaxtyped(typechecker=typechecked)
    def query(self, x: Shaped[np.ndarray, "N 3"]) -> Shaped[np.ndarray, "N"]:
        """
        Query the unsigned distance of a point to the sphere
        """
        d_c2x = np.linalg.norm(x - self.center[None], dim=-1)
        return np.abs(d_c2x - self.radius)
