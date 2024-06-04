"""
sphere.py
"""

from jaxtyping import Shaped, jaxtyped
import numpy as np
import taichi as ti
import taichi.math as tm
from typeguard import typechecked


@ti.dataclass
class Sphere:

    center: tm.vec3
    """Center of the sphere"""
    radius: ti.f32
    """Radius of the sphere"""

    def __init__(
            self,
            center: tm.vec3,
            radius: ti.f32,
        ):
        self.center = center
        self.radius = radius

    @ti.func
    def query(self, x: tm.vec3):
        """
        Query the unsigned distance of a point to the sphere
        """
        distance = tm.length(x - self.center)
        distance = distance - self.radius
        return distance
