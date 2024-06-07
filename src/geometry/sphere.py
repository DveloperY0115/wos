"""
sphere.py
"""

import taichi as ti
import taichi.math as tm


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
    def query_dist(self, x: tm.vec3):
        """
        Query the signed distance of a point to the sphere.
        """
        distance = tm.length(x - self.center)
        distance = distance - self.radius
        return distance

    @ti.func
    def query_boundary(self, x: tm.vec3):
        """
        Query the boundary condition defined over the sphere.
        """
        bd_val = 0.0
        if x[0] >= 0 and x[1] >= 0:
            bd_val = 50.0
        elif x[0] < 0 and x[1] >= 0:
            bd_val = 0.0
        elif x[0] >= 0 and x[1] < 0:
            bd_val = 0.0
        else:
            bd_val = -50.0

        return bd_val