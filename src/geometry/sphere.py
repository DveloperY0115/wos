"""
sphere.py
"""

import taichi as ti
import taichi.math as tm

from .base_class import Geometry


@ti.dataclass
class Sphere(Geometry):

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
    
    @ti.func
    def query_source(self, x: tm.vec3):
        """
        Query the source term defined inside the sphere.
        """
        dist = tm.length(x - self.center)
        src_val = 0.0
        if dist < 0.5 * self.radius:
            src_val = 10.0
        return src_val

    @ti.func
    def query_screen_constant(self, x: tm.vec3):
        """
        Query the screening constant defined over the sphere.
        """
        return 1e-1