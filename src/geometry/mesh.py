"""
mesh.py
"""

import igl
from jaxtyping import Float, Int64
import numpy as np
import taichi as ti
import taichi.math as tm

from .base_class import Geometry
from ..utils.taichi_types import Float2DArray, Int2DArray


# @ti.dataclass
class Mesh(Geometry):

    # v: Float2DArray
    # """Vertices of the mesh"""
    # f: Int2DArray
    # """Faces of the mesh"""

    def __init__(
        self,
        v: Float2DArray,
        f: Int2DArray,
    ):
        self.v = v
        self.f = f

    @ti.func
    def query_dist(self, x: tm.vec3):
        """
        Query the signed distance of a point to the sphere.
        """
        v_np = self.v.to_numpy()
        f_np = self.f.to_numpy()
        distance = igl.signed_distance(x, v_np, f_np)
        return distance

    @ti.func
    def query_boundary(self, x: tm.vec3):
        """
        Query the boundary condition defined over the sphere.
        """
        bd_val = 10.0
        # if not self.src_only:
        #     if x[0] >= 0 and x[1] >= 0:
        #         bd_val = 50.0
        #     elif x[0] < 0 and x[1] >= 0:
        #         bd_val = 0.0
        #     elif x[0] >= 0 and x[1] < 0:
        #         bd_val = 0.0
        #     else:
        #         bd_val = -50.0

        return bd_val
    
    @ti.func
    def query_source(self, x: tm.vec3):
        """
        Query the source term defined inside the sphere.
        """
        src_val = 0.0
        return src_val
