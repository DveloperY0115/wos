"""
mesh.py
"""

from pathlib import Path

import taichi as ti
import taichi.math as tm

from .base_class import Geometry
from ..structure.TriMesh import TriMesh


class Mesh(Geometry):

    def __init__(
        self,
        obj_path: Path,
    ):
        self.obj_path = obj_path
        assert self.obj_path.exists()

        # Initialize mesh
        self.mesh = TriMesh()
        self.mesh.add_obj(str(self.obj_path))
        self.mesh.setup_layout()
        self.mesh.update_to_device()
        self.mesh.build_bvh()
        self.mesh.setup_vert()
        print("Initialized mesh")

    @ti.func
    def query_dist(self, x: tm.vec3):
        """
        Query the signed distance of a point to the sphere.
        """
        distance, _ = self.mesh.bvh.signed_distance(x)
        return distance

    @ti.func
    def query_boundary(self, x: tm.vec3):
        """
        Query the boundary condition defined over the sphere.
        """
        bd_val = 10.0

        if x[2] < 0 and x[1] >= 0:
            bd_val = 50.0
        elif x[2] < 0 and x[1] < 0:
            bd_val = 0.0
        elif x[2] >= 0 and x[1] < 0:
            bd_val = -50.0
        else:
            bd_val = 0.0

        return bd_val
    
    @ti.func
    def query_source(self, x: tm.vec3):
        """
        Query the source term defined inside the sphere.
        """
        src_val = 0.0
        return src_val
