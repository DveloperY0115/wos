"""
base_class.py

A base class for defining geometries.
"""

import taichi as ti
import taichi.math as tm

class Geometry:
    def __init__(self):
        pass

    @ti.func
    def query_dist(self, x: tm.vec3):
        raise NotImplementedError()

    @ti.func
    def query_boundary(self, x: tm.vec3):
        raise NotImplementedError()

    @ti.func
    def query_source(self, x: tm.vec3):
        raise NotImplementedError()

    @ti.func
    def query_screen_constant(self, x: tm.vec3):
        raise NotImplementedError()