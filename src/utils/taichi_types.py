"""
types.py

Definitions of types used in the project.
"""

import taichi as ti


"""Struct"""
Scene = ti.template()  # FIXME: Is this a right way to type a struct?

"""Field Types"""
VectorField = ti.template()
MatrixField = ti.template()

"""N-D Floating-Point Array Types"""
Float1DArray = ti.types.ndarray(ti.f32, ndim=1)
Float2DArray = ti.types.ndarray(ti.f32, ndim=2)
Float3DArray = ti.types.ndarray(ti.f32, ndim=3)

"""N-D Integer Array Types"""
Int1DArray = ti.types.ndarray(ti.i32, ndim=1)
Int2DArray = ti.types.ndarray(ti.i32, ndim=2)
Int3DArray = ti.types.ndarray(ti.i32, ndim=3)