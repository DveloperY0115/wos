"""
types.py

Definitions of types used in the project.
"""

import taichi as ti
import taichi.math as tm

"""Field Types"""
VectorField = ti.template()
MatrixField = ti.template()

"""N-D Array Types"""
Float1DArray = ti.types.ndarray(ti.f32, ndim=1)
Float2DArray = ti.types.ndarray(ti.f32, ndim=2)
Float3DArray = ti.types.ndarray(ti.f32, ndim=3)

