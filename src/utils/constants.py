"""
constants.py

A collection of constants used in the project.
"""

from enum import Enum

EPS = 1e-8
"""A small positive constant used for numerical stability."""

class EquationType(Enum):
    """
    An enumeration of the types of equations supported by solvers.
    """
    laplace = 0
    """Laplace's equation"""
    poisson = 1
    """Poisson's equation"""
    screened_poisson = 2
    """Screened Poisson's equation"""
