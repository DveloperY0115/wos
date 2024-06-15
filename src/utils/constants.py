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
    LAPLACE = 0
    """Laplace's equation"""
    POISSON = 1
    """Poisson's equation"""
    SCREENED_POISSON = 2
    """Screened Poisson's equation"""
