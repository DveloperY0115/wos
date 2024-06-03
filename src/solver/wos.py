"""
wos.py

An implementation of the Walk-on-Spheres algorithm.
"""

from typing import Any

from jaxtyping import Bool, Float, Int, jaxtyped
from typeguard import typechecked
import torch
from torch import Tensor
import tyro


@jaxtyped(typechecker=typechecked)
def wos_step(
    query_pts: Float[Tensor, "N 3"],
    scene: Any,
    iter_idx: int,
    terminated: Bool[Tensor, "N"],
    walk_cnt: Int[Tensor, "N"],
) -> Float[Tensor, "N"]:
    """
    Takes a single step of the Walk-on-Spheres algorithm.
    """
    pass