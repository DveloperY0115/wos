"""
sampling.py

A collection of utility functions related to random sampling.
"""

from typing import Union

from jaxtyping import Shaped, jaxtyped
import numpy as np
import torch
from torch import Tensor
from typeguard import typechecked


@jaxtyped(typechecker=typechecked)
def uniform_sphere(
    n_sample: int,
    radius: float,
    device: Union[str, torch.device] = "cpu",
) -> Shaped[Tensor, "N 3"]:
    """
    Implements uniform sampling from a sphere in 3D space.
    """
    eps1 = torch.rand(n_sample, device=device)
    eps2 = torch.rand(n_sample, device=device)

    theta = 2 * np.pi * eps1

    # Uniform sampling on the unit sphere
    z = 1 - 2 * eps2
    x = torch.cos(theta) * torch.sqrt(1 - z * z)
    y = torch.sin(theta) * torch.sqrt(1 - z * z)

    # Scale the samples to the desired radius
    samples = radius * torch.stack([x, y, z], dim=1)

    return samples

@jaxtyped(typechecker=typechecked)
def uniform_ball(
    n_sample: int,
    radius: float,
    device: Union[str, torch.device] = "cpu",
) -> Shaped[Tensor, "N 3"]:
    """
    Implements uniform sampling from a ball in 3D space.
    """
    eps1 = torch.rand(n_sample, device=device)
    eps2 = torch.rand(n_sample, device=device)
    eps3 = torch.rand(n_sample, device=device)

    # Uniform sampling inside the unit ball
    z = (eps1 ** (1 / 3)) * (1 - 2 * eps2)
    x = torch.sqrt(eps1 ** (2 / 3) - z * z) * torch.cos(2 * np.pi * eps3)
    y = torch.sqrt(eps1 ** (2 / 3) - z * z) * torch.sin(2 * np.pi * eps3)

    # Scale the samples to the desired radius
    samples = radius * torch.stack([x, y, z], dim=1)

    return samples