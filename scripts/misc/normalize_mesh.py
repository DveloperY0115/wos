
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import igl
import tyro

from src.utils.geometry_utils import normalize_mesh


@dataclass
class Args:

    mesh_file: Path
    """Mesh file to normalize"""
    method: Literal["unit_cube", "unit_sphere"] = "unit_cube"
    """Normalization method"""


def main(args: Args):
    
    assert args.mesh_file.exists(), f"Mesh file not found: {args.mesh_file}"
    out_file = args.mesh_file.parent / f"{args.mesh_file.stem}_{args.method}{args.mesh_file.suffix}"

    # Load mesh
    v, f = igl.read_triangle_mesh(str(args.mesh_file))
    
    # print(v.max(axis=0), v.min(axis=0))
    # return

    # Normalize mesh
    v = normalize_mesh(v, method=args.method)    
    print(v.max(axis=0), v.min(axis=0))

    # Save normalized mesh
    igl.write_triangle_mesh(str(out_file), v, f)
    print(f"Saved mesh file to {out_file}")


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
