
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import tyro

from src.structure import TriMesh
from src.structure import TriMesh_gpu


@dataclass
class Args:

    device_type: Literal["cpu", "gpu"] = "cpu"
    """Device to use """

    out_dir: Path = Path("outputs/test_bvh")
    """Output directory"""

    img_height: int = 512
    """Height of the image"""
    img_width: int = 512
    """Width of the image"""
    z: float = 0.0
    """The z-coordinate of the plane (i.e., slice) to visualize the heat map"""


def main(args: Args):
    
    if args.device_type == "cpu":
        ti.init(arch=ti.cpu)
    elif args.device_type == "gpu":
        ti.init(arch=ti.gpu)
    else:
        raise ValueError(f"Invalid device type: {args.device_type}")

    # Load mesh and build BVH
    if args.device_type == "cpu":
        tri_mesh = TriMesh.TriMesh()
    elif args.device_type == "gpu":
        tri_mesh = TriMesh.TriMesh()
    else:
        raise ValueError(f"Invalid device type: {args.device_type}")

    # tri_mesh.add_obj("model/Test.obj")
    tri_mesh.add_obj("./data/spot_unit_cube.obj")
    #tri_mesh.add_obj("model/Simple.obj")
    #tri_mesh.add_obj("model/Normal.obj")
    #tri_mesh.add_obj("model/Large.obj")
    tri_mesh.setup_layout()
    tri_mesh.update_to_device()
    tri_mesh.build_bvh()
    tri_mesh.setup_vert()
    # tri_mesh.write_bvh()

    # Save data after tree construction
    out_dir = args.out_dir / args.device_type
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {str(out_dir)}")

    # Initialize query points
    xs = np.linspace(-0.75, 0.75, args.img_width)
    ys = np.linspace(-0.75, 0.75, args.img_height)
    xx, yy = np.meshgrid(xs, ys, indexing="ij")

    gui = ti.GUI("BVH", (args.img_height, args.img_width))

    t = 0

    while gui.running:
        z = -0.1

        query_pts = np.stack(
            [np.full_like(xx.flatten(), z), yy.flatten(), xx.flatten()],
            axis=1,
        )
        query_pts_ = ti.Vector.field(3, dtype=ti.f32, shape=query_pts.shape[0])
        query_pts_.from_numpy(query_pts.astype(np.float32))
        query_pts = query_pts_

        dists = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))
        tri_mesh.bvh.signed_distance_field(query_pts, dists)
        dists = dists.to_numpy()
        dists = dists.reshape(args.img_height, args.img_width)

        # TODO: Signed distance query
        dists_vis = dists.copy()
        dists_vis = plt.cm.coolwarm(plt.Normalize()(dists_vis))
        gui.set_image(dists_vis)
        gui.show()
         
        t += 1



if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
