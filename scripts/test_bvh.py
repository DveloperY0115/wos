
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti
import tyro

from src.structure import TriMesh


@dataclass
class Args:

    img_height: int = 512
    img_width: int = 512
    z: float = 0.0


def main(args: Args):
    
    ti.init(arch=ti.cpu)

    # Load mesh and build BVH
    tri_mesh = TriMesh.TriMesh()
    # tri_mesh.add_obj("model/Test.obj")
    tri_mesh.add_obj("../data/xyzrgb_dragon_unit_cube.obj")
    #tri_mesh.add_obj("model/Simple.obj")
    #tri_mesh.add_obj("model/Normal.obj")
    #tri_mesh.add_obj("model/Large.obj")
    tri_mesh.setup_layout()
    tri_mesh.update_to_device()
    tri_mesh.build_bvh()
    tri_mesh.setup_vert()
    # tri_mesh.write_bvh()

    # Initialize query points
    xs = np.linspace(-2.0, 2.0, args.img_width)
    ys = np.linspace(-2.0, 2.0, args.img_height)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")

    # Flatten the query points
    query_pts = np.stack(
        [xx.flatten(), yy.flatten(), np.full_like(xx.flatten(), args.z)],
        axis=1,
    )
    query_pts_ = ti.Vector.field(3, dtype=ti.f32, shape=query_pts.shape[0])
    query_pts_.from_numpy(query_pts.astype(np.float32))
    query_pts = query_pts_

    gui = ti.GUI("BVH", (args.img_height, args.img_width))

    while gui.running:
        sol = ti.ndarray(dtype=ti.f32, shape=(query_pts.shape[0]))

        # TODO: Signed distance query
        sol_vis = plt.cm.coolwarm(plt.Normalize()(sol))
        gui.set_image(sol)
        gui.show()


if __name__ == "__main__":
    main(
        tyro.cli(Args)
    )
