
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
    tri_mesh.add_obj("./data/spot_unit_cube.obj")
    #tri_mesh.add_obj("model/Simple.obj")
    #tri_mesh.add_obj("model/Normal.obj")
    #tri_mesh.add_obj("model/Large.obj")
    tri_mesh.setup_layout()
    tri_mesh.update_to_device()
    tri_mesh.build_bvh()
    tri_mesh.setup_vert()
    # tri_mesh.write_bvh()

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
