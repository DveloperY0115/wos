
import pywavefront
import taichi as ti

from .bvh import LBvh as Bvh
from .bvh import Primitive

MAX_STACK_SIZE =  32
INF_VALUE      = 1000000.0

@ti.data_oriented
class TriMesh:
    def __init__(self):
        self.prim = Primitive.Primitive()
        self.tri_mesh     =  ti.Vector.field(3, float)
        self.line_bvh      =  ti.Vector.field(3, float)
        self.indice_bvh   =  ti.field(dtype=ti.i32)


    @ti.pyfunc
    def add_obj(self, filename):
        scene = pywavefront.Wavefront(filename)
        scene.parse() 

        for name in scene.materials:
            ######process vert#########
            num_vert = len(scene.materials[name].vertices)
            v_format = scene.materials[name].vertex_format
            
            inner_index = 0
            prim_vert_index = 0
            vertex = [0]*9
            while inner_index < num_vert:
                if v_format == 'T2F_V3F':
                    for i in range(3):
                        vertex[i+prim_vert_index] = scene.materials[name].vertices[inner_index + 2 + i]
                    inner_index += 5
   
                if v_format == 'T2F_N3F_V3F':
                    for i in range(3):
                        vertex[i+prim_vert_index] = scene.materials[name].vertices[inner_index + 5 + i]
                    inner_index += 8

                if v_format == 'N3F_V3F':
                    for i in range(3):
                        vertex[i+prim_vert_index] = scene.materials[name].vertices[inner_index + 3 + i]
                    inner_index += 6

                if v_format== 'V3F':
                    for i in range(3):
                        vertex[i+prim_vert_index] = scene.materials[name].vertices[inner_index + 0 + i]
                    inner_index += 3   
                prim_vert_index += 3
                ######process triangle#########
                if prim_vert_index == 9 :
                    self.prim.add_tri([vertex[0],vertex[1],vertex[2]],[vertex[3],vertex[4],vertex[5]],[vertex[6],vertex[7],vertex[8]])
                    prim_vert_index      =0

    @ti.pyfunc
    def setup_layout(self):
        self.prim.setup_layout()
        self.bvh = Bvh.LBvh(self.prim)
        self.bvh.setup_layout()

        ti.root.dense(ti.i, self.prim.primitive_count*3).place(self.tri_mesh )
        ti.root.dense(ti.i, self.bvh.node_count*8).place(self.line_bvh  )
        ti.root.dense(ti.i, self.bvh.node_count*48).place(self.indice_bvh  )


    @ti.pyfunc
    def update_to_device(self):
        self.prim.update_to_device()
        self.bvh.update_to_device()
        #self.prim.print_info()

    @ti.pyfunc
    def build_bvh(self):
        self.bvh.build()   

    @ti.pyfunc
    def get_center(self):
        c = [0]*3
        for i in range(3):
            c[i] = (self.prim.minboundarynp[0,i] +self.prim.maxboundarynp[0,i])*0.5
        return c

    @ti.pyfunc
    def get_size(self):
        s = [0]*3
        for i in range(3):
            s[i] = self.prim.maxboundarynp[0,i] -self.prim.minboundarynp[0,i]
        return s

    @ti.kernel
    def setup_vert(self):
        for i in range(self.prim.primitive_count):
            self.tri_mesh[3*i+0] = self.prim.tri_p1(i)
            self.tri_mesh[3*i+1] = self.prim.tri_p2(i)
            self.tri_mesh[3*i+2] = self.prim.tri_p3(i)

        for i in range(self.bvh.node_count):
            min_v3,max_v3 = self.bvh.get_node_min_max(i)

            vertex_index = int(i* 8)
            self.line_bvh [vertex_index+0] = ti.Vector([min_v3[0], min_v3[1], min_v3[2]])
            self.line_bvh [vertex_index+1] = ti.Vector([min_v3[0], min_v3[1], max_v3[2]])
            self.line_bvh [vertex_index+2] = ti.Vector([max_v3[0], min_v3[1], max_v3[2]])
            self.line_bvh [vertex_index+3] = ti.Vector([max_v3[0], min_v3[1], min_v3[2]])
            self.line_bvh [vertex_index+4] = ti.Vector([min_v3[0], max_v3[1], min_v3[2]])
            self.line_bvh [vertex_index+5] = ti.Vector([min_v3[0], max_v3[1], max_v3[2]])
            self.line_bvh [vertex_index+6] = ti.Vector([max_v3[0], max_v3[1], max_v3[2]])
            self.line_bvh [vertex_index+7] = ti.Vector([max_v3[0], max_v3[1], min_v3[2]])


            index = int(i* 48)
            self.indice_bvh [index+0]  = vertex_index+0
            self.indice_bvh [index+1]  = vertex_index+1
            self.indice_bvh [index+2]  = vertex_index+1
            self.indice_bvh [index+3]  = vertex_index+2
            self.indice_bvh [index+4]  = vertex_index+2
            self.indice_bvh [index+5]  = vertex_index+3
            self.indice_bvh [index+6]  = vertex_index+3
            self.indice_bvh [index+7]  = vertex_index+0

            self.indice_bvh [index+8]  = vertex_index+4
            self.indice_bvh [index+9]  = vertex_index+5
            self.indice_bvh [index+10]  = vertex_index+5
            self.indice_bvh [index+11]  = vertex_index+6
            self.indice_bvh [index+12] = vertex_index+6
            self.indice_bvh [index+13] = vertex_index+7
            self.indice_bvh [index+14] = vertex_index+7
            self.indice_bvh [index+15] = vertex_index+4

            self.indice_bvh [index+16] = vertex_index+0
            self.indice_bvh [index+17] = vertex_index+1
            self.indice_bvh [index+18] = vertex_index+1
            self.indice_bvh [index+19] = vertex_index+5
            self.indice_bvh [index+20] = vertex_index+5
            self.indice_bvh [index+21] = vertex_index+4
            self.indice_bvh [index+22] = vertex_index+4
            self.indice_bvh [index+23] = vertex_index+0

            self.indice_bvh [index+24] = vertex_index+2
            self.indice_bvh [index+25] = vertex_index+3
            self.indice_bvh [index+26] = vertex_index+3
            self.indice_bvh [index+27] = vertex_index+7
            self.indice_bvh [index+28] = vertex_index+7
            self.indice_bvh [index+29] = vertex_index+6
            self.indice_bvh [index+30] = vertex_index+6
            self.indice_bvh [index+31] = vertex_index+2

            self.indice_bvh [index+32] = vertex_index+1
            self.indice_bvh [index+33] = vertex_index+2
            self.indice_bvh [index+34] = vertex_index+2
            self.indice_bvh [index+35] = vertex_index+6
            self.indice_bvh [index+36] = vertex_index+6
            self.indice_bvh [index+37] = vertex_index+5
            self.indice_bvh [index+38] = vertex_index+5
            self.indice_bvh [index+39] = vertex_index+1

            self.indice_bvh [index+40] = vertex_index+0
            self.indice_bvh [index+41] = vertex_index+4
            self.indice_bvh [index+42] = vertex_index+4
            self.indice_bvh [index+43] = vertex_index+7
            self.indice_bvh [index+44] = vertex_index+7
            self.indice_bvh [index+45] = vertex_index+3
            self.indice_bvh [index+46] = vertex_index+3
            self.indice_bvh [index+47] = vertex_index+0



    @ti.pyfunc
    def write_bvh(self):
        fo = open("bvh.obj", "w")
        vertex_index = 1
        node_np = self.bvh.bvh_node.to_numpy()

        for i in range(self.bvh.node_count):

            if node_np[i,4] > 0:
                min_v3 = [node_np[i][6], node_np[i][7], node_np[i][8]]
                max_v3 = [node_np[i][9], node_np[i][10], node_np[i][11]]

                print ("v %f %f %f" %   (min_v3[0], min_v3[1], min_v3[2]), file = fo)
                print ("v %f %f %f" %   (min_v3[0], min_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], min_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], min_v3[1], min_v3[2]), file = fo)
                print ("v %f %f %f" %   (min_v3[0], max_v3[1], min_v3[2]), file = fo)
                print ("v %f %f %f" %   (min_v3[0], max_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], max_v3[1], max_v3[2]), file = fo)
                print ("v %f %f %f" %   (max_v3[0], max_v3[1], min_v3[2]), file = fo)
                
                print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+1, vertex_index+2, vertex_index+3), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+4, vertex_index+5, vertex_index+6, vertex_index+7), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+1, vertex_index+5, vertex_index+4), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+2, vertex_index+3, vertex_index+7, vertex_index+6), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+1, vertex_index+2, vertex_index+6, vertex_index+5), file = fo)
                print ("f %d %d %d %d" %   (vertex_index+0, vertex_index+4, vertex_index+7, vertex_index+3), file = fo)
                vertex_index += 8
        fo.close()

    #@ti.kernel
    #def rotate(self, angle:ti.f32):
    #    for i in range(self.prim.primitive_count):
    #        rot = ti.Matrix([[ti.cos(angle), 0.0,-ti.sin(angle)],[0.0, 1.0, 0.0],[ti.sin(angle), 0.0,ti.cos(angle)]])
    #        for j in range(3):
    #            self.tri_mesh[3*i+j] = rot @ self.tri_mesh[3*i+j]
    #    for i in range(self.bvh.node_count):
    #        rot = ti.Matrix([[ti.cos(angle), 0.0,-ti.sin(angle)],[0.0, 1.0, 0.0],[ti.sin(angle), 0.0,ti.cos(angle)]])
    #        for j in range(8):
    #            self.line_bvh [8*i+j] = rot @ self.line_bvh [8*i+j]
