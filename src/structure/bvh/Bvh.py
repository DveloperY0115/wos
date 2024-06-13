"""
Code borrowed from https://github.com/lyd405121/ti-bvh
"""

import taichi as ti
# import Primitive as Primitive
from . import Primitive

from ...utils.taichi_types import (
    Float1DArray,
    VectorField,
)

#            0                  | 1         | 2             | 3             | 4             |5          |6      |9      | 
#            32bit              | 32bit     | 32bit         | 32bit         | 32bit         |32bit      |96bit  |96bit 
#self.bvh_node  : is_leaf axis  |left_node    right_node      parent_node     leaf_index     leaf_count  min_v3  max_v3   12
#            1bit   2bit     

IS_LEAF         = 1
NOD_VEC_SIZE    = 12
MAX_PRIM        = 6

@ti.data_oriented
class Bvh:
    def __init__(self, prim):
        
        self.prim            = prim
        self.primitive_count = prim.primitive_count
        if MAX_PRIM > 1:
            self.leaf_count      = int(self.primitive_count / MAX_PRIM + 1)
        else:
            self.leaf_count      = self.primitive_count

        self.node_count      = self.leaf_count*2-1
        self.bvh_node         = ti.Vector.field(NOD_VEC_SIZE, dtype=ti.f32)


    ########################host function#####################################
    def print_node_info(self):
        bvh_node = self.bvh_node.to_numpy()
        fo = open("nodelist.txt", "w")
        for index in range(self.node_count):
            is_leaf = int(bvh_node[index, 0]) & 0x0001
            left    = int(bvh_node[index, 1])
            right   = int(bvh_node[index, 2])

            parent  = int(bvh_node[index, 3])
            prim_index = int(bvh_node[index, 4])
            prim_count = int(bvh_node[index, 5])

            min_point = [bvh_node[index, 6], bvh_node[index, 7],  bvh_node[index, 8]]
            max_point = [bvh_node[index, 9], bvh_node[index, 10], bvh_node[index, 11]]
            chech_pass = 1
            leaf_node_count = 0

            if is_leaf == IS_LEAF:
                leaf_node_count += 1
            else:
                for i in range(3):
                    if (min_point[i] != min(bvh_node[left, 6+i], bvh_node[right, 6+i])) & (max_point[i] != max(bvh_node[left, 9+i], bvh_node[right, 9+i])):
                        chech_pass = 0
                        break
                    
            if chech_pass == 1:
                print("node:%d l:%d r:%d p:%d leaf:%d %d  min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, prim_index,prim_count, min_point[0],min_point[1],min_point[2],\
                    max_point[0],max_point[1],max_point[2]), file = fo)
            else:
                print("xxxx:%d l:%d r:%d p:%d leaf:%d %d  min:%.2f %.2f %.2f max:%.2f %.2f %.2f"%(index, left, right, parent, prim_index,prim_count,  min_point[0],min_point[1],min_point[2],\
                    max_point[0],max_point[1],max_point[2]), file = fo)
        fo.close()

    @ti.pyfunc
    def setup_layout(self):
        ti.root.dense(ti.i, self.node_count).place(self.bvh_node )

    ############node manipulate ##############
    @ti.func
    def init_bvh_node(self, index):

        self.bvh_node[index][0]  = -1.0
        self.bvh_node[index][1]  = -1.0
        self.bvh_node[index][2]  = -1.0
        self.bvh_node[index][3]  = -1.0
        self.bvh_node[index][4]  = -1.0
        self.bvh_node[index][5]  = -1.0       
        self.bvh_node[index][6]  = Primitive.INF_VALUE
        self.bvh_node[index][7]  = Primitive.INF_VALUE
        self.bvh_node[index][8]  = Primitive.INF_VALUE
        self.bvh_node[index][9]  = -Primitive.INF_VALUE
        self.bvh_node[index][10] = -Primitive.INF_VALUE
        self.bvh_node[index][11] = -Primitive.INF_VALUE

    @ti.func
    def set_node_type(self,index,type):
        self.bvh_node[index][0] = float(int(self.bvh_node[index][0]) & (0xfffe | type))
    
    @ti.func
    def set_node_axis(self, index, axis):
        axis = axis<<1
        self.bvh_node[index][0] =float(int(self.bvh_node[index][0]) & (0xfff9 | type))
    
    @ti.func
    def set_node_left(self,  index, left):
        self.bvh_node[index][1]  = float(left)
    @ti.func
    def set_node_right(self,  index, right):
        self.bvh_node[index][2]  = float(right)
    @ti.func
    def set_node_parent(self, index, parent):
        self.bvh_node[index][3]  = float(parent)
    @ti.func
    def set_node_leaf_index(self, index, leaf):
        self.bvh_node[index][4]  = float(leaf)
    @ti.func
    def set_node_leaf_count(self, index, count):
        self.bvh_node[index][5]  = float(count)
    @ti.func
    def set_node_min_max(self,  index, minv,maxv):
        self.bvh_node[index][6]  = minv[0]
        self.bvh_node[index][7]  = minv[1]
        self.bvh_node[index][8]  = minv[2]
        self.bvh_node[index][9]  = maxv[0]
        self.bvh_node[index][10] = maxv[1]
        self.bvh_node[index][11] = maxv[2]

    @ti.func
    def get_node_has_box(self,  index):
        return (self.bvh_node[index][6] <= self.bvh_node[index][9]) and (self.bvh_node[index][7] <= self.bvh_node[index][10]) and (self.bvh_node[index][8] <= self.bvh_node[index][11])

    @ti.func
    def get_node_child(self, index):
        return int(self.bvh_node[index][1]), int(self.bvh_node[index][2])
    @ti.func
    def get_node_parent(self, index):
        return int(self.bvh_node[index][3])
    @ti.func
    def get_node_leaf_index(self,  index):
        return int(self.bvh_node[index][4])
    @ti.func
    def get_node_leaf_count(self,  index):
        return int(self.bvh_node[index][5])
    @ti.func
    def get_node_min_max(self,  index):
        return ti.Vector([self.bvh_node[index][6], self.bvh_node[index][7], self.bvh_node[index][8] ]),ti.Vector([self.bvh_node[index][9], self.bvh_node[index][10], self.bvh_node[index][11] ])

    ## you must inplemnt this
    @ti.func
    def get_node_prim_index(self,  leaf_index, count):
        raise NotImplementedError()
    @ti.func
    def get_node_prim_count(self,  leaf_index, count):
        raise NotImplementedError()


    @ti.func
    def ray_trace_brute_force(self,  origin, direction):
        hit_t       = Primitive.INF_VALUE
        hit_pos     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_bary    = ti.Vector([0.0, 0.0]) 
        hit_prim    = -1

        #brute force check
        prim_index = 0
        while prim_index < self.prim.primitive_count:
            t, pos, bary = self.prim.intersect_prim(origin, direction, prim_index)
            if ( t < hit_t ) & (t > 0.0):
                hit_t       = t
                hit_pos     = pos
                hit_bary    = bary
                hit_prim    = prim_index

                minv,maxv = self.prim.aabb(prim_index)  
                #print("bf",prim_index, t,minv,maxv )   
            prim_index +=1
        return  hit_t, hit_pos, hit_bary, hit_prim

    @ti.func
    def signed_distance_brute_force(self,  p, water_tight=1):
        sd          = Primitive.INF_VALUE
        hit_prim    = -1
        #brute force check
        prim_index = 0
        while prim_index < self.prim.primitive_count:
            t = self.prim.unsigned_distance(prim_index, p)
            if (t < sd) :
                sd       = t
                hit_prim = prim_index
                print("bf",prim_index, t)   
            prim_index +=1
        return  sd,ti.Vector([0.0, 1.0, 0.0])




    ############Client Interface############## 
    @ti.kernel
    def ray_trace_cpu(self,  origin: ti.types.vector(3, ti.f32), direction:ti.types.vector(3, ti.f32))-> ti.types.vector(4, ti.f32):
        direction_n = ti.math.normalize(direction)
        #hit_t, hit_pos, hit_bary, hit_prim = self.ray_trace_brute_force(origin,direction_n)
        hit_t, hit_pos, hit_bary, hit_prim = self.ray_trace(origin,direction_n)
        return ti.Vector([hit_t,hit_pos.x,hit_pos.y,hit_pos.z])
    
    @ti.kernel
    def singed_distance_cpu(self,  point: ti.types.vector(3, ti.f32))-> ti.types.vector(7, ti.f32):
        t, closest = self.signed_distance(point)
        #t, normal = self.signed_distance_brute_force(point)
        sign = 1.0
        if (t > 0.0):
            sign = -1.0
        gradient  = (closest - point) * sign

        '''
        #complicated way to do this
        eps = 0.01
        for i in range(3):
            offset = ti.Vector([0.0, 0.0, 0.0]) 
            offset[i] = eps
            tp,closestp = self.signed_distance(point + offset) 
            tn,closestn = self.signed_distance(point - offset) 
            gradient[i] = (tp - tn) / (eps * 2.0)
        '''

        gradient = gradient.normalized()
        return ti.Vector([t,closest.x,closest.y,closest.z,sign*gradient.x,sign*gradient.y,sign*gradient.z])

    @ti.kernel
    def signed_distance_field(
        self,
        pts: VectorField,
        dists_: Float1DArray,
    ):
        for i in range(pts.shape[0]):
            dists_[i], _ = self.signed_distance(pts[i])

        # t, closest = self.signed_distance(pts)
        # #t, normal = self.signed_distance_brute_force(point)
        # sign = 1.0
        # if (t > 0.0):
        #     sign = -1.0
        # gradient  = (closest - pts) * sign

        '''
        #complicated way to do this
        eps = 0.01
        for i in range(3):
            offset = ti.Vector([0.0, 0.0, 0.0]) 
            offset[i] = eps
            tp,closestp = self.signed_distance(point + offset) 
            tn,closestn = self.signed_distance(point - offset) 
            gradient[i] = (tp - tn) / (eps * 2.0)
        '''

        # gradient = gradient.normalized()
        # return ti.Vector([t,closest.x,closest.y,closest.z,sign*gradient.x,sign*gradient.y,sign*gradient.z])

    ############Interface##############
    @ti.func
    def ray_trace(self,  origin, direction):

        hit_t       = Primitive.INF_VALUE
        hit_pos     = ti.Vector([0.0, 0.0, 0.0]) 
        hit_bary    = ti.Vector([0.0, 0.0]) 
        hit_prim    = -1
        MAX_SIZE    = 32
        stack       = ti.Vector([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stack_pos   = 0
        #ckeck_count = 0

        # depth first use stack
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[stack_pos]
            stack[stack_pos] = 0
            stack_pos  = stack_pos-1
            leaf_index = self.get_node_leaf_index(node_index)

            if leaf_index >= 0 :
                leaf_count = self.get_node_leaf_count(node_index)
                count = 0
                while count < leaf_count:
                    prim_index = self.get_node_prim_index(leaf_index, count) 
                    t, pos, bary = self.prim.intersect_prim(origin, direction, prim_index)
                    #ckeck_count += 1
                    if ( t < hit_t ) & (t > 0.0):
                        hit_t       = t
                        hit_pos     = pos
                        hit_bary    = bary
                        hit_prim    = prim_index
                        #print("hierachy",prim_index,count ,leaf_index, t,leaf_count)   
                    count+=1
            else:
                '''
                #depth first search
                min_v,max_v = self.get_node_min_max( node_index)
                ret,min,max = self.prim.slabs(origin, direction,min_v,max_v)
                #ckeck_count += 1
                if min < hit_t:
                    #push
                    left_node,right_node = self.get_node_child(node_index)
                    #depth first search
                    stack[stack_pos+1] = left_node
                    stack[stack_pos+2] = right_node
                    stack_pos              += 2

                '''
                # seems more clever way  
                left_node,right_node = self.get_node_child(node_index)
                lmin_v,lmax_v = self.get_node_min_max( left_node)
                rmin_v,rmax_v = self.get_node_min_max( right_node)

                retl,minl,maxl = self.prim.slabs(origin, direction,lmin_v,lmax_v)
                retr,minr,maxr = self.prim.slabs(origin, direction,rmin_v,rmax_v)
                #ckeck_count += 2
                if minr < hit_t  and minr<=minl :
                    if minl < hit_t :
                        stack_pos += 1
                        stack[stack_pos] = left_node
                    stack_pos += 1
                    stack[stack_pos] = right_node

                if minl < hit_t and minl<minr :
                    if minr < hit_t :
                        stack_pos += 1
                        stack[stack_pos] = right_node
                    stack_pos += 1
                    stack[stack_pos] = left_node
                
            #print(stack)
        if stack_pos == MAX_SIZE:
            print("overflow, need larger stack",origin)
        #print(ckeck_count)
        return  hit_t, hit_pos, hit_bary, hit_prim
        


    @ti.func
    def signed_distance(self,  p, water_tight=1):
        #print("tt",max(ti.Vector([0,0,0]),  ti.Vector([0.0,0.0,0.0])))
        closest_prim= -1
        closest_p   = ti.Vector([0.0,0.0,0.0])
        sd = Primitive.INF_VALUE
        MAX_SIZE    = 32
        stack       = ti.Vector([0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        stack_pos   = 0

        # depth first use stack
        while (stack_pos >= 0) & (stack_pos < MAX_SIZE):
            #pop
            node_index = stack[stack_pos]
            stack[stack_pos] = 0
            stack_pos  = stack_pos-1
            leaf_index = self.get_node_leaf_index(node_index)

            if leaf_index > 0 :
                leaf_count = self.get_node_leaf_count(node_index)
                count = 0
                while count < leaf_count:
                    prim_index = self.get_node_prim_index(leaf_index, count)
                    t,cp = self.prim.unsigned_distance(prim_index, p)
                    if  t < sd :
                        sd       = t
                        closest_prim = prim_index
                        closest_p    = cp
                        #print("hierachy",prim_index,count ,leaf_index, t,leaf_count)   
                    count+=1
            else:
                #depth first search
                min_v,max_v = self.get_node_min_max( node_index)
                t,closet    = self.prim.sdf_box(p,min_v,max_v)
                if t < sd :
                    #push
                    left_node,right_node = self.get_node_child(node_index)
                    #depth first search
                    stack[stack_pos+1] = left_node
                    stack[stack_pos+2] = right_node
                    stack_pos              += 2

        if stack_pos == MAX_SIZE:
            print("overflow, need larger stack")


        #in out check
        sd_extra    = 0.0
        min0,max0 = self.get_node_min_max(0)

        if self.prim.inside_box(p, min0,max0) == 1:
            #sign check
            if (water_tight):
                if (self.prim.is_inside(closest_prim, p) ):
                    sd = -sd
            else:
                #   This is not a watertight mesh
                #   We need to sample more direction
                #    _____
                #   |   
                #   |____|
                origin = p
                direction = ti.math.normalize((p - closest_p))
                hit_t, hit_pos, hit_bary,hit_prim = self.ray_trace(origin, direction)
                if (hit_t < Primitive.INF_VALUE):
                    sd = -sd

        return sd + sd_extra, closest_p

    @ti.func
    def coliison(self):
        #to be done
        a = 0