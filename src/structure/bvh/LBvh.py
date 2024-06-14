"""
Code borrowed from https://github.com/lyd405121/ti-bvh
"""

import time as time

import taichi as ti

from . import Primitive
from . import Sort
from . import Bvh



@ti.data_oriented
class LBvh(Bvh.Bvh):
    def __init__(self, prim):
        Bvh.Bvh.__init__(self, prim)
        self.radix_sort      = Sort.Sort(prim)
        self.bvh_done        = ti.field(dtype=ti.i32, shape=[1])


    @ti.pyfunc
    def setup_layout(self):
        Bvh.Bvh.setup_layout(self)
        self.radix_sort.setup_layout()

    @ti.pyfunc
    def update_to_device(self):
        self.radix_sort.update_to_device()

    @ti.pyfunc
    def build(self):

        start_time = time.perf_counter()
        self.radix_sort.sort()
        self.radix_sort.merge(Bvh.MAX_PRIM)
        self.leaf_count = self.radix_sort.leaf_count.to_numpy()[0]
        self.node_count = self.leaf_count*2 - 1
        print("***************leaf count:%d *******************"%(self.leaf_count))

        self.build_bvh()
        done_prev = 0
        done_num  = 0
        while done_num < self.leaf_count-1:
            self.gen_aabb()
            done_num  = self.bvh_done.to_numpy()
            if done_num == done_prev:
                break
            done_prev = done_num
        
        if done_num != self.leaf_count-1:
            print("aabb gen error!!!!!!!!!!!!!!!!!!!%d"%done_num)
            print(self.bvh_done.to_numpy())
        else:
            print("***************aabb gen suc***************")

        end_time = time.perf_counter()
        print("***************{:.4f} sec build***************".format(end_time - start_time))

        #self.print_node_info()

    ############algrithm##############
    @ti.func
    def common_upper_bits(self, lhs, rhs) :
        x    = lhs ^ rhs
        ret  = 32
        while x > 0:
            x  = x>>1
            ret  -=1
            #print(ret, lhs, rhs, x, find, ret)
        #print(x)
        return ret


    @ti.func
    def determineRange(self, idx):
        l_r_range = ti.cast(ti.Vector([0, self.leaf_count-1]), ti.i32)

        if idx != 0:

            self_code = self.radix_sort.morton_code[idx][0]
            l         = idx-1
            r         = idx+1
            l_code    = self.radix_sort.morton_code[l][0]
            r_code    = self.radix_sort.morton_code[r][0]

            if  (l_code == self_code ) or (r_code == self_code) :
                print("fatal error!!")

            L_delta   = self.common_upper_bits(self_code, l_code)
            R_delta   = self.common_upper_bits(self_code, r_code)

            d = -1
            if R_delta > L_delta:
                d = 1
            delta_min = min(L_delta, R_delta)
            l_max = 2
            delta = -1
            i_tmp = idx + d * l_max

            if ( (0 <= i_tmp) &(i_tmp < self.leaf_count)):
                delta = self.common_upper_bits(self_code, self.radix_sort.morton_code[i_tmp][0])


            while delta > delta_min:
                l_max <<= 1
                i_tmp = idx + d * l_max
                delta = -1
                if ( (0 <= i_tmp) & (i_tmp < self.leaf_count)):
                    delta = self.common_upper_bits(self_code, self.radix_sort.morton_code[i_tmp][0])

            l = 0
            t = l_max >> 1

            while(t > 0):
                i_tmp = idx + (l + t) * d
                delta = -1
                if ( (0 <= i_tmp) & (i_tmp < self.leaf_count)):
                    delta = self.common_upper_bits(self_code, self.radix_sort.morton_code[i_tmp][0])
                if(delta > delta_min):
                    l += t
                t >>= 1

            l_r_range[0] = idx
            l_r_range[1] = idx + l * d
            if(d < 0):
                tmp          = l_r_range[0]
                l_r_range[0] = l_r_range[1]
                l_r_range[1] = tmp 

        return l_r_range
        
    @ti.func
    def findSplit(self, first, last):
        first_code = self.radix_sort.morton_code[first][0]
        last_code  = self.radix_sort.morton_code[last][0]
        delta_node = self.common_upper_bits(first_code, last_code)
        split = first
        stride = last - first
        while 1:
            stride = (stride + 1) >> 1
            middle = split + stride
            if (middle < last):
                delta = self.common_upper_bits(first_code, self.radix_sort.morton_code[middle][0])
                if (delta > delta_node):
                    split = middle
            if stride <= 1:
                break
        return split

    @ti.kernel
    def build_bvh(self):
        for i in self.bvh_node:
            self.init_bvh_node(i)
            self.bvh_done[0] = 0

        for i in self.bvh_node:
            if i >= self.leaf_count-1:
                self.set_node_type( i, Bvh.IS_LEAF)
                leaf_index = self.radix_sort.morton_code[i-self.leaf_count+1][1]
                leaf_count = self.radix_sort.morton_code[i-self.leaf_count+1][2]

                self.set_node_leaf_index( i, leaf_index)
                self.set_node_leaf_count( i, leaf_count)
                min_v3 =ti.Vector([Primitive.INF_VALUE,Primitive.INF_VALUE,Primitive.INF_VALUE])
                max_v3 =ti.Vector([-Primitive.INF_VALUE,-Primitive.INF_VALUE,-Primitive.INF_VALUE])

                count = 0
                while count < leaf_count:
                    prim_index = self.radix_sort.morton_code_s[leaf_index + count][1]
                    min_tmp,max_tmp = self.prim.aabb(prim_index)

                    for l in  ti.static(range(3)):
                        min_v3[l] = ti.min(min_v3[l], min_tmp[l])
                        max_v3[l] = ti.max(max_v3[l], max_tmp[l])
                    #if prim_index == 1054:
                    #    print("hi",min_tmp,max_tmp,min_v3,max_v3,i)
                    count +=1
                self.set_node_min_max(i, min_v3,max_v3)
            else:
                self.set_node_type( i, 1-Bvh.IS_LEAF)
                l_r_range   = self.determineRange(i)
                spilt       = self.findSplit(l_r_range[0], l_r_range[1])
                left_node   = spilt
                right_node  = spilt + 1


                if l_r_range[0] == l_r_range[1]:
                    print(l_r_range, spilt, left_node, right_node,"wrong")
                #else:
                #    print(l_r_range, spilt,left_node, right_node)

                if min(l_r_range[0], l_r_range[1]) == spilt :
                    left_node  += self.leaf_count - 1
            
                if max(l_r_range[0], l_r_range[1]) == spilt + 1:
                    right_node  += self.leaf_count - 1

                self.set_node_left(   i, left_node)
                self.set_node_right(  i, right_node)
                self.set_node_parent( left_node, i)
                self.set_node_parent( right_node, i)

    @ti.kernel
    def gen_aabb(self):
        for i in self.bvh_node:
            if (self.get_node_has_box(i) == 0):
                left_node, right_node = self.get_node_child(i) 
                
                is_left_rdy  = self.get_node_has_box(left_node)
                is_right_rdy = self.get_node_has_box(right_node)

                if (is_left_rdy and is_right_rdy) > 0:
                    l_min, l_max = self.get_node_min_max(left_node)  
                    r_min, r_max = self.get_node_min_max(right_node)  
                    self.set_node_min_max(i, ti.min(l_min, r_min), ti.max(l_max, r_max))
                    self.bvh_done[0] += 1
                    #print("ok", i, left_node, right_node)
    ## 
    @ti.func
    def get_node_prim_index(self,  leaf_index, count):
        return self.radix_sort.morton_code_s[count + leaf_index][1]

    @ti.func
    def get_node_prim_count(self,  leaf_index, count):
        return self.radix_sort.morton_code_s[count + leaf_index][2]