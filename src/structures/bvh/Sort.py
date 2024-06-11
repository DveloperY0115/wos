import taichi as ti

from . import Primitive


@ti.data_oriented
class Sort:
    def __init__(self, prim):
        
        self.prim            = prim
        self.primitive_count = prim.primitive_count
        self.leaf_count      = ti.field( dtype=ti.i32, shape=(1))

        self.minboundarynp   = prim.minboundarynp
        self.maxboundarynp   = prim.maxboundarynp

        self.min_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))
        self.max_boundary    = ti.Vector.field(3, dtype=ti.f32, shape=(1))


        self.radix_count_zero = ti.field(dtype=ti.i32, shape=[1])
        self.radix_offset     = ti.Vector.field(2, dtype=ti.i32)
 
        self.morton_code_s    = ti.Vector.field(2, dtype=ti.i32)
        self.morton_code_d    = ti.Vector.field(2, dtype=ti.i32)
        self.morton_code      = ti.Vector.field(3, dtype=ti.i32)


    ########################host function#####################################
    @ti.pyfunc
    def blelloch_scan_host(self, mask, move):
        self.radix_sort_predicate(mask, move)

        for i in range(1, self.primitive_bit+1):
            self.blelloch_scan_reduce(1<<i)

        for i in range(self.primitive_bit+1, 0, -1):
            self.blelloch_scan_downsweep(1<<i)

        #print(self.radix_offset.to_numpy(), self.radix_count_zero.to_numpy())

    @ti.pyfunc
    def radix_sort_host(self):
        for i in range(30):
            mask   = 0x00000001 << i
            self.blelloch_scan_host(mask, i)
            #print("********************", self.radix_count_zero.to_numpy())
            self.radix_sort_fill(mask, i)

    @ti.pyfunc
    def print_morton_reslut(self, morton, only_error):
        tmp = morton.to_numpy()
        leaf = self.leaf_count.to_numpy()
        for i in range(0, leaf[0]):
            if i > 0:
                if tmp[i,0] < tmp[i-1,0]:
                    print(i, tmp[i,:], tmp[i-1,:], "!!!!!!!!!!wrong!!!!!!!!!!!!")
                elif tmp[i,0] == tmp[i-1,0]:
                    print(i, tmp[i,:], tmp[i-1,:], "~~~~~~equal~~~~~~~~~~~~~")
                elif not only_error:
                    print(i, tmp[i,:])
            elif not only_error:
                print(i, tmp[i,:])  


    @ti.pyfunc
    def get_pot_num(self, num):
        m = 1
        while m<num:
            m=m<<1
        return m >>1

    @ti.pyfunc
    def get_pot_bit(self, num):
        m   = 1
        cnt = 0
        while m<num:
            m=m<<1
            cnt += 1
        return cnt
        
    @ti.pyfunc
    def setup_layout(self):
        self.primitive_pot   = (self.get_pot_num(self.primitive_count)) << 1
        self.primitive_bit   = self.get_pot_bit(self.primitive_pot)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_s)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code_d)
        ti.root.dense(ti.i, self.primitive_count ).place(self.morton_code)
        ti.root.dense(ti.i, self.primitive_pot   ).place(self.radix_offset)

    @ti.pyfunc
    def update_to_device(self):
        self.max_boundary.from_numpy(self.maxboundarynp)
        self.min_boundary.from_numpy(self.minboundarynp)


    @ti.pyfunc
    def sort(self):
        self.build_morton_3d()
        print("morton code is built")

        self.radix_sort_host()
        print("radix sort  is done")




        #self.print_morton_reslut(self.morton_code, True)
        self.print_morton_reslut(self.morton_code, False)

    ############algrithm##############
    @ti.func
    def expandBits(self,  x):
        '''
        # nvidia blog : https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/
        v = ( (v * 0x00010001) & 0xFF0000FF)
        v = ( (v * 0x00000101) & 0x0F00F00F)
        v = ( (v * 0x00000011) & 0xC30C30C3)
        v = ( (v * 0x00000005) & 0x49249249)
        taichi can not handle it, so i change that to bit operate
        '''
        x = (x | (x << 16)) & 0x030000FF
        x = (x | (x <<  8)) & 0x0300F00F
        x = (x | (x <<  4)) & 0x030C30C3
        x = (x | (x <<  2)) & 0x09249249
        return x

    @ti.func
    def morton3D(self, x, y, z):
        x = min(max(x * 1024.0, 0.0), 1023.0)
        y = min(max(y * 1024.0, 0.0), 1023.0)
        z = min(max(z * 1024.0, 0.0), 1023.0)
        xx = self.expandBits(ti.cast(x, dtype = ti.i32))
        yy = self.expandBits(ti.cast(y, dtype = ti.i32))
        zz = self.expandBits(ti.cast(z, dtype = ti.i32))
        #return zz  | (yy << 1) | (xx<<2)
        code = xx  | (yy << 1) | (zz<<2)
        if code == 0:
            print(x,y,z)
        return code

    @ti.kernel
    def build_morton_3d(self):
        for i in range(self.prim.primitive_count):
            centre_p = self.prim.center(i)
            norm_p    = (centre_p - self.min_boundary[0])/(self.max_boundary[0] - self.min_boundary[0])
            self.morton_code_s[i][0] = self.morton3D(norm_p.x, norm_p.y, norm_p.z)
            self.morton_code_s[i][1] = i
            self.morton_code[i][0]   = 0
            self.morton_code[i][1]   = 1
            self.morton_code[i][2]   = 0
            #print(self.morton_code_s[i][0] )

    @ti.kernel
    def radix_sort_predicate(self,  mask: ti.i32, move: ti.i32):
        for i in self.radix_offset:
            if i < self.primitive_count:
                self.radix_offset[i][1]       = (self.morton_code_s[i][0] & mask ) >> move
                self.radix_offset[i][0]       = 1-self.radix_offset[i][1]
                ti.atomic_add(self.radix_count_zero[0], self.radix_offset[i][0]) 
            else:
                self.radix_offset[i][0]       = 0
                self.radix_offset[i][1]       = 0

 
    @ti.kernel
    def blelloch_scan_reduce(self, mod: ti.i32):
        for i in self.radix_offset:
            if (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                self.radix_offset[i][0] += self.radix_offset[prev_index][0]
                self.radix_offset[i][1] += self.radix_offset[prev_index][1]

    @ti.kernel
    def blelloch_scan_downsweep(self, mod: ti.i32):
        for i in self.radix_offset:

            if mod == (self.primitive_pot*2):
                self.radix_offset[self.primitive_pot-1] = ti.Vector([0,0])
            elif (i+1)%mod == 0:
                prev_index = i - (mod>>1)
                if prev_index >= 0:
                    tmpV   = self.radix_offset[prev_index]
                    self.radix_offset[prev_index] = self.radix_offset[i]
                    self.radix_offset[i] += tmpV

    @ti.kernel
    def radix_sort_fill(self,  mask: ti.i32, move: ti.i32):
        for i in self.morton_code_s:
            condition = (self.morton_code_s[i][0] & mask ) >> move
            if condition == 1:
                offset = self.radix_offset[i][1] + self.radix_count_zero[0]
                self.morton_code_d[offset] = self.morton_code_s[i]
            else:
                offset = self.radix_offset[i][0] 
                self.morton_code_d[offset] = self.morton_code_s[i]
        for i in self.morton_code_s:
            self.morton_code_s[i]    = self.morton_code_d[i]
            self.radix_count_zero[0] = 0

    @ti.pyfunc
    def merge(self, max_prim):
        #max_prim = ts.clamp(max_prim,1,8)

        #WorseCase.obj max_prim = 1
        #count equal statuation
        # prim index: 2 3 1 0 4 5 6 7 8 9
        # morton    : 3 3 3 4 4 5 6 6 6 7
        # condition : 0 0 0 1 0 1 1 0 0 1
        
        if max_prim > 1:
            self.merge_num_condition(max_prim)
        self.merge_equal_condition(max_prim)
        #scan
        # input     : 0 0 0 1 0 1 1 0 0 1
        # scan      : 0 0 0 1 1 2 3 3 3 4
        # scatter   ：3 2 1 3 1
        self.hillis_scan_for_merge_host()
        self.scatter_for_merge()

        #construct
        # motorn    : 3 4 5 6 7
        # scan      ：0 3 5 6 9  start index of prim
        # count     ：3 2 1 3 1  num of same prim
        self.hillis_scan_for_merge_host()
        self.merge_construct()


        #self.print_morton_reslut(self.morton_code, True)
        #self.print_morton_reslut(self.morton_code, False)

        #print("merge equal morton and small node is done")
        #print("************************") 
        #tmps = self.morton_code_s.to_numpy()
        #tmpd = self.morton_code_d.to_numpy()
        #tmp = self.morton_code.to_numpy()
        #for j in range(0, self.primitive_count):
        #    print(j,tmps[j,:],tmpd[j,:],tmp[j,:])  
        #print("************************") 


    @ti.kernel
    def merge_num_condition(self, max_prim:ti.i32):
        for i in self.morton_code_d:
            if i%max_prim == 0:
                self.morton_code_d[i][0] = 1
                self.morton_code_d[i][1] = 1
            else:
                self.morton_code_d[i][0] = 0
                self.morton_code_d[i][1] = 0

    @ti.kernel
    def merge_equal_condition(self, max_prim:ti.i32):
        for i in self.morton_code_d:
            if i == 0:
                self.morton_code_d[i][0] = 0
                self.morton_code_d[i][1] = 0
            else:
                if(self.morton_code_s[i][0] == self.morton_code_s[i-1][0]):
                    self.morton_code_d[i][0] = 0
                    self.morton_code_d[i][1] = 0
                elif max_prim == 1:
                    self.morton_code_d[i][0] = 1
                    self.morton_code_d[i][1] = 1


    @ti.pyfunc
    def hillis_scan_for_merge_host(self):
        mod = 1
        while mod < self.primitive_pot:
            self.hillis_scan_reduce_for_merge(mod)
            mod *= 2

    @ti.kernel
    def hillis_scan_reduce_for_merge(self, mod: ti.i32):
        for i in self.morton_code_d:
            if i+mod < self.primitive_count:
                self.morton_code_d[i+mod][1] += self.morton_code_d[i][0]

        for i in self.morton_code_d:
            self.morton_code_d[i][0] = self.morton_code_d[i][1]


    
    @ti.kernel
    def scatter_for_merge(self):
        for i in self.morton_code_d:
            pos = self.morton_code_d[i][0]
            self.morton_code[pos][1]  = 0
            self.morton_code[pos][2]  += 1
            
        for i in self.morton_code_d:
            self.morton_code_d[i][0] = self.morton_code[i][2] 
            self.morton_code_d[i][1] = self.morton_code[i][2] 
            if self.morton_code[i][2] > 0:
                self.leaf_count[0] += 1
   


    @ti.kernel
    def merge_construct(self):
        for i in self.morton_code_d:
            self.morton_code[i][1]  = self.morton_code_d[i][0] -  self.morton_code[i][2]
            self.morton_code[i][0]  = self.morton_code_s[self.morton_code[i][1]][0]


 



