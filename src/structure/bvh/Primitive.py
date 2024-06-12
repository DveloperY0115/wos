
import numpy as np
import taichi as ti
import taichi.math as tm

#memory layout
#Primitive   type           0       1   2   3   4   5   6   7   8   9   (10 x float32)
#            0  tri         type    | A         |B          |C          |
#            1  sphere      type    |center     |r  |
#            2  quad        type    |center     |v1         |v2         |
#            3  box         type    |center     |min        |max        |

#to do
#            4  circle      type    |center     |r  |nomal  |

PRIM_DATA_SIZE  = 10
PRIM_TRI        = 0
PRIM_SPHERE     = 1
PRIM_QUAD       = 2
PRIM_BOX        = 3
INF_VALUE       = 1000000.0

@ti.data_oriented
class Primitive:
    def __init__(self):
        self.maxboundarynp           = np.ones(shape=(1,3), dtype=np.float32)
        self.minboundarynp           = np.ones(shape=(1,3), dtype=np.float32)
        for i in range(3):
            self.maxboundarynp[0, i] = -INF_VALUE
            self.minboundarynp[0, i] = INF_VALUE

        self.primitive_cpu           = []
        self.primitives              = ti.field(dtype=ti.f32)
        self.primitive_count = 0

    ###Add func
    @ti.pyfunc
    def add_tri(self, a, b, c):
        self.primitive_cpu.append([PRIM_TRI,a[0],a[1],a[2],b[0],b[1],b[2],c[0],c[1],c[2]])
        for i in range(3):
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], a[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], a[i])
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], b[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], b[i])
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], c[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], c[i])
        self.primitive_count+=1

    @ti.pyfunc
    def add_sphere(self, center, r):
        self.primitive_cpu.append([PRIM_SPHERE,center[0],center[1],center[2],r,0.0,0.0,0.0,0.0])
        for i in range(3):
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], center[i]-r)
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], center[i]+r)
        self.primitive_count+=1

    @ti.pyfunc
    def add_quad(self, center, v1,v2):
        self.primitive_cpu.append([PRIM_QUAD,center[0],center[1],center[2],v1[0],v1[1],v1[2],v2[0],v2[1],v2[2]])
        for i in range(3):
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], center[i] + v1[i] + v2[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], center[i] + v1[i] + v2[i])
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], center[i] + v1[i] - v2[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], center[i] + v1[i] - v2[i])
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], center[i] - v1[i] + v2[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], center[i] - v1[i] + v2[i])
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], center[i] - v1[i] - v2[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], center[i] - v1[i] - v2[i])                        
        self.primitive_count+=1

    @ti.pyfunc
    def add_box(self, min_v3, max_v3):
        center = [0]*3
        for i in range(3):
            center[i] = (min_v3[i] + max_v3[i]) *0.5
            self.maxboundarynp[0, i] = max(self.maxboundarynp[0, i], max_v3[i])
            self.minboundarynp[0, i] = min(self.minboundarynp[0, i], min_v3[i])
        self.primitive_cpu.append([PRIM_BOX,center[0],center[1],center[2],min_v3[0],min_v3[1],min_v3[2],max_v3[0],max_v3[1],max_v3[2]])
        self.primitive_count+=1

    @ti.pyfunc
    def setup_layout(self):
        self.primitive_count = len(self.primitive_cpu)
        print("***************prim count:%d *******************"%(self.primitive_count))
        self.primitive_np  = np.zeros(shape=(self.primitive_count, PRIM_DATA_SIZE), dtype=np.float32)
        for i in range(self.primitive_count):
            for j in range(PRIM_DATA_SIZE):
                self.primitive_np[i,j] = self.primitive_cpu[i][j]

        ti.root.dense(ti.ij, [self.primitive_count, PRIM_DATA_SIZE] ).place(self.primitives)

    @ti.pyfunc
    def update_to_device(self):

        print("***************bounding***********************")
        print(self.minboundarynp, self.maxboundarynp)
        print("**************************************")
        self.primitives.from_numpy(self.primitive_np)


    #DATA process
    @ti.func
    def prim_type(self, index):
        return int(self.primitives[index,0])


    @ti.func
    def radius(self, index):
        return self.primitives[index,4] 

    @ti.func
    def center(self, index):
        tmp_center = ti.Vector([self.primitives[index,1],self.primitives[index,2],self.primitives[index,3]])
        if self.prim_type(index) == PRIM_TRI:
            tmp_center[0] += self.primitives[index,4]
            tmp_center[1] += self.primitives[index,5]
            tmp_center[2] += self.primitives[index,6]
            tmp_center[0] += self.primitives[index,7]
            tmp_center[1] += self.primitives[index,8]
            tmp_center[2] += self.primitives[index,9]
            tmp_center    *= 1.0/3.0
        return tmp_center

    @ti.func
    def tri_p1(self, index):
        return ti.Vector([self.primitives[index,1],self.primitives[index,2],self.primitives[index,3]])

    @ti.func
    def tri_p2(self, index):
        return ti.Vector([self.primitives[index,4],self.primitives[index,5],self.primitives[index,6]])

    @ti.func
    def tri_p3(self, index):
        return ti.Vector([self.primitives[index,7],self.primitives[index,8],self.primitives[index,9]])

    @ti.func
    def quad_v1(self, index):
        return ti.Vector([self.primitives[index,4],self.primitives[index,5],self.primitives[index,6]])

    @ti.func
    def quad_v2(self, index):
        return ti.Vector([self.primitives[index,7],self.primitives[index,8],self.primitives[index,9]])

    @ti.func
    def box_min(self, index):
        return ti.Vector([self.primitives[index,4],self.primitives[index,5],self.primitives[index,6]])

    @ti.func
    def box_max(self, index):
        return ti.Vector([self.primitives[index,7],self.primitives[index,8],self.primitives[index,9]])

    @ti.func
    def aabb(self, index):
        min_v3 = ti.Vector([INF_VALUE,INF_VALUE,INF_VALUE])
        max_v3 = ti.Vector([-INF_VALUE,-INF_VALUE,-INF_VALUE])
        if self.prim_type(index) == PRIM_TRI:
            for i in range(3):
                min_v3[i] = ti.min(min_v3[i], self.tri_p1(index)[i])
                max_v3[i] = ti.max(max_v3[i], self.tri_p1(index)[i])
                min_v3[i] = ti.min(min_v3[i], self.tri_p2(index)[i])
                max_v3[i] = ti.max(max_v3[i], self.tri_p2(index)[i])
                min_v3[i] = ti.min(min_v3[i], self.tri_p3(index)[i])
                max_v3[i] = ti.max(max_v3[i], self.tri_p3(index)[i])
            #print(min_v3,index)

        elif self.prim_type(index) == PRIM_SPHERE:
            for i in range(3):
                min_v3[i] = ti.min(min_v3[i], self.center(index)[i]-self.radius(index))
                max_v3[i] = ti.max(max_v3[i], self.center(index)[i]+self.radius(index))
        elif self.prim_type(index) == PRIM_QUAD:
            for i in range(3):
                min_v3[i] = ti.min(min_v3[i], self.center(index)[i] + self.quad_v1(index)[i] + self.quad_v2(index)[i] )
                max_v3[i] = ti.max(max_v3[i], self.center(index)[i] + self.quad_v1(index)[i] + self.quad_v2(index)[i] )
                min_v3[i] = ti.min(min_v3[i], self.center(index)[i] + self.quad_v1(index)[i] - self.quad_v2(index)[i] )
                max_v3[i] = ti.max(max_v3[i], self.center(index)[i] + self.quad_v1(index)[i] - self.quad_v2(index)[i] )
                min_v3[i] = ti.min(min_v3[i], self.center(index)[i] - self.quad_v1(index)[i] + self.quad_v2(index)[i] )
                max_v3[i] = ti.max(max_v3[i], self.center(index)[i] - self.quad_v1(index)[i] + self.quad_v2(index)[i] )
                min_v3[i] = ti.min(min_v3[i], self.center(index)[i] - self.quad_v1(index)[i] - self.quad_v2(index)[i] )
                max_v3[i] = ti.max(max_v3[i], self.center(index)[i] - self.quad_v1(index)[i] - self.quad_v2(index)[i] )
        elif self.prim_type(index) == PRIM_BOX:
            min_v3 = self.box_min(index)
            max_v3 = self.box_max(index)
        return min_v3,max_v3

    ############algrithm##############          
    @ti.func
    def sample(self, index, u1, u2, u3):
        p = ti.Vector([0.0, 0.0, 0.0])
        n = ti.Vector([0.0, 0.0, 0.0])

        if self.prim_type(index)== PRIM_TRI:
            # https:#www.zhihu.com/question/31706710/answer/53665229
            # https:#stackoverflow.com/questions/68493050/sample-uniformly-random-points-within-a-triangle
            p1 = self.tri_p1(index)
            p2 = self.tri_p2(index)
            p3 = self.tri_p3(index)
            if(u1+u2>1.0):
                u1 = 1.0 - u1
                u2 = 1.0 - u2
            p = p1 + (p3-p1)*u1 + (v2-p1)*u2 
            n =(p2- p1).cross(p3- p1)
            n.normalized()
        elif self.prim_type(index)== PRIM_SPHERE:
            z = 1.0 - 2.0 * u1
            r = ti.sqrt(tm.clamp(1.0 - z * z, 0.0, 1.0))
            phi = 2.0 * 3.1415926 * u2
            x = r * ti.cos(phi)
            y = r * ti.sin(phi)

            n = ti.Vector([x, y, z])
            p = self.center(index) + ti.Vector([x, y, z]) * self.radius(index)

        elif self.prim_type(index) == PRIM_QUAD:
            v1 = self.quad_v1(index)
            v2 = self.quad_v2(index)
            n = v1.cross(v2)
            n.normalized()
            p = self.center(index) +  (u1-0.5) *2.0 * v1 + (u2-0.5) *2.0 *v2

        elif self.prim_type(index) == PRIM_BOX:
            #most case will not using this func
            #first get 6 planes(quad): AX+BY+CZ+D =0 ---> normal = [A,B,C]  why? https:#math.stackexchange.com/questions/1595393/what-is-the-normal-vector-to-the-plane-axbycz-d
            #calculate 6 planes area, use random to choose which planes
            #sample as a quad
            p = p
            n = n
        return p,n

    @ti.func
    def area(self, index):
        tmp_area = 0.0
        if self.prim_type(index)== PRIM_TRI:
            # https:#www.zhihu.com/question/31706710/answer/53665229
            # https:#stackoverflow.com/questions/68493050/sample-uniformly-random-points-within-a-triangle
            p1 = self.tri_p1(index)
            p2 = self.tri_p2(index)
            p3 = self.tri_p3(index)
            tmp_area = ti.math.cross(p2- p1,p3- p1) * 0.5

        elif self.prim_type(index)== PRIM_SPHERE:
            tmp_area = 4.0 * 3.1415926 * self.radius(index)*self.radius(index)
        elif self.prim_type(index) == PRIM_QUAD:
            v1 = self.quad_v1(index)
            v2 = self.quad_v2(index)
            tmp_area = ti.math.cross(v1, v2) * 4.0

        elif self.prim_type(index) == PRIM_BOX:
            #most case will not using this func
            tmp_area = 1.0
        return tmp_area

    
    @ti.func
    def slabs_brute_force(self, origin, direction, minv, maxv):
        # brute force slabs
        ret    = 0
        tmax = INF_VALUE
        tmin = INF_VALUE
        tx = -1.0
        ty = -1.0
        tz = -1.0
        inside = 1

        if (origin.x < minv.x):
            if (direction.x != 0.0):
                tx = (minv.x - origin.x) / direction.x
            inside = 0
        elif (origin.x > maxv.x):
            if (direction.x != 0.0):
                tx = (maxv.x - origin.x) / direction.x
            inside = 0
        if (origin.y < minv.y):
            if (direction.y != 0.0):
                ty = (minv.y - origin.y) / direction.y
            inside = 0
        elif (origin.y > maxv.y):
            if (direction.y != 0.0):
                ty = (maxv.y - origin.y) / direction.y
            inside = 0
        if (origin.z < minv.z):
            if (direction.z != 0.0):
                tz = (minv.z - origin.z) / direction.z
            inside = 0
        elif (origin.z > maxv.z):
            if (direction.z != 0.0):
                tz = (maxv.z - origin.z) / direction.z
            inside = 0


        if (inside==1):
            tmax = 0.0
            tmin = 0.0
            ret = 1
        else:
            tmax = tx
            taxis = 0
            if (ty > tmax):
                tmax = ty
                taxis = 1
            if (tz > tmax):
                tmax = tz
                taxis = 2

            if (tmax < 0.0):
                ret = 0
            else:

                hit = origin + direction * tmax
                if ((hit.x < minv.x  or hit.x > maxv.x) and taxis != 0):
                    ret = 0
                elif ((hit.y < minv.y  or hit.y > maxv.y) and taxis != 1):
                    ret = 0
                elif ((hit.z < minv.z  or hit.z > maxv.z) and taxis != 2):
                    ret = 0
                else:
                    tmin = tmax
                    ret = 1

        #print(ret,origin,direction,minv,maxv)
        return ret,tmin,tmax

    @ti.func
    def slabs(self, origin, direction, minv, maxv):
        # most effcient algrithm for ray intersect aabb 
        # en vesrion: https:#www.researchgate.net/publication/220494140_An_Efficient_and_Robust_Ray-Box_Intersection_Algorithm
        # cn version: https:#zhuanlan.zhihu.com/p/138259656
        ret  = 1
        tmin = 0.0
        tmax = INF_VALUE
        for i in ti.static(range(3)):
            if abs(direction[i]) < 0.000001:
                if ( (origin[i] < minv[i]) | (origin[i] > maxv[i])):
                    ret = 0
                    tmin = INF_VALUE
            else:
                ood = 1.0 / direction[i] 
                t1 = (minv[i] - origin[i]) * ood 
                t2 = (maxv[i] - origin[i]) * ood
                if(t1 > t2):
                    temp = t1 
                    t1 = t2
                    t2 = temp 
                if(t1 > tmin):
                    tmin = t1
                if(t2 < tmax):
                    tmax = t2 
                if(tmin > tmax) :
                    ret=0
                    tmin = INF_VALUE
        return ret,tmin,tmax


    @ti.func
    def sdf_box(self, p,  min,  max):
        closest = ti.Vector([ti.math.clamp(p.x, min.x, max.x), ti.math.clamp(p.y, min.y, max.y), ti.math.clamp(p.z, min.z, max.z)])
        #https://iquilezles.org/articles/distfunctions/
        b = (max - min) * 0.5
        c = (max + min) * 0.5
        q = ti.Vector([abs(p.x - c.x), abs(p.y - c.y), abs(p.z - c.z)]) - b
        dis = (ti.max(q,  ti.Vector([0.0,0.0,0.0]))).norm() + ti.math.min(ti.math.max(q.x, ti.math.max(q.y, q.z)), 0.0)
        return dis,closest

    @ti.func
    def intersect_prim(self, origin, direction, index):

        hit_t     = INF_VALUE
        hit_pos   = ti.Vector([INF_VALUE,INF_VALUE,INF_VALUE])
        hit_bary  = ti.Vector([INF_VALUE,INF_VALUE]) #sometimes we wish bary to calculate texture_uv or tangent or sampled normal 

        if self.prim_type(index) == PRIM_TRI:
            # https:#www.scratchapixel.com/lessons/3d-basic-rendering/ray-tracing-rendering-a-triangle/ray-triangle-intersection-geometric-solution
            p0 = self.tri_p1(index)
            p1 = self.tri_p2(index)
            p2 = self.tri_p3(index)
            #print(index, p0,p1,p2)

            ab = p1 - p0
            ac = p2 - p0
            n = ab.cross(ac)

            d = -direction.dot(n)
            ood = 1.0 / d
            ap = origin - p0

            t = ap.dot(n) * ood
            if (t >= 0.0):
                e = -direction.cross(ap)
                v = ac.dot(e) * ood
                if (v >= 0.0 and v <= 1.0):
                    w = -ab.dot(e) * ood
                    if (w >= 0.0 and v + w <= 1.0):
                        u = 1.0- v - w
                        hit_bary = ti.Vector([u, v])
                        hit_t = t
                        hit_pos = origin + t*direction
        elif self.prim_type(index) == PRIM_SPHERE:
            #   h1    h2          -->two hitpoint
            # o--*--p--*--->d     -->Ray
            #   \   |
            #    \  |
            #     \ |
            #      c              -->circle centre
            r      = self.radius(index)
            centre = self.center(index)
            oc     = centre - origin
            dis_oc_square = oc.dot(oc)
            dis_op        = direction.dot (oc)
            dis_cp        = ti.sqrt(dis_oc_square - dis_op * dis_op)
            if (dis_cp < r):
                # h1 is nearer than h2
                # because h1 = o + t*d
                # so  |ch| = radius = |c - d - t*d| = |oc - td|
                # so  radius*radius = (oc - td)*(oc -td) = oc*oc +t*t*d*d -2*t*(oc*d)
                #so d*d*t^2   -2*(oc*d)* t + (oc*oc- radius*radius) = 0

                #cal ax^2+bx+c = 0
                a = direction.dot(direction)
                b = -2.0 * dis_op
                c = dis_oc_square - r*r

                hit_t = (-b - ti.sqrt(b * b - 4.0 * a * c)) / 2.0 / a
                hit_pos = origin + hit_t * direction
        elif self.prim_type(index) == PRIM_QUAD:
            c  = self.center(index)
            v1 = self.quad_v1(index)
            v2 = self.quad_v2(index)
            v1 = v1 / v1.dot(v1)
            v2 = v2 / v2.dot(v2)
            n = v1.cross(v2)
            n.normalized()

            dt = direction.dot(n)
            PO = (c - origin)
            t = n.dot(PO) / dt
            if (t > 0.0):
                p = origin + direction * t
                vi = p - c
                a1 = v1.dot(vi)
                a2 = v2.dot(vi)
                if (a1 > -1.0 and a1 < 1.0 and a2 > -1.0 and a2 < 1.0):
                    hit_t=   t
                    hit_pos = p

        elif self.prim_type(index) == PRIM_BOX:
            ret,hit_t,tmax = self.slabs(origin, direction, self.box_min(index), self.box_max(index))
            hit_pos = origin + hit_t * direction
        return hit_t, hit_pos, hit_bary
    


    @ti.func
    def inside_box(self,  p, minv, maxv):
        ret = 0
        if (p.x >minv.x and p.y > minv.y and p.z > minv.z and p.x < maxv.x and p.y < maxv.y and p.z < maxv.z):
            ret = 1
        return ret


    @ti.func
    def is_inside(self, index, p):
        ret = 0
        if self.prim_type(index) == PRIM_TRI:
            #https:#www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
            p0 = self.tri_p1(index)
            p1 = self.tri_p2(index)
            p2 = self.tri_p3(index)
            edge0 = p1 - p0
            edge1 = p2 - p0
            n = edge0.cross(edge1)
            pa = p - p0
            #for hard edge, we need a tolerance
            if (pa.dot(n) > -0.01):
                ret = 0
            else:
                ret = 1
        elif self.prim_type(index) == PRIM_SPHERE:
            ret     = 0  
        elif self.prim_type(index) == PRIM_QUAD:
            ret     = 0
        elif self.prim_type(index) == PRIM_BOX:
            minv,maxv = self.aabb(index)
            ret     = self.inside_box(p,minv,maxv )
        return ret

    @ti.func
    def unsigned_distance(self, index, p):
        dis     = INF_VALUE
        closet  = p
        if self.prim_type(index) == PRIM_TRI:
            #https:#www.geometrictools.com/Documentation/DistancePoint3Triangle3.pdf
            p0 = self.tri_p1(index)
            p1 = self.tri_p2(index)
            p2 = self.tri_p3(index)

            diff  = p0 - p
            edge0 = p1 - p0
            edge1 = p2 - p0

            a00 = edge0.dot(edge0)
            a01 = edge0.dot(edge1)
            a11 = edge1.dot(edge1)
            b0 = diff.dot(edge0)
            b1 = diff.dot(edge1)
            det = ti.math.max(a00 * a11 - a01 * a01, 0.0)
            s = a01 * b1 - a11 * b0
            t = a01 * b0 - a00 * b1

            if (s + t <= det):
                if (s < 0.0):
                    if (t < 0.0):  
                        if (b0 < 0.0):
                            t = 0.0
                            if (-b0 >= a00):
                                s = 1.0
                            else:
                                s = -b0 / a00
                        else:
                            s = 0.0
                            if (b1 >= 0.0):
                                t = 0.0
                            elif (-b1 >= a11):
                                t = 1.0
                            else:
                                t = -b1 / a11
                    else:  # region 3
                        s = 0.0
                        if (b1 >= 0.0):
                            t = 0.0
                        elif (-b1 >= a11):
                            t = 1.0
                        else:
                            t = -b1 / a11
                elif (t < 0.0) : # region 5
                    t = 0.0
                    if (b0 >= 0.0):
                        s = 0.0
                    elif (-b0 >= a00):
                        s = 1.0
                    else:
                        s = -b0 / a00
                else : # region 0
                    # minimum at interior point
                    s /= det
                    t /= det
            else:
                tmp0 = 0.0
                tmp1  = 0.0
                numer  = 0.0
                denom = 0.0

                if (s < 0.0):  # region 2
                    tmp0 = a01 + b0
                    tmp1 = a11 + b1
                    if (tmp1 > tmp0):
                        numer = tmp1 - tmp0
                        denom = a00 - 2.0 * a01 + a11
                        if (numer >= denom):
                            s = 1.0
                            t = 0.0
                        else:
                            s = numer / denom
                            t = 1.0 - s
                    else:
                        s = 0.0
                        if (tmp1 <= 0.0):
                            t = 1.0
                        elif (b1 >= 0.0):
                            t = 0.0
                        else:
                            t = -b1 / a11
                elif (t < 0.0) : # region 6
                    tmp0 = a01 + b1
                    tmp1 = a00 + b0
                    if (tmp1 > tmp0):
                        numer = tmp1 - tmp0
                        denom = a00 - 2.0 * a01 + a11
                        if (numer >= denom):
                            t = 1.0
                            s = 0.0
                        else:
                            t = numer / denom
                            s = 1.0 - t
                    else:
                        t = 0.0
                        if (tmp1 <= 0.0):
                            s = 1.0
                        elif (b0 >= 0.0):
                            s = 0.0
                        else:
                            s = -b0 / a00
                else:  # region 1
                    numer = a11 + b1 - a01 - b0
                    if (numer <= 0.0):
                        s = 0.0
                        t = 1.0
                    else:
                        denom = a00 - 2.0 * a01 + a11
                        if (numer >= denom):
                            s = 1.0
                            t = 0.0
                        else:
                            s = numer / denom
                            t = 1.0 - s
            closet = p0 + s * edge0 + t * edge1
            dis= (p  - closet).norm() 
            #print(dis,closet,p0,p1,p2)
        elif self.prim_type(index) == PRIM_SPHERE:
            dis     = INF_VALUE       
        elif self.prim_type(index) == PRIM_QUAD:
            dis     = INF_VALUE 
        elif self.prim_type(index) == PRIM_BOX:
            dis     = INF_VALUE
        return dis,closet

    @ti.kernel
    def print_info(self):
        for i in range(self.primitive_count):
            print(i,self.primitives[i,0],self.tri_p1(i),self.tri_p2(i),self.tri_p3(i))