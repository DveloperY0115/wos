import taichi as ti

ti.init(ti.cpu)
v6 = ti.types.vector(6, ti.f32)


@ti.func
def vector_add(a, b):
    for i in ti.static(range(6)):
        a[i] += b[i]
    return a

@ti.kernel
def test_kernel(a: v6, b: v6):
    print(f"add (ti.func): {vector_add(a, b)}")
    print(f"a: {a}")
    for i in ti.static(range(6)):
        a[i] += b[i]
    print(f"add (ti.kernel): {a}")

@ti.kernel
def test_kernel_assign(x: v6):
    if False:
        print(f"x (before): {x}")
        for i in ti.static(range(6)):
            x[i] = i
        print(f"x (after): {x}")
    else:
        test_kernel_assign_func(x)

@ti.func
def test_kernel_assign_func(x: v6):
    print(f"x (before): {x}")
    for i in ti.static(range(6)):
        x[i] = i
    print(f"x (after): {x}")
    
if __name__ == "__main__":
    x = ti.Vector([1, 2, 3, 4, 5, 6], ti.f32)
    y = ti.Vector([1, 1, 1, 1, 1, 1], ti.f32)
    test_kernel(x, y)

    z = ti.Vector([0, 0, 0, 0, 0, 0], ti.f32)
    test_kernel_assign(z)
