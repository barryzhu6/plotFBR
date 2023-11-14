import prop
import numpy as np
from multiprocessing import Pool
from skimage.measure import marching_cubes

N1=40
N2=12
N3=23

f1 = lambda x: prop.visual.sinfbr(x, N1, 3.5, 10.)
f2 = lambda x,y: prop.visual.sphfbr(x, y, N2)

v = np.fromfile('dataNov10/vectest.data', dtype=complex).reshape((N1*N2*N3,-1))[:,14]

def prob(a):
    r, t, p = prop.visual.cart2sph(*a)
    a = f1(r)
    b = f2(t,p)
    return np.sum(v * np.kron(a,b))

if __name__ == "__main__":
    n = 100
    r = 10
    g = np.linspace(-r,r,n)
    points = np.array(np.meshgrid(g,g,g, indexing='ij')).reshape(3,-1).T

    with Pool(4) as p:
        out = p.map(prob, points)
    arr = np.array(out).reshape(n, n, n)
    show = abs(arr)**2
    # show = arr.real
    print(show.min(),show.max())
    step = 2 * r / (n - 1)
    verts, faces, _, _ = marching_cubes(
    show,
    level=0.002,
    spacing=(step, step, step),
    )
    verts -= r
    filename = "isosurface.obj"
    with open(filename, "w") as f:
        f.write("o Isosurface\n")
        for vert_coords in verts:
            x, y, z = vert_coords
            f.write("v %s %s %s \n" % (x, y, z))
        for vert_ids in faces:
            id_1, id_2, id_3 = vert_ids + 1
            f.write("f %s %s %s \n" % (id_1, id_2, id_3))

