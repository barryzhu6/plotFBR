import prop
import numpy as np
import plotly.graph_objects as go
from multiprocessing import Pool
N1=40
N2=12
N3=23

f1 = lambda x: prop.visual.sinfbr(x, N1, 3.5, 10.)
f2 = lambda x,y: prop.visual.sphfbr(x, y, N2)

v = np.fromfile('dataNov10/vectest.data', dtype=complex).reshape((N1*N2*N3,-1))[:,12]

def prob(a):
    r, t, p = prop.visual.cart2sph(*a)
    a = f1(r)
    b = f2(t,p)
    return np.sum(v * np.kron(a,b))

if __name__ == "__main__":
    n = 50
    r = 10
    g = np.linspace(-r,r,n)
    points = np.array(np.meshgrid(g,g,g, indexing='ij')).reshape(3,-1).T

    with Pool(4) as p:
        out = p.map(prob, points)
    arr = np.array(out).reshape(n, n, n)
    show = abs(arr)**2
    # show = arr.real
    s1, s2, s3 = np.meshgrid(g, g, g, indexing='ij')
    fig = go.Figure(data=go.Volume(
        x=s1.flatten(),
        y=s2.flatten(),
        z=s3.flatten(),
        value=show.flatten(),
        isomin=0.001,
        isomax=show.max(),
        opacity=0.1, # needs to be small to see through all surfaces
        surface_count=30, # needs to be a large number for good volume rendering
    ))
    fig.show()