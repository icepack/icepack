
import numpy as np
from firedrake import sqrt, inner, grad, dx, assemble

def _diameter(mesh):
    X = mesh.coordinates.dat.data
    _, d = np.shape(X)
    Ls = np.array([np.max(X[:, k]) - np.min(X[:, k]) for k in range(d)])
    return np.min(Ls)

def norm(u, norm_type='L2'):
    if norm_type == 'L2':
        form, p = inner(u, u) * dx, 2

    if norm_type == 'H01':
        form, p = inner(grad(u), grad(u)) * dx, 2

    if norm_type == 'H1':
        mesh = u.ufl_domain()
        #TODO: smarter way of getting a scale for the H1-norm
        L = _diameter(mesh)
        form, p = inner(u, u) * dx + L**2 * inner(grad(u), grad(u)) * dx, 2

    if norm_type == 'L1':
        form, p = sqrt(inner(u, u)) * dx, 1

    if norm_type == 'TV':
        form, p = sqrt(inner(grad(u), grad(u))) * dx, 1

   #TODO: L-infinity norm

    return assemble(form)**(1/p)
