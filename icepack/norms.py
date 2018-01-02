
import numpy as np
from firedrake import sqrt, inner, grad, dx, assemble

def _diameter(mesh):
    X = mesh.coordinates.dat.data
    _, d = np.shape(X)
    Ls = np.array([np.max(X[:, k]) - np.min(X[:, k]) for k in range(d)])
    return np.min(Ls)

def norm(u, norm_type='L2'):
    """Compute the norm of a field

    Computes one of any number of norms of a scalar or vector field. The
    available options are:

    - ``L2``: :math:`\|u\|^2 = \int_\Omega|u|^2dx`

    - ``H01``: :math:`\|u\|^2 = \int_\Omega|\\nabla u|^2dx`

    - ``H1``: :math:`\|u\|^2 = \int_\Omega\\left(|u|^2 + L^2|\\nabla u|^2\\right)dx`

    - ``L1``: :math:`\|u\| = \int_\Omega|u|dx`

    - ``TV``: :math:`\|u\| = \int_\Omega|\\nabla u|dx`

    - ``Linfty``: :math:`\|u\| = \max_{x\in\Omega}|u(x)|`

    The extra factor :math:`L` in the :math:`H^1` norm is the diameter of
    the domain in the infinity metric. This extra factor is included to
    make the norm scale appropriately with the size of the domain.
    """
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

    if norm_type == 'Linfty':
        data = u.dat.data_ro
        if len(data.shape) == 1:
            return np.max(np.abs(data))
        elif len(data.shape) == 2:
            return np.max(np.sqrt(np.sum(data**2, 1)))

    return assemble(form)**(1/p)
