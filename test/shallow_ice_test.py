# Copyright (C) 2020 by Jessica Badgeley <badgeley@uw.edu>
#
# This file is part of icepack.
#
# icepack is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# The full text of the license can be found in the file LICENSE in the
# icepack source directory or at <http://www.gnu.org/licenses/>.

import numpy as np
import pygmsh
import os
import matplotlib.pyplot as plt
import firedrake
from firedrake import inner, grad, interpolate, max_value, assemble
import icepack
from icepack.constants import (
    ice_density as ρ_I,
    glen_flow_law as n,
    gravity as g
)

# Test our numerical solvers against this analytical solution.
R = 100e3 # Radius of ice sheet in meters
R_mesh = R * .75 # Radius of mesh (set to <1 to ignore interpolation
                 # errors at the edge)
alpha = firedrake.Constant(R)
T = firedrake.Constant(254.15)
A = icepack.rate_factor(T)
A0 = 2 * A * (ρ_I * g) ** n / (n + 2)

def make_mesh(R, δx):
    geometry = pygmsh.built_in.Geometry()
    x1 = geometry.add_point([-R, 0, 0], lcar=δx)
    x2 = geometry.add_point([+R, 0, 0], lcar=δx)
    center1 = geometry.add_point([0, 0, 0,], lcar=δx)
    arcs = [geometry.add_circle_arc(x1, center1, x2),
            geometry.add_circle_arc(x2, center1, x1)]
    line_loop = geometry.add_line_loop(arcs)
    plane_surface = geometry.add_plane_surface(line_loop)
    physical_lines = [geometry.add_physical(arc) for arc in arcs]
    physical_surface = geometry.add_physical(plane_surface)
    with open('shallow-ice.geo', 'w') as geo_file:
        geo_file.write(geometry.get_code())
    transform_file = "gmsh -v 0 -2 -format msh2 -o shallow-ice.msh shallow-ice.geo"
    os.system(transform_file)
    mesh = firedrake.Mesh('shallow-ice.msh')
    remove_geo = "rm shallow-ice.geo"
    remove_mesh = "rm shallow-ice.msh"
    os.system(remove_geo)
    os.system(remove_mesh)
    return mesh


# Pick a geometry in order to have an exact solution. We'll use the
# Bueler profile from section 5.6.3 in Greve and Blatter (2009).
# Citation: Greve, Ralf, and Heinz Blatter. Dynamics of ice sheets
# and glaciers. Springer Science & Business Media, 2009.
def Bueler_profile(mesh, R):
    x, y = firedrake.SpatialCoordinate(mesh)
    r = firedrake.sqrt(x**2 + y**2)
    h_divide = (2 * R * (alpha/A0)**(1/n) * (n-1)/n)**(n/(2*n+2))
    h_part2 = (n+1)*(r/R) - n*(r/R)**((n+1)/n) + n*(max_value(1-(r/R),0))**((n+1)/n) - 1
    h_expr = (h_divide/((n-1)**(n/(2*n+2)))) * (max_value(h_part2,0))**(n/(2*n+2))
    return h_expr


def exact_u(h_expr, Q):
    h = interpolate(h_expr,Q)
    u_exact = -A0 * h**(n + 1) * inner(grad(h), grad(h)) * grad(h)
    return u_exact


def norm(v):
    return icepack.norm(v, norm_type='L2')


def test_diagnostic_solver_convergence():
    model = icepack.models.ShallowIce()

    for degree in range(1, 4):
        delta_x, error = [], []
        for N in range(10, 110 - 20 * (degree - 1), 10):
            mesh = make_mesh(R_mesh, R / N)

            Q = firedrake.FunctionSpace(mesh, 'CG', degree)
            V = firedrake.VectorFunctionSpace(mesh, 'CG', degree)

            h_expr = Bueler_profile(mesh, R)
            u_exact = interpolate(exact_u(h_expr, Q), V)

            h = interpolate(h_expr, Q)
            s = interpolate(h_expr, Q)
            u = firedrake.Function(V)

            solver = icepack.solvers.FlowSolver(model)
            u_num = solver.diagnostic_solve(
                velocity=u,
                thickness=h,
                surface=s,
                fluidity=A
            )
            error.append(norm(u_exact - u_num) / norm(u_exact))
            delta_x.append(R / N)

            print(delta_x[-1], error[-1])

            assert assemble(model.scale(velocity=u_num)) > 0

        log_delta_x = np.log2(np.array(delta_x))
        log_error = np.log2(np.array(error))
        slope, intercept = np.polyfit(log_delta_x, log_error, 1)

        print('log(error) ~= {:g} * log(dx) + {:g}'.format(slope, intercept))
        assert slope > 0.9
