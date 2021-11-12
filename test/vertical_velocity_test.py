# Copyright (C) 2019-2020 by Daniel Shapero <shapero@uw.edu>
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
import firedrake
import icepack
import icepack.utilities
from icepack.constants import (ice_density as ρ_I, water_density as ρ_W,
                               gravity as g, glen_flow_law as n)



def test_vertical_velocity():

    Lx, Ly = 20e3, 20e3
    nx, ny = 48, 48

    mesh2d = firedrake.RectangleMesh(nx, ny, Lx, Ly)
    mesh = firedrake.ExtrudedMesh(mesh2d, layers=1)
    Q = firedrake.FunctionSpace(mesh, family='CG', degree=2, vfamily='DG', vdegree=0)
    Q3D = firedrake.FunctionSpace(mesh, family='DG', degree=2, vfamily='GL', vdegree=6)
    V = firedrake.VectorFunctionSpace(mesh, dim=2, family='CG', degree=2, vfamily='GL', vdegree=5)
    # note we should call the families in the vertical velocity solver.

    x, y, ζ = firedrake.SpatialCoordinate(mesh)

    u_inflow=1.0
    v_inflow=2.0
    mu=.003
    mv=.001

    b=firedrake.interpolate(firedrake.Constant(0.0),Q)
    s=firedrake.interpolate(firedrake.Constant(1000.0),Q)
    h = firedrake.interpolate(s-b,Q)
    u=firedrake.interpolate(firedrake.as_vector((mu*x+u_inflow,mv*y+v_inflow)),V)

    m=-0.01

    def analytic_vertical_velocity(h,ζ,mu,mv,m,Q3D):
	    return firedrake.interpolate(firedrake.Constant(m)-(firedrake.Constant(mu+mv)*h*ζ),Q3D)

    w=firedrake.interpolate(icepack.utilities.vertical_velocity(u,h,m=m)*h,Q3D)
    w_analytic=analytic_vertical_velocity(h,ζ,mu,mv,m,Q3D)


    assert(np.mean(np.abs(w.dat.data-w_analytic.dat.data))<10e-9)
