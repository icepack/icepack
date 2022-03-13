# Copyright (C) 2020-2021 by Daniel Shapero <shapero@uw.edu>
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

r"""Solvers for ice physics models"""

import firedrake
from firedrake import dx, inner, Constant
from icepack.optimization import MinimizationProblem, NewtonSolver
from ..utilities import default_solver_parameters
from icepack.calculus import grad, div, FacetNormal


class FlowSolver:
    def __init__(self, model, **kwargs):
        r"""Solves the diagnostic and prognostic models of ice physics

        This class is responsible for efficiently solving the physics
        problem you have chosen. (This is contrast to classes like
        IceStream, which is where you choose what that physics problem
        is.) If you want to make your simulation run faster, you can select
        different solvers and options.

        Parameters
        ----------
        model
            The flow model object -- IceShelf, IceStream, etc.
        dirichlet_ids : list of int, optional
            Numerical IDs of the boundary segments where the ice velocity
            should be fixed
        side_wall_ids : list of int, optional
            Numerical IDs of the boundary segments where the ice velocity
            should have no normal flow
        diagnostic_solver_type : {'icepack', 'petsc'}, optional
            Use hand-written optimization solver ('icepack') or PETSc SNES
            ('petsc'), defaults to 'icepack'
        diagnostic_solver_parameters : dict, optional
            Options for the diagnostic solver; defaults to a Newton line
            search method with direct factorization of the Hessian using
            MUMPS
        prognostic_solver_type : {'lax-wendroff', 'implicit-euler'}, optional
            Timestepping scheme to use for prognostic equations, defaults
            to Lax-Wendroff
        prognostic_solver_parameters : dict, optional
            Options for prognostic solve routine; defaults to direct
            factorization of the flux matrix using MUMPS

        Examples
        --------

        Create a flow solver with inflow on boundary segments 1 and 2
        using the default solver configuration.

        >>> model = icepack.models.IceStream()
        >>> solver = icepack.solvers.FlowSolver(model, dirichlet_ids=[1, 2])

        Use an iterative linear solver to hopefully accelerate the code.

        >>> opts = {
        ...     'dirichlet_ids': [1, 2],
        ...     'diagnostic_solver_type': 'petsc',
        ...     'diagnostic_solver_parameters': {
        ...         'ksp_type': 'cg',
        ...         'pc_type': 'ilu',
        ...         'pc_factor_fill': 2
        ...     },
        ...     'prognostic_solver_parameters': {
        ...         'ksp_type': 'gmres',
        ...         'pc_type': 'sor'
        ...     }
        ... }
        >>> solver = icepack.solvers.FlowSolver(model, **opts)
        """
        self._model = model
        self._fields = {}

        # Prepare the diagnostic solver
        diagnostic_parameters = kwargs.get(
            "diagnostic_solver_parameters", default_solver_parameters
        )

        if "diagnostic_solver_type" in kwargs.keys():
            solver_type = kwargs["diagnostic_solver_type"]
            if isinstance(solver_type, str):
                solvers_dict = {"icepack": IcepackSolver, "petsc": PETScSolver}
                solver_type = solvers_dict[solver_type]
        else:
            solver_type = IcepackSolver

        self._diagnostic_solver = solver_type(
            self.model,
            self._fields,
            diagnostic_parameters,
            dirichlet_ids=kwargs.pop("dirichlet_ids", []),
            side_wall_ids=kwargs.pop("side_wall_ids", []),
        )

        # Prepare the prognostic solver
        prognostic_parameters = kwargs.get(
            "prognostic_solver_parameters", default_solver_parameters
        )

        if "prognostic_solver_type" in kwargs.keys():
            solver_type = kwargs["prognostic_solver_type"]
            if isinstance(solver_type, str):
                solvers_dict = {
                    "implicit-euler": ImplicitEuler,
                    "lax-wendroff": LaxWendroff,
                }
                solver_type = solvers_dict[solver_type]
        else:
            solver_type = LaxWendroff

        self._prognostic_solver = solver_type(
            self.model.continuity, self._fields, prognostic_parameters
        )

    @property
    def model(self):
        r"""The physics model that this object solves"""
        return self._model

    @property
    def fields(self):
        r"""Dictionary of all fields that are part of the simulation"""
        return self._fields

    def diagnostic_solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the ice velocity"""
        return self._diagnostic_solver.solve(**kwargs)

    def prognostic_solve(self, dt, **kwargs):
        r"""Solve the prognostic model physics for the new value of the ice
        thickness"""
        return self._prognostic_solver.solve(dt, **kwargs)


class IcepackSolver:
    def __init__(
        self, model, fields, solver_parameters, dirichlet_ids=[], side_wall_ids=[]
    ):
        r"""Diagnostic solver implementation using hand-written Newton line
        search optimization algorithm"""
        self._model = model
        self._fields = fields
        self._solver_parameters = solver_parameters.copy()
        self._tolerance = self._solver_parameters.pop("tolerance", 1e-12)
        self._max_iterations = self._solver_parameters.pop("max_iterations", 50)
        self._dirichlet_ids = dirichlet_ids
        self._side_wall_ids = side_wall_ids

    def setup(self, **kwargs):
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError(
                        "Input %s field has type %s, must be Constant or Function!"
                        % (name, type(field))
                    )

        # Create homogeneous BCs for the Dirichlet part of the boundary
        u = self._fields["velocity"]
        V = u.function_space()
        # NOTE: This will have to change when we do Stokes!
        if hasattr(V._ufl_element, "_sub_element"):
            bcs = firedrake.DirichletBC(V, Constant((0, 0)), self._dirichlet_ids)
        else:
            bcs = firedrake.DirichletBC(V, Constant(0), self._dirichlet_ids)
        if not self._dirichlet_ids:
            bcs = None

        # Find the numeric IDs for the ice front
        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        ice_front_ids_comp = set(self._dirichlet_ids + self._side_wall_ids)
        ice_front_ids = list(set(boundary_ids) - ice_front_ids_comp)

        # Create the action and scale functionals
        _kwargs = {"side_wall_ids": self._side_wall_ids, "ice_front_ids": ice_front_ids}
        action = self._model.action(**self._fields, **_kwargs)
        scale = self._model.scale(**self._fields, **_kwargs)

        # Set up a minimization problem and solver
        quadrature_degree = self._model.quadrature_degree(**self._fields)
        params = {"quadrature_degree": quadrature_degree}
        problem = MinimizationProblem(action, scale, u, bcs, params)
        self._solver = NewtonSolver(
            problem,
            self._tolerance,
            solver_parameters=self._solver_parameters,
            max_iterations=self._max_iterations,
        )

    def solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the ice velocity"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self._fields[name].assign(field)

        # Solve the minimization problem and return the velocity field
        self._solver.solve()
        u = self._fields["velocity"]
        return u.copy(deepcopy=True)


class PETScSolver:
    def __init__(
        self, model, fields, solver_parameters, dirichlet_ids=[], side_wall_ids=[]
    ):
        r"""Diagnostic solver implementation using PETSc SNES"""
        self._model = model
        self._fields = fields
        self._solver_parameters = solver_parameters
        self._dirichlet_ids = dirichlet_ids
        self._side_wall_ids = side_wall_ids

    def setup(self, **kwargs):
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError(
                        "Input %s field has type %s, must be Constant or Function!"
                        % (name, type(field))
                    )

        # Create homogeneous BCs for the Dirichlet part of the boundary
        u = self._fields["velocity"]
        V = u.function_space()
        bcs = firedrake.DirichletBC(V, u, self._dirichlet_ids)
        if not self._dirichlet_ids:
            bcs = None

        # Find the numeric IDs for the ice front
        boundary_ids = u.ufl_domain().exterior_facets.unique_markers
        ice_front_ids_comp = set(self._dirichlet_ids + self._side_wall_ids)
        ice_front_ids = list(set(boundary_ids) - ice_front_ids_comp)

        # Create the action and scale functionals
        _kwargs = {"side_wall_ids": self._side_wall_ids, "ice_front_ids": ice_front_ids}
        action = self._model.action(**self._fields, **_kwargs)
        F = firedrake.derivative(action, u)

        quad_degree = self._model.quadrature_degree(**self._fields)
        problem = firedrake.NonlinearVariationalProblem(
            F, u, bcs, form_compiler_parameters={"quadrature_degree": quad_degree}
        )
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

    def solve(self, **kwargs):
        r"""Solve the diagnostic model physics for the ice velocity"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                if isinstance(field, firedrake.Function):
                    self._fields[name].assign(field)

        # Solve the minimization problem and return the velocity field
        self._solver.solve()
        u = self._fields["velocity"]
        return u.copy(deepcopy=True)


class ImplicitEuler:
    def __init__(self, continuity, fields, solver_parameters):
        r"""Prognostic solver implementation using the 1st-order, backward
        Euler timestepping scheme

        This solver is included for backward compatibility only. We do not
        recommend it and the Lax-Wendroff scheme is preferable by far.
        """
        self._continuity = continuity
        self._fields = fields
        self._solver_parameters = solver_parameters

    def setup(self, **kwargs):
        r"""Create the internal data structures that help reuse information
        from past prognostic solves"""
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError(
                        "Input %s field has type %s, must be Constant or Function!"
                        % (name, type(field))
                    )

        dt = firedrake.Constant(1.0)
        dh_dt = self._continuity(dt, **self._fields)
        h = self._fields["thickness"]
        h_0 = h.copy(deepcopy=True)
        q = firedrake.TestFunction(h.function_space())
        F = (h - h_0) * q * dx - dt * dh_dt

        problem = firedrake.NonlinearVariationalProblem(F, h)
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

        self._thickness_old = h_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        r"""Compute the thickness evolution after time `dt`"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self._fields[name].assign(field)

        h = self._fields["thickness"]
        self._thickness_old.assign(h)
        self._timestep.assign(dt)
        self._solver.solve()
        return h.copy(deepcopy=True)


class LaxWendroff:
    def __init__(self, continuity, fields, solver_parameters):
        r"""Prognostic solver implementation using the 2st-order implicit
        Lax-Wendroff timestepping scheme

        This method introduces additional diffusion along flowlines compared
        to the implicit Euler scheme. This tends to reduce the magnitude of
        possible spurious oscillations.
        """
        self._continuity = continuity
        self._fields = fields
        self._solver_parameters = solver_parameters

    def setup(self, **kwargs):
        r"""Create the internal data structures that help reuse information
        from past prognostic solves"""
        for name, field in kwargs.items():
            if name in self._fields.keys():
                self._fields[name].assign(field)
            else:
                if isinstance(field, firedrake.Constant):
                    self._fields[name] = firedrake.Constant(field)
                elif isinstance(field, firedrake.Function):
                    self._fields[name] = field.copy(deepcopy=True)
                else:
                    raise TypeError(
                        "Input %s field has type %s, must be Constant or Function!"
                        % (name, type(field))
                    )

        dt = firedrake.Constant(1.0)
        h = self._fields["thickness"]
        u = self._fields["velocity"]
        h_0 = h.copy(deepcopy=True)

        Q = h.function_space()
        mesh = Q.mesh()
        n = FacetNormal(mesh)
        outflow = firedrake.max_value(0, inner(u, n))
        inflow = firedrake.min_value(0, inner(u, n))

        # Additional streamlining terms that give 2nd-order accuracy
        q = firedrake.TestFunction(Q)
        ds = firedrake.ds if mesh.layers is None else firedrake.ds_v
        flux_cells = -div(h * u) * inner(u, grad(q)) * dx
        flux_out = div(h * u) * q * outflow * ds
        flux_in = div(h_0 * u) * q * inflow * ds
        d2h_dt2 = flux_cells + flux_out + flux_in

        dh_dt = self._continuity(dt, **self._fields)
        F = (h - h_0) * q * dx - dt * (dh_dt + 0.5 * dt * d2h_dt2)

        problem = firedrake.NonlinearVariationalProblem(F, h)
        self._solver = firedrake.NonlinearVariationalSolver(
            problem, solver_parameters=self._solver_parameters
        )

        self._thickness_old = h_0
        self._timestep = dt

    def solve(self, dt, **kwargs):
        r"""Compute the thickness evolution after time `dt`"""
        if not hasattr(self, "_solver"):
            self.setup(**kwargs)
        else:
            for name, field in kwargs.items():
                self._fields[name].assign(field)

        h = self._fields["thickness"]
        self._thickness_old.assign(h)
        self._timestep.assign(dt)
        self._solver.solve()
        return h.copy(deepcopy=True)
