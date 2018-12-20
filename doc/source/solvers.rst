
Physics solvers
===============

The solver for the diagnostic equations of glacier flow is one of the most important components of icepack.
Since ice is shear-thinning, the elliptic PDE for the ice velocity is nonlinear.
The diagnostic equations, however, are actually the derivative of a nonlinear scalar functional, the *action*.
Moreover, the action functionals for most glacier models are convex, so the velocity is the unique minimizer.
We can exploit this special structure to define nonlinear solvers that require less hand-tuning than a solver for an arbitrary nonlinear system of equations would.

Let :math:`u`, :math:`h` be the ice velocity and thickness respectively.
The *strain rate tensor* is the symmetrized gradient of the ice velocity:

.. math::
    \dot\varepsilon = \frac{1}{2}\left(\nabla u + \nabla u^\top\right)

The *membrane stress tensor* :math:`M` represents the traction around an infinitesimal area element.
It is defined in terms of the the strain rate:

.. math::
    M = 2\mu(\dot\varepsilon + \text{tr}(\dot\varepsilon)\cdot I)

where :math:`\mu` is the ice viscosity.
The shallow shelf equations are:

.. math::
    \nabla\cdot hM - \frac{1}{2}\varrho g\nabla h^2 = 0

where :math:`\varrho` is the reduced density of ice over seawater and :math:`g` is the acceleration due to gravity.

The *action functional* for the shallow shelf equations is:

.. math::
    E(u) = \int_\Omega\left(\frac{n}{n + 1}h M : \dot\varepsilon - \frac{1}{2}\varrho gh^2\nabla\cdot u\right)dx.

Using the usual methods of the variational calculus, one can show that the shallow shelf equations are the functional derivative of :math:`E`.
Moreover, with a bit more calcuation, one can also show that :math:`E` is *convex* -- the second derivative is positive-definite.
This means that it has a unique minimizer.

Since the solution of the shallow shelf equations minimizes the action, we can make a convergence criterion for our solver based on the value of the action rather than on, say, the norm of the residual.
Internally, icepack uses Newton's method to solve the minimization problem:

.. math::
    u_{k + 1} = u_k - d^2E(u_k)^{-1}dE(u_k).

The *Newton decrement* is the quantity

.. math::
    \Delta = \frac{1}{2}\langle dE(u_k), d^2E(u_k)^{-1}dE(u_k)\rangle.

The Newton decrement is always positive, and it approximates the difference between the current value of the action functional and the minimum value.
By evaluating the ratio of the Newton decrement to that of some positive scale functional -- in icepack we use just the first term of the action -- we can get a dimensionless measure of how much relative improvement there is to be had by doing one more iteration of Newton's method.
When this ratio is sufficiently small, the iteration has converged.
Empirically, I've found that even with a very small tolerances like 10\ :sup:`-12` the method converges in 8-10 iterations.

The nice part about setting a convergence tolerance this way is that the action is an "objective" quantity -- whether you use finite element methods or finite volume methods, piecewise linear or quadratic basis functions, etc., you're always computing the same thing.
The 2-norm of the residual, however, depends on how you discretize the problem.
