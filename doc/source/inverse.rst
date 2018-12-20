
Inverse solvers
===============

The goal of inverse solvers is to estimate an unobservable field :math:`\theta`, like the basal friction coefficient, from observable fields like the ice velocity :math:`u^o` and thickness :math:`h`.
There are three components necessary for formulating the estimation problem:

- the *forward map* :math:`G` which maps the parameters :math:`\theta` to the velocity :math:`u`,
- the *model-data misfit* functional :math:`E(u - u^o)`, which quantifies the extent to which the computed velocity :math:`u` differs from observations,
- and the *regularization* functional :math:`R(\theta)` which penalizes spurious oscillations in the inferred parameter.

These are all combined into one *objective functional*

.. math::
    J(\theta) = E(G(\theta) - u^o) + R(\theta).

We can approximate the real value of the parameters by finding the minimizer of the objective.

Most glacier flow modeling packages use the BFGS algorithm to solve this optimization problem.
In practice, the BFGS algorithm can converge slowly or even fail without some degree of hand-tuning, which is often difficult for users that are not experts in numerical optimization.
Instead, icepack uses the *Gauss-Newton* approximation to the second derivative:

.. math::
    d^2J(\theta) \approx \frac{dG}{d\theta}^*\cdot \frac{d^2E}{du^2}\cdot \frac{dG}{d\theta} + \frac{d^2R}{d\theta^2} \equiv H.

This operator is symmetric and positive-definite, so the search direction

.. math::
    \phi = -H^{-1}\frac{dJ}{d\theta}

is a descent direction for :math:`J`.
I've found that using the Gauss-Newton operator to define a search direction is, in every case, faster and more reliable than BFGS.

If you read the source code for the Gauss-Newton solver, some of the decisions might seem a bit opaque, so some commentary on the implementation is in order.
How you go about solving the linear system for the search direction :math:`\phi` can affect the overall performance and robustness of the algorithm.
The operator :math:`H` is dense, so forming it explicitly is out of the question.
To use the conjugate gradient algorithm, however, it's enough to be able to multiply :math:`H` by a vector.
Taking the preconditioner :math:`M` to be the sum of the finite element mass matrix and :math:`d^2R` has been sufficient so far -- the solver converges in 20-40 iterations even for large meshes.

The usual conjugate gradient method as found in, say, Trefethen and Bau relies on certain quantities being positive, but this can fail in finite-precision arithmetic.
We're using a slight variation on the CG algorithm that offers a better guarantee of positivity.
First, we compute the preconditioned residual :math:`z` and search direction :math:`p`:

.. math::
    \begin{align}
    z_0 & = M^{-1}(dJ - H\phi_0), \\
    p_0 & = z_0.
    \end{align}

Until the algorithm has converged, we update as follows:

.. math::
    \begin{align}
    \alpha_k & = \langle Mz_k, z_k\rangle / \langle Hp_k, p_k\rangle \\
    \phi_{k + 1} & = \phi_k + \alpha_k p_k \\
    z_{k + 1} & = z_k - \alpha_k M^{-1}Hp_k \\
    \beta_{k + 1} & = \langle Mz_{k + 1}, z_{k + 1}\rangle / \langle Mz_k, z_k\rangle \\
    p_{k + 1} & = \beta_k p_k + z_{k + 1}
    \end{align}

The inner products can all be assembled in such a way to guarantee positivity.
We can use decrease in the quantity

.. math::
    \mathscr{J}(\phi) = \frac{1}{2}\langle H\phi, \phi\rangle - \langle dJ, \phi\rangle

relative to the :math:`H`-norm of :math:`\phi` as an "objective" convergence criterion.
