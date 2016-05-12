
#ifndef ICEPACK_NUMERICS_OPTIMIZATION_HPP
#define ICEPACK_NUMERICS_OPTIMIZATION_HPP

#include <map>

#include <icepack/field.hpp>

namespace icepack {
  namespace numerics {

    /**
     * This procedure is for computing a bounding interval [a, b] in which to
     * search for the minimum of some functional in a line search.
     *
     * See E. Polak, "Optimization: Algorithms and Consistent Approximations",
     * pp. 30-31 for the Armijo rule.
     */
    template <typename Functional>
    double armijo(
      const Functional& f,
      const double theta,
      const double alpha,
      const double beta
    )
    {
      Assert(alpha < 1.0, ExcInternalError());
      Assert(beta < 1.0, ExcInternalError());

      std::map<int, double> fs;

      // Make a function that memoizes values of the objective functional so
      // that we avoid repeating expensive computations.
      const auto F =
        [&](const int k)
        {
          if (fs.find(k) != fs.end())
            return fs[k];

          fs[k] = f(std::pow(beta, k));
          return fs[k];
        };

      int k = 1;
      const double f0 = f(0);

      while (true) {
        bool decreasing = F(k) - f0 <= std::pow(beta, k) * alpha * theta;
        bool curvature  = F(k - 1) - f0 > std::pow(beta, k - 1) * alpha * theta;

        if (decreasing and curvature) return std::pow(beta, k);

        if (decreasing) k -= 1;
        if (curvature) k += 1;
      }

      return 0;
    }


    /**
     * This procedure finds the minimum of a function of one variable in an
     * interval using golden section search, which is like a bisection method
     * that reduces the interval size by the inverse of the golden ratio at
     * every iteration.
     *
     * The input function is assumed to be convex.
     */
    template <typename Functional>
    double
    golden_section_search(const Functional& f, double a, double b, double eps)
    {
      const double phi = 0.5 * (std::sqrt(5.0) - 1);

      double fa = f(a), fb = f(b);

      while (std::abs(fa - fb) > eps) {
        const double L = b - a;
        const double A = a + L * (1 - phi);
        const double B = b - L * (1 - phi);

        const double fA = f(A), fB = f(B);

        /* There are four cases to consider if `f` is convex. In these two
         * cases, the minimum lies in the interval [A, b]:
         *  ^                 ^
         *  |  o              |  o
         *  |                 |          o
         *  |    o            |    o
         *  |      o          |       o
         *  |          o      |
         *  *------------>    *------------>
         *     a A B   b         a A  B  b
         *
         * and in these two cases, the minimum lies in the interval [a, B].
         * ^                  ^
         * |         o        |         o
         * |                  | o
         * |       o          |       o
         * |     o            |    o
         * | o                |
         * *------------->    *------------>
         *   a   A B b          a  A  B  b
         *
         * These cases are characterized by whether f(A) >= f(B) or vice versa.
         */

        if (fA >= fB) {
          a = A;
          fa = fA;
        } else {
          b = B;
          fb = fB;
        }
      }

      return (a + b) / 2;
    }


    /**
     * Given a cost functional `F`, a starting point `phi`, the derivative `df`
     * of `F` at `phi`, and a search direction `p`, search for the minimum of
     * `F` along `p`.
     */
    template <int rank, int dim, typename Functional>
    FieldType<rank, dim> line_search(
      const Functional& F,
      const FieldType<rank, dim>& phi,
      const FieldType<rank, dim, dual>& df,
      const FieldType<rank, dim, primal>& p,
      const double eps
    )
    {
      // Compute the angle between the search direction and the gradient of the
      // objective functional; we need this
      const double theta = inner_product(df, p);

      // Make a lambda function giving the value of the objective along the
      // search direction.
      const auto f = [&](const double alpha) { return F(phi + alpha * p); };

      // Find an endpoint to bound the line search.
      // TODO: make the parameters 1.0e-4, 0.5 adjustable, but default to the
      // values given in Nocedal & Wright.
      const double end_point = armijo(f, theta, 1.0e-4, 0.5);

      // Locate the minimum of the objective functional along the search,
      // within the bounds obtained using the Armijo rule.
      const double alpha = golden_section_search(f, 0.0, end_point, eps);

      return phi + alpha * p;
    }


    /**
     * Two-loop recursion for L-BFGS
     */
    template <int dim>
    Field<dim> lbfgs_two_loop(
      const DualField<dim>& df,
      const std::vector<Field<dim> >& s,
      const std::vector<DualField<dim> >& y,
      const std::vector<double>& rho,
      const double gamma
    )
    {
      const size_t m = s.size();

      DualField<dim> q(df);
      std::vector<double> alpha(m);

      for (int i = m - 1; i > 0; --i) {
        alpha[i] = rho[i] * inner_product(q, s[i]);
        q = q - alpha[i] * y[i];
      }

      Field<dim> r = gamma * transpose(q);
      for (unsigned int i = 0; i < m; ++i) {
        const double beta = rho[i] * inner_product(y[i], r);
        r = r + (alpha[i] - beta) * s[i];
      }

      return r;
    }


    /**
     * Given a cost functional `F` and a method `dF` to compute its gradient,
     * find an approximate minimizer starting from the guess `phi_start`,
     * stopping when the improvement in the cost functional is less than `eps`.
     * This uses the limited-memory Broyden-Fletcher-Goldfarb-Shanno (BFGS)
     * algorithm with the previous `m` steps.
     */
    template <int dim, typename Functional, typename Gradient>
    Field<dim> lbfgs(
      const Functional& F,
      const Gradient& dF,
      const Field<dim>& phi_start,
      const unsigned int m,
      const double eps
    )
    {
      double cost_old = std::numeric_limits<double>::infinity();
      double cost = F(phi_start);

      Field<dim> phi(phi_start);
      DualField<dim> df = dF(phi_start);
      const auto& discretization = phi.get_discretization();

      // Create vectors for storing the last few differences of the guesses,
      // differences of the gradients, and the inverses of their inner products
      std::vector<Field<dim> > s(m, Field<dim>(discretization));
      std::vector<DualField<dim> > y(m, DualField<dim>(discretization));
      std::vector<double> rho(m);

      // As an initial guess, we will assume that the Hessian inverse is a
      // multiple of the inverse of the mass matrix
      double gamma = 1.0;

      for (unsigned int k = 0; std::abs(cost_old - cost) > eps; ++k) {
        const Field<dim> p = -lbfgs_two_loop(df, s, y, rho, gamma);
        const Field<dim> phi1 = line_search(F, phi, df, p, eps);
        const DualField<dim> df1 = dF(phi1);

        s[0] = phi1 - phi;
        y[0] = df1 - df;

        const double cos_angle = inner_product(y[0], s[0]);
        const double norm_y = norm(y[0]);
        rho[0] = 1.0 / cos_angle;
        gamma = cos_angle / (norm_y * norm_y);

        std::rotate(s.begin(), s.begin() + 1, s.end());
        std::rotate(y.begin(), y.begin() + 1, y.end());
        std::rotate(rho.begin(), rho.begin() + 1, rho.end());

        phi = phi1;
        df = df1;
        cost_old = cost;
        cost = F(phi);
      }

      return phi;
    }

  } // End of namespace numerics
} // End of namespace icepack

#endif
