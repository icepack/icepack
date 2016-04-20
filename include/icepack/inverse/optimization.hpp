
#ifndef ICEPACK_INVERSE_OPTIMIZATION_HPP
#define ICEPACK_INVERSE_OPTIMIZATION_HPP

#include <map>

namespace icepack {
  namespace inverse {

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
      const FieldType<rank, dim>& df,
      const FieldType<rank, dim>& p,
      const double eps
    )
    {
      // Compute the angle between the search direction and the gradient of the
      // objective functional; we need this
      const double theta = inner_product(p, df);

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
     * Given a cost functional `F` and a method `dF` to compute its gradient,
     * find an approximate minimizer starting from the guess `phi0`, stopping
     * when the improvement in the cost functional is less than `eps`.
     */
    template <int rank, int dim, typename Functional, typename Gradient>
    FieldType<rank, dim> gradient_descent(
      const Functional& F,
      const Gradient& dF,
      const FieldType<rank, dim>& phi0,
      const double eps
    )
    {
      double cost_old = std::numeric_limits<double>::infinity();
      double cost = F(phi0);

      FieldType<rank, dim> phi(phi0);

      while (std::abs(cost_old - cost) > eps) {
        cost_old = cost;

        // Compute the gradient of the objective functional.
        const FieldType<rank, dim> df = dF(phi);

        // Compute a search direction.
        const FieldType<rank, dim> p = -rms_average(phi) * df / norm(df);

        // Update the current guess.
        phi = line_search(F, phi, df, p, eps);
        cost = F(phi);
      }

      return phi;
    }

  } // End of namespace inverse
} // End of namespace icepack

#endif
