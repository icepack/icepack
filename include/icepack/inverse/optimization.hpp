
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
     * Given a cost functional `F` and a method `dF` to compute its gradient,
     * find an approximate minimizer starting from the guess `phi0`, stopping
     * when the improvement in the cost functional is less than `eps`.
     */
    template <typename T, typename Functional, typename Gradient>
    T gradient_descent(
      const Functional& F,
      const Gradient& dF,
      const T& phi0,
      const double eps
    )
    {
      double cost_old = std::numeric_limits<double>::infinity();
      double cost = F(phi0);

      T phi(phi0);

      while (std::abs(cost_old - cost) > eps) {
        cost_old = cost;

        // Compute the gradient of the objective functional.
        const auto df = dF(phi);

        // Compute a search direction.
        const auto p = -rms_average(phi) * df / norm(df);

        // Compute the inner product of the gradient of the objective
        // functional and the search direction.
        const double theta = -rms_average(phi) * norm(df);

        // Make a lambda function for computing the value of the objective
        // functional along the line starting at `u` in the direction `p`
        const auto f = [&](const double alpha) { return F(phi + alpha * p); };

        // Find a bounding interval in which to perform a line search.
        const double end_point = armijo(f, theta, 1.0e-4, 0.5);

        // Locate the minimum within the bounding interval.
        const double alpha = golden_section_search(f, 0.0, end_point, eps);

        // Update the current guess.
        phi = phi + alpha * p;
        cost = F(phi);
      }

      return phi;
    }

  } // End of namespace inverse
} // End of namespace icepack

#endif
