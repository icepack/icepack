
#ifndef ICEPACK_INVERSE_OPTIMIZATION_HPP
#define ICEPACK_INVERSE_OPTIMIZATION_HPP

#include <map>

namespace icepack {
  namespace inverse {

    template <typename Functional>
    double armijo(
      const Functional& f,
      const double theta,
      const double alpha,
      const double beta
    )
    {
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

        // TODO: picture explaining this and assert that it's not bizarre

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

  }
}

#endif
