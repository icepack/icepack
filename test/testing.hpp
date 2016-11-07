
#ifndef ICEPACK_TESTING_HPP
#define ICEPACK_TESTING_HPP

#include <cassert>

#define check(cond)                                                     \
  if (!(cond)) {                                                        \
    std::cerr << "Test " << #cond << std::endl                          \
              << "at   " << __FILE__ << ":" << __LINE__ << std::endl    \
              << "failed." << std::endl;                                \
    abort();                                                            \
  }

#define check_real(val1, val2, tol)                                     \
  {                                                                     \
    const double diff = std::abs(val1 - val2);                          \
    if (diff > tol) {                                                   \
      std::cerr << "|" << #val1 << " - " << #val2 << "| = " << diff     \
                << std::endl                                            \
                << "  > " << #tol << " = " << (tol) << std::endl        \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;   \
      abort();                                                          \
    }                                                                   \
  }

#define check_fields(phi1, phi2, tol)                                   \
  {                                                                     \
    const double diff = icepack::dist(phi1, phi2) / norm(phi2);         \
    if (diff > tol) {                                                   \
      std::cerr << "||" << #phi1 << " - " << #phi2 << "|| "             \
                << "/ ||" << #phi2 << "|| = " << diff << std::endl      \
                << "   > " << #tol << " = " << (tol) << std::endl       \
                << "at " << __FILE__ << ":" << __LINE__ << std::endl;   \
      abort();                                                          \
    }                                                                   \
  }


#include <set>
#include <string>

#include <deal.II/grid/tria.h>

namespace icepack {
  namespace testing {

    std::set<std::string> get_cmdline_args(int argc, char ** argv);

    dealii::Triangulation<2>
    rectangular_glacier(
      double length,
      double width,
      unsigned int num_levels = 5
    );

    bool is_decreasing(const std::vector<double>& seq);
  }
}

#endif
