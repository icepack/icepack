
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

  }
}

#endif
