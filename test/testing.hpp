
#ifndef ICEPACK_TESTING_HPP
#define ICEPACK_TESTING_HPP

#include <set>
#include <string>

namespace icepack {
  namespace testing {

    std::set<std::string> get_cmdline_args(int argc, char ** argv);

  }
}

#endif
