
#include <set>
#include <string>

namespace icepack {
  namespace testing {

    std::set<std::string> get_cmdline_args(int argc, char ** argv)
    {
      std::set<std::string> args;
      for (int k = 0; k < argc; ++k)
        args.insert(std::string(argv[k]));

      return args;
    }

  }
}
