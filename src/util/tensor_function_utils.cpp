
#include <icepack/util/tensor_function_utils.hpp>

namespace icepack {
  namespace internal {

    // Cue unabashed boilerplate.

    CoordFunctions<2> coord_functions(
      const Function<2>& phi0, const Function<2>& phi1
    )
    {
      CoordFunctions<2> coords;
      coords[0] = &phi0;
      coords[1] = &phi1;
      return coords;
    }

    CoordFunctions<3> coord_functions(
      const Function<3>& phi0, const Function<3>& phi1, const Function<3>& phi2
    )
    {
      CoordFunctions<3> coords;
      coords[0] = &phi0;
      coords[1] = &phi1;
      coords[2] = &phi2;
      return coords;
    }

    template class TensorFunctionFromScalarFunctions<2>;
  }
}
