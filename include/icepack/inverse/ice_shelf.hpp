
#ifndef ICEPACK_INVERSE_ICE_SHELF_HPP
#define ICEPACK_INVERSE_ICE_SHELF_HPP

#include <icepack/glacier_models/ice_shelf.hpp>

namespace icepack {
  namespace inverse {

    /**
     * Given an ice shelf, observed velocities and a candidate temperature,
     * return the gradient of the mean-square error with respect to the
     * temperature, i.e. for use in an inverse problem.
     */
    DualField<2> gradient(
      const IceShelf& ice_shelf,
      const Field<2>& thickness,
      const Field<2>& temperature,
      const VectorField<2>& u_observed,
      const Field<2>& sigma
    );

  }
}

#endif