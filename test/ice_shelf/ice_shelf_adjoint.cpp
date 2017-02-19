
#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>
#include "../testing.hpp"

using namespace dealii;
using namespace icepack;

const double rho = rho_ice * (1 - rho_ice / rho_water);
const double temp = 263.15;
const double A = pow(rho * gravity / 4, 3) * rate_factor(temp);

const double u0 = 100;
const double length = 2000;
const double width = 500;
const double h0 = 500;
const double delta_h = 100;


class Thickness : public Function<2>
{
public:
  Thickness() {}

  double value(const Point<2>& x, const unsigned int = 0) const
  {
    return h0 - delta_h/length * x[0];
  }
};


class DeltaTau : public TensorFunction<1, 2>
{
public:
  DeltaTau() {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    Tensor<1, 2> v;
    const double p = x[0] / length;
    v[0] = p * (1.0 - p) * 0.01;
    return v;
  }
};


class Temperature : public Function<2>
{
public:
  Temperature() {}

  double value(const Point<2>&, const unsigned int = 0) const
  {
    return temp;
  }
};

class Velocity : public TensorFunction<1, 2>
{
public:
  Velocity() {}

  Tensor<1, 2> value(const Point<2>& x) const
  {
    const double q = 1 - pow(1 - delta_h * x[0] / (length * h0), 4);

    Tensor<1, 2> v;
    v[0] = u0 + 0.25 * A * q * length * pow(h0, 4) / delta_h;
    v[1] = 0.0;

    return v;
  }
};


int main(int argc, char ** argv)
{
  const bool verbose = argc == 2 &&
    (strcmp(argv[1], "-v") == 0 ||
     strcmp(argv[1], "--verbose") == 0);

  /**
   * Create a model object and input data
   */

  const double length = 2000.0, width = 500.0;
  Triangulation<2> tria = testing::rectangular_glacier(length, width);
  const double dx = dealii::GridTools::minimal_cell_diameter(tria);

  const IceShelf ice_shelf(tria, 1);

  const Field<2> h = ice_shelf.interpolate(Thickness());
  const Field<2> theta = ice_shelf.interpolate(Temperature());
  const VectorField<2> u0 = ice_shelf.interpolate(Velocity());


  /**
   * Solve for the model adjoint around some velocity field
   */

  const DualVectorField<2> tau = ice_shelf.driving_stress(h);
  const DualVectorField<2> r = ice_shelf.derivative(h, theta, u0);
  check(norm(r) / norm(tau) < dx*dx);

  const DualVectorField<2> d_tau = transpose(ice_shelf.interpolate(DeltaTau()));
  const VectorField<2> lambda = ice_shelf.adjoint_solve(h, theta, u0, d_tau);

  if (verbose)
    lambda.write("lambda.ucd", "lambda");

  return 0;
}
