
#include <map>
#include <random>

#include <deal.II/grid/grid_generator.h>

#include <icepack/physics/constants.hpp>
#include <icepack/physics/viscosity.hpp>
#include <icepack/numerics/optimization.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>
#include <icepack/inverse/error_functionals.hpp>
#include <icepack/inverse/regularization.hpp>
#include <icepack/inverse/ice_shelf.hpp>

using dealii::Tensor;
using dealii::Point;
using dealii::Function;
using dealii::ConstantFunction;

using icepack::Discretization;
using icepack::FieldType;
using icepack::Field;
using icepack::DualField;
using icepack::VectorField;
using icepack::rho_ice;
using icepack::rho_water;
using icepack::gravity;

namespace numerics = icepack::numerics;
namespace inverse = icepack::inverse;
using inverse::Regularizer;
using inverse::SquareGradient;
using inverse::TotalVariation;


// Some physical constants
extern const double rho, temp, delta_temp, A, u0, length, width, h0, delta_h;


// A synthetic velocity field. This will be the initial guess for the velocity.
// It is the exact solution for the ice velocity when the ice thickness
// is linear and the ice temperature is constant. The true velocities will be
// synthesized by computing the velocity field for a non-constant temperature.
class Velocity : public dealii::TensorFunction<1, 2>
{
public:
  Velocity() = default;
  Tensor<1, 2> value(const Point<2>& x) const;
};


// A synthetic thickness field
class Thickness : public Function<2>
{
public:
  Thickness() = default;
  double value(const Point<2>& x, const unsigned int = 0) const;
};


// A smooth synthetic temperature profile which is parabolic in each direction.
// This temperature is easy to recover for using any regularization method.
class ParabolicTemperature : public Function<2>
{
public:
  ParabolicTemperature() = default;
  double value(const Point<2>& x, const unsigned int = 0) const;
};


// A synthetic temperature profile which is discontinuous in the across-flow
// direction. This profile is easy to recover using total variation or
// anisotropic smoothing, but more difficult with the square gradient.
class AlongFlowTemperature : public Function<2>
{
public:
  AlongFlowTemperature() = default;
  double value(const Point<2>& x, const unsigned int = 0) const;
};


// A synthetic temperature profile with discontinuities both along- and across-
// flow. This profile is easy to recover only with total variation filtering.
class DiscontinuousTemperature : public Function<2>
{
public:
  DiscontinuousTemperature() = default;
  double value(const Point<2>& x, const unsigned int = 0) const;
};


// Perturb a field with Gaussian noise with standard deviation `sigma`
template <int rank, int dim>
FieldType<rank, dim>
add_noise(const FieldType<rank, dim>& phi, const double sigma)
{
  FieldType<rank, dim> psi(phi);

  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<> z(0.0, sigma);

  for (auto& v: psi.get_coefficients())
    v += z(gen);

  return psi;
}


// A regularization functional which just returns 0.0.
class NullRegularizer : public inverse::Regularizer<2>
{
public:
  NullRegularizer() = default;
  double operator()(const Field<2>&) const;
  DualField<2> derivative(const Field<2>&) const;
  Field<2> filter(const Field<2>&, const DualField<2>& f) const;
};


// A function for getting all the command-line arguments
std::map<std::string, std::string> get_cmdline_args(int argc, char ** argv);


// Print out usage information for the program
void help();
