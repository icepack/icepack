
#include <icepack/physics/constants.hpp>
#include <icepack/numerics/optimization.hpp>
#include <icepack/inverse/error_functionals.hpp>
#include <icepack/inverse/ice_shelf.hpp>
#include "../testing.hpp"

using dealii::Tensor;
using dealii::Point;
using dealii::ConstantFunction;
using icepack::Field;
using icepack::DualField;
using icepack::VectorField;
using icepack::DualVectorField;
using icepack::rho_ice;
using icepack::rho_water;
using icepack::gravity;

// Some physical constants
const double rho = rho_ice * (1 - rho_ice / rho_water);
const double temp = 262.15;
const double delta_temp = -10.0;
const double A = pow(rho * gravity / 4, 3) * icepack::rate_factor(temp);

const double u0 = 100.0;
const double length = 20000.0, width = 20000.0;
const double h0 = 600.0;
const double delta_h = 300.0;

// A type alias to save us from the Teutonic tendency to the verbose
template <int dim>
using Fn = dealii::ScalarFunctionFromFunctionObject<dim>;


// A synthetic velocity field; the data we will be using to invert for the
// temperature
class Velocity : public dealii::TensorFunction<1, 2>
{
public:
  Velocity() = default;

  Tensor<1, 2> value(const Point<2>& x) const
  {
    const double q = 1 - pow(1 - delta_h * x[0] / (length * h0), 4);

    Tensor<1, 2> v; v[1] = 0.0;
    v[0] = u0 + 0.25 * A * q * length * pow(h0, 4) / delta_h;

    return v;
  }
};


int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("-v") || args.count("--verbose");

  const Fn<2> temperature([&](const Point<2>& x)
                          {
                            const double X = x[0] / length;
                            const double Y = x[1] / width;
                            const double q = X * (1 - X) * Y * (1 - Y);
                            return temp + 4 * q * delta_temp;
                          });

  const Fn<2> thickness([&](const Point<2>& x)
                        {
                          const double X = x[0] / length;
                          return h0 - delta_h * X;
                        });


  /* ----------------------------------------------
   * Make a model object and interpolate exact data
   * ---------------------------------------------- */
  auto tria = icepack::testing::rectangular_glacier(length, width);
  icepack::IceShelf ice_shelf(tria, 1);
  const auto& discretization = ice_shelf.get_discretization();

  const Field<2> h = ice_shelf.interpolate(thickness);

  // Our first guess is that the temperature field is a constant.
  const Field<2> theta_guess = ice_shelf.interpolate(ConstantFunction<2>(temp));

  // For a constant temperature field and a thickness field of constant slope,
  // there's an exact solution for the ice velocity, which we will take as our
  // initial guess.
  const VectorField<2> u_guess = ice_shelf.interpolate(Velocity());

  // Make the true temperature field, which is what we're trying to invert for;
  // in real life of course we don't know this.
  const Field<2> theta_true = ice_shelf.interpolate(temperature);

  // The true velocity that we would measure for an ice shelf with the given
  // thickness and temperature can be obtained by solving for it numerically.
  const VectorField<2> u_true = ice_shelf.diagnostic_solve(h, theta_true, u_guess);

  // Make up a sensible value for the measurement error variance. In real life,
  // the remote sensing people have to tell us this.
  const Field<2> sigma = ice_shelf.interpolate(ConstantFunction<2>(10.0));


  /* ---------------------
   * Make some functionals
   * --------------------- */
  const auto F =
    [&](const Field<2>& theta)
    {
      const VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u_guess);
      return icepack::inverse::square_error(u, u_true, sigma);
    };

  const auto dF =
    [&](const Field<2>& theta)
    {
      const VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u_guess);
      const DualVectorField<2> du = icepack::inverse::misfit(u, u_true, sigma);
      const VectorField<2> lambda = ice_shelf.adjoint_solve(h, theta, u, du);
      return icepack::inverse::gradient(ice_shelf, h, theta, u, lambda);
    };


  /* ---------------------------------------------
   * Test that the gradient routine works properly
   * --------------------------------------------- */

  const DualField<2> df = dF(theta_guess);
  const Field<2> p = -transpose(df);
  const double prod = icepack::inner_product(df, p);

  const auto f = [&](const double beta){ return F(theta_guess + beta * p); };
  const double beta_max = icepack::numerics::armijo(f, prod, 1.0e-4, 0.5);

  const double cost = F(theta_guess);
  const double delta = 0.5;

  const size_t num_samples = 16;
  std::vector<double> errors(num_samples);
  for (size_t n = 0; n < num_samples; ++n) {
    const double delta_theta = std::pow(delta, n) * beta_max;
    const double delta_F = (F(theta_guess + delta_theta * p) - cost) / delta_theta;
    errors[n] = std::abs(1.0 - prod / delta_F);

    if (verbose)
      std::cout << delta_theta << " "
                << delta_F << " "
                << prod << " "
                << 1.0 - prod / delta_F << std::endl;
  }

  check(icepack::testing::is_decreasing(errors));

  return 0;
}
