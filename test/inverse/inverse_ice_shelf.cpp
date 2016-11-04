
#include <icepack/physics/constants.hpp>
#include <icepack/numerics/optimization.hpp>
#include <icepack/inverse/error_functionals.hpp>
#include <icepack/inverse/regularization.hpp>
#include <icepack/inverse/ice_shelf.hpp>
#include "../testing.hpp"

using dealii::Tensor;
using dealii::Point;
using dealii::ConstantFunction;

using namespace icepack;

using icepack::inverse::Regularizer;
using SG = icepack::inverse::SquareGradient<2>;
using TV = icepack::inverse::TotalVariation<2>;


// Some physical constants
const double rho = rho_ice * (1 - rho_ice / rho_water);
const double temp = 263.15;
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
  /**
   * Parse command-line arguments
   */
  std::set<std::string> args = testing::get_cmdline_args(argc, argv);

  const bool verbose = args.count("-v") || args.count("--verbose");
  const bool tv = args.count("-tv") || args.count("--total-variation");


  /* ------------------------------
   * Make some synthetic input data
   * ------------------------------ */

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
  dealii::Triangulation<2> tria = testing::rectangular_glacier(length, width);
  IceShelf ice_shelf(tria, 1);
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


  /* -------------------------
   * Solve the inverse problem
   * ------------------------- */
  VectorField<2> u(u_guess);

  // Compute the error of our crude guess
  const double area = dealii::GridTools::volume(tria);
  double mean_error =
    dist(theta_guess, theta_true) / (std::sqrt(area) * std::abs(delta_temp));
  double mean_residual = inverse::square_error(u, u_true, sigma) / area;

  if (verbose)
    std::cout << "Initial velocity error:    " << mean_residual << std::endl
              << "Initial temperature error: " << mean_error << std::endl;

  // Typical length and temperature scales for the problem. Used to fix the
  // regularization parameter.
  const double length_scale = length;
  const double theta_scale = 30.0;
  const double alpha = length_scale / theta_scale;

  // Create an object for computing the regularization functional.
  // Depending on command-line arguments, this is either the square gradient
  // or the total variation.
  std::unique_ptr<Regularizer<2> > regularizer;
  if (tv) regularizer = std::unique_ptr<TV>(new TV(discretization, alpha, 0.5));
  else regularizer = std::unique_ptr<SG>(new SG(discretization, alpha));

  // Create some lambda functions which will calculate the objective functional
  // and its gradient for a given value of the temperature field, but capture
  // all the other data like the model object, ice thickness, etc.
  const auto F =
    [&](const Field<2>& theta)
    {
      u = ice_shelf.diagnostic_solve(h, theta, u_guess);
      return inverse::square_error(u, u_true, sigma) + (*regularizer)(theta);
    };

  const auto dF =
    [&](const Field<2>& theta)
    {
      const auto dE = inverse::gradient(ice_shelf, h, theta, u_true, sigma);
      const auto dR = regularizer->derivative(theta);
      return DualField<2>(dE + dR);
    };

  // Stop the iteration when the improvement from one iterate to the next is
  // less than this tolerance.
  const double tolerance = 5.0e-2;

  // Use a descent algorithm to find a good value of the temperature
  Field<2> theta = numerics::lbfgs(F, dF, theta_guess, 6, tolerance);

  // Compute the final misfits in velocity and temperature.
  u = ice_shelf.diagnostic_solve(h, theta, u);
  mean_residual = inverse::square_error(u, u_true, sigma) / area;
  mean_error = dist(theta, theta_true) / (std::sqrt(area) * std::abs(delta_temp));

  if (verbose)
    std::cout << "Final velocity error:      " << mean_residual << std::endl
              << "Final temperature error:   " << mean_error << std::endl;

  check(mean_residual < 0.01);

  if (verbose) {
    theta_true.write("theta_true.ucd", "theta");
    u_true.write("u_true.ucd", "u");
    theta.write("theta.ucd", "theta");
    u.write("u.ucd", "u");

    const Field<2> delta_theta = theta_true - theta;
    delta_theta.write("delta_theta.ucd", "delta_theta");
  }

  return 0;
}
