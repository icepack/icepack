
#include <deal.II/grid/grid_generator.h>

#include <icepack/physics/constants.hpp>
#include <icepack/inverse/mean_square_error.hpp>
#include <icepack/inverse/optimization.hpp>
#include <icepack/inverse/ice_shelf.hpp>

using dealii::Tensor;
using dealii::Point;
using dealii::ConstantFunction;

using namespace icepack;


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



int main()
{
  /* ---------------
   * Generate a mesh
   * --------------- */
  dealii::Triangulation<2> mesh;
  dealii::GridGenerator
    ::hyper_rectangle(mesh, Point<2>(0.0, 0.0), Point<2>(length, width));

  for (auto cell: mesh.active_cell_iterators())
    for (unsigned int face_number = 0;
         face_number < dealii::GeometryInfo<2>::faces_per_cell; ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id(1);

  const unsigned int num_levels = 5;
  mesh.refine_global(num_levels);


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
  IceShelf ice_shelf(mesh, 1);

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

  // Compute the error of our initial guess
  double J = inverse::mean_square_error(u_guess, u_true, sigma);

  std::cout << 0 << ", " << J << std::endl;

  // Compute the gradient of of the objective functional. Note that this field
  // has dimensions of inverse temperature.
  Field<2> dJ = inverse::gradient(ice_shelf, h, theta_guess, u_true, sigma);

  // Compute a direction in which to search for a better solution. In order to
  // get the units right, we can normalize the gradient and multiply by the
  // average temperature.
  Field<2> p = -rms_average(theta_guess) * dJ / norm(dJ);


  // First, find an interval in which to perform the line search using the
  // Armijo method. See Polak's "Optimization", pp. 30-31.
  const auto f =
    [&](const double beta)
    {
      Field<2> theta = theta_guess + beta * p;
      VectorField<2> u = ice_shelf.diagnostic_solve(h, theta, u_guess);
      return inverse::mean_square_error(u, u_true, sigma);
    };

  double beta =
    inverse::armijo(f, -rms_average(theta_guess) * norm(dJ), 1.0e-4, 0.5);

  std::cout << beta << ", " << f(beta) << std::endl;


  // Next, use something slightly smarter than the bisection method to find the
  // minimum value of the objective functional between 0.0 and the endpoint
  // we just computed using the Armijo method.
  beta = inverse::golden_section_search(f, 0.0, beta, 1.0e-3);

  std::cout << beta << "," << f(beta) << std::endl;

  return 0;
}
