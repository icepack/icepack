
#include <deal.II/grid/grid_generator.h>
#include <icepack/field.hpp>
#include "../testing.hpp"

using dealii::Point;
using dealii::Tensor;
using dealii::Function;
using dealii::TensorFunction;
using dealii::ExcInternalError;
template <int dim> using Fn = dealii::ScalarFunctionFromFunctionObject<dim>;

using icepack::Discretization;
using icepack::Field;
using icepack::VectorField;
using icepack::FieldType;
using icepack::interpolate;
namespace DefaultUpdateFlags = icepack::DefaultUpdateFlags;

int main()
{
  const unsigned int num_levels = 5;
  const double dx = 1.0 / (1 << num_levels);

  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  dealii::Triangulation<2> mesh;
  dealii::GridGenerator::hyper_rectangle(mesh, p1, p2);
  mesh.refine_global(num_levels);

  const Discretization<2> discretization(mesh, 1);

  const Fn<2> Phi1([](const Point<2>& x){return std::exp(-x*x);});
  const Fn<2> Phi2([](const Point<2>& x){return x[0]*x[0] - x[1]*x[1];});
  const Fn<2> Phi3([](const Point<2>& x){return std::cos(x[0] + x[1]);});

  const Field<2> phi1 = interpolate(discretization, Phi1);
  const Field<2> phi2 = interpolate(discretization, Phi2);
  const Field<2> phi3 = interpolate(discretization, Phi3);

  // Test assignment of expression templates to fields
  {
    Field<2> phi(discretization), psi(discretization);
    phi = phi1 + phi2;
    psi = phi1;
    psi += phi2;
    check_fields(phi, psi, dx*dx);

    phi = 2 * phi1 + phi2;
    psi = phi1;
    psi *= 2;
    psi += phi2;
    check_fields(phi, psi, dx*dx);

    phi = phi1 + 3*phi2 + phi3;
    psi = phi2;
    psi *= 3;
    psi += phi1;
    psi += phi3;
    check_fields(phi, psi, dx*dx);

    phi = phi1 + 7*phi2 + 3*phi3;
    psi = phi2;
    psi *= 7.0/3;
    psi += phi3;
    psi *= 3;
    psi += phi1;
    check_fields(phi, psi, dx*dx);

    phi = phi1 - phi2;
    psi = phi1;
    psi -= phi2;
    check_fields(phi, psi, dx*dx);

    phi = -phi1;
    psi = phi1;
    psi *= -1.0;
    check_fields(phi, psi, dx*dx);
  }

  // Test assign and add/subtract expression templates to fields
  {
    Field<2> phi(discretization), psi(discretization);
    phi = phi1;
    phi += 2 * phi2;
    phi -= phi2 + phi3;
    psi = phi1;
    psi += phi2;
    psi -= phi3;
    check_fields(phi, psi, dx*dx);
  }

  // Test initializing fields from expression templates
  {
    const Field<2> phi(phi1 + 2 * phi2);
    Field<2> psi(phi2);
    psi *= 2;
    psi += phi1;
  }

  return 0;
}
