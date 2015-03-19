
#include "driving_stress.hpp"
#include "physical_constants.hpp"

#include <deal.II/base/tensor.h>

using dealii::Tensor;

DrivingStress::DrivingStress (const Function<2>& _thickness,
                              const Function<2>& _surface)
  :
  thickness(_thickness),
  surface(_surface)
{ }


void DrivingStress::vector_value (const Point<2>& x, Vector<double>& values) const
{
  const double h = thickness.value(x);
  Tensor<1, 2> grad = surface.gradient(x);
  values[0] = -rho_ice * gravity * h * grad[0];
  values[1] = -rho_ice * gravity * h * grad[1];
}
