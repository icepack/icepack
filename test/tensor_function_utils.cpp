
#include <random>

#include <deal.II/base/function_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <icepack/util/tensor_function_utils.hpp>

const unsigned int num_levels = 3;
const double dx = 1.0 / (1 << num_levels);

using namespace dealii;
using namespace icepack;
using std::abs;


template <int dim>
class F : public Function<dim>
{
public:
  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    return x[0] * x[1];
  }
};


template <int dim>
class G : public Function<dim>
{
public:
  double value(const Point<dim>& x, const unsigned int = 0) const
  {
    return x[0]*x[0] - x[1]*x[1];
  }
};


int main()
{
  const Point<2> p1(0.0, 0.0), p2(1.0, 1.0);
  Triangulation<2> triangulation;
  GridGenerator::hyper_rectangle(triangulation, p1, p2);
  triangulation.refine_global(num_levels);

  const F<2> f;
  const G<2> g;

  // Create a TensorFunction from two scalar Functions
  const auto psi = util::TensorFunctionFromScalarFunctions<2>(f, g);

  // Generate a bunch of random poins in the unit square
  std::random_device device;
  std::mt19937 rng;
  rng.seed(device());
  std::uniform_real_distribution<> u(0, 1);

  const size_t num_points = 32;
  std::vector<Point<2> > points(num_points);
  for (size_t k = 0; k < num_points; ++k) {
    points[k][0] = u(rng);
    points[k][1] = u(rng);
  }

  // Check that the TensorFunction and its coordinate functions agree at a
  // few points
  for (size_t k = 0; k < num_points; ++k) {
    const Point<2>& p = points[k];
    const Tensor<1, 2> v = psi.value(p);
    Tensor<1, 2> w;
    w[0] = f.value(p);
    w[1] = g.value(p);

    Assert((v - w).norm() < 1.0e-15, ExcInternalError());
  }

  return 0;
}
