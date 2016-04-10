
#include <icepack/grid_data.hpp>

using namespace icepack;

int main()
{
  // Make the coordinate values
  const size_t nx = 128, ny = 192;
  const double dx = 2.0, dy = 2.0;
  const double x0 = 3.0, y0 = 7.0;

  std::vector<double> xs(nx), ys(ny);

  for (size_t i = 0; i < ny; ++i)
    ys[i] = y0 + i * dy;

  for (size_t j = 0; j < nx; ++j)
    xs[j] = x0 + j * dx;

  std::array<std::vector<double>, 2> coordinate_values = {{xs, ys}};

  // Make some synthetic data
  dealii::Table<2, double> data(nx, ny);

  for (size_t i = 0; i < ny; ++i)
    for (size_t j = 0; j < nx; ++j)
      data[j][i] = xs[j] + ys[i];

  // Make on of the measurements a missing data point
  const double missing = -9999.0;
  const size_t I = ny / 2, J = nx / 2;
  data[J][I] = missing;

  // Make the gridded data object
  const GridData grid_data(coordinate_values, data, missing);

  // Check that a point near the missing data is masked
  const Point<2> x(x0 + (J + 0.5) * dx, y0 + (I + 0.5) * dy);
  if (not grid_data.is_masked(x))
    return 1;

  const Point<2> y(x0 + dx, y0 + dy);
  if (grid_data.is_masked(y))
    return 1;

  return 0;
}
