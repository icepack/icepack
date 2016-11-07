
#include <fstream>
#include <icepack/physics/viscosity.hpp>
#include "../testing.hpp"

int main(int argc, char ** argv)
{
  const auto args = icepack::testing::get_cmdline_args(argc, argv);
  const bool verbose = args.count("-v") || args.count("--verbose");

  icepack::Rheology rheology(3.0);

  constexpr size_t num_temps = 6;
  const double thetas[num_temps] =
    {243.2, 248.2, 253.2, 258.2, 263.2, 268.2};

  const double delta = 0.5;
  const size_t num_deltas = 16;

  for (size_t n = 0; n < num_temps; ++n) {
    const double theta = thetas[n];
    const double B = rheology.B(theta);
    const double dB = rheology.dB(theta);

    std::vector<double> errors(num_deltas);
    for (size_t k = 0; k < num_deltas; ++k) {
      const double delta_theta = std::pow(delta, k);
      const double delta_B = (rheology.B(theta + delta_theta) - B) / delta_theta;
      errors[k] = std::abs(1.0 - dB / delta_B);
      if (verbose) std::cout << errors[k] << " ";
    }
    if (verbose) std::cout << std::endl;

    check(icepack::testing::is_decreasing(errors));
  }

  return 0;
}
