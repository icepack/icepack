
#include "data.hpp"

// Some physical constants
const double rho = rho_ice * (1 - rho_ice / rho_water);
const double temp = 263.15;
const double delta_temp = 5.0;
const double A = pow(rho * gravity / 4, 3) * icepack::rate_factor(temp);

const double u0 = 100.0;
const double length = 20000.0, width = 20000.0;
const double h0 = 600.0;
const double delta_h = 300.0;


Tensor<1, 2> Velocity::value(const Point<2>& x) const
{
  const double q = 1 - pow(1 - delta_h * x[0] / (length * h0), 4);

  Tensor<1, 2> v; v[1] = 0.0;
  v[0] = u0 + 0.25 * A * q * length * pow(h0, 4) / delta_h;

  return v;
}


double Thickness::value(const Point<2>& x, const unsigned int) const
{
  const double X = x[0] / length;
  return h0 - delta_h * X;
}

double ParabolicTemperature::value(const Point<2>& x, const unsigned int) const
{
  const double X = x[0] / length;
  const double Y = x[1] / width;
  double q = 0.0;
  if (0.25 < X and X < 0.75 and
      0.25 < Y and Y < 0.75)
    q = 256 * (X - 0.25) * (0.75 - X) * (Y - 0.25) * (0.75 - Y);
  return temp + q * delta_temp;
}

double AlongFlowTemperature::value(const Point<2>& x, const unsigned int) const
{
  const double Y = x[1] / width;

  if (Y < 0.25 || Y > 0.75) return temp;
  return temp + delta_temp;
}

double
DiscontinuousTemperature::value(const Point<2>& x, const unsigned int) const
{
  const double X = x[0] / length;
  const double Y = x[1] / width;

  if (Y > 0.25 && Y < 0.75 && X > 0.25 && X < 0.75)
    return temp + delta_temp;

  return temp;
}

double NullRegularizer::operator()(const Field<2>&) const
{
  return 0.0;
}

DualField<2> NullRegularizer::derivative(const Field<2>& u) const
{
  return DualField<2>(u.get_discretization());
}

Field<2> NullRegularizer::filter(const Field<2>&, const DualField<2>& f) const
{
  return transpose(f);
}


std::map<std::string, std::string> get_cmdline_args(int argc, char ** argv)
{
  std::map<std::string, std::string> args;
  for (int k = 0; k < argc - 1; ++k) {
    if (argv[k][0] == '-') {
      std::string key(argv[k]);
      std::string val("");
      if (argv[k+1][0] != '-')
        val = argv[k+1];
      args.insert(std::make_pair(key, val));
    }
  }

  if (argv[argc-1][0] == '-')
    args.insert(std::make_pair(std::string(argv[argc-1]), ""));

  return args;
}


void help()
{
  static const std::string message(
    "This program performs synthetic inversions for the temperature of an   \n"
    "ice shelf with simple geometry and input data. You can explore the     \n"
    "capacity of an inverse method to infer temperature fields of varying   \n"
    "degrees of smoothness by changing the regularization method used,      \n"
    "adding noise to the input data, etc.                                   \n"
    "                                                                       \n"
    "Command-line arguments:                                                \n"
    "  -h: print this message                                               \n"
    "  -output <filename>: save the iterates of the temperature field to    \n"
    "    files filename000.ucd, filename001.ucd, etc.                       \n"
    "  -sigma <sigma>: add Gaussian noise with standard deviation sigma to  \n"
    "    the observed data over the true data to see how the inverse method \n"
    "    copes with inaccurate observations                                 \n"
    "  -tol <eps>: convergence tolerance for the iteration; defaults to 1e-3\n"
    "  -regularization <method>: regularize the inverse problem using the   \n"
    "    functional <method>, which must be one of either:                  \n"
    "      * square-gradient                                                \n"
    "      * total-variation                                                \n"
    "  -length <L>: smoothing length scale for regularization in km; the    \n"
    "    domain size is 20km                                                \n"
    "  -temp-profile <name>: name of the temperature profile to use, which  \n"
    "    must be one of:                                                    \n"
    "      * parabolic: a smooth profile which is parabolic in x and y      \n"
    "      * rectangle: a discontinuous profile which has a jump in y       \n"
    "      * square: a discontinuous profile with jumps in x and y          \n"
  );
  std::cout << message << std::endl;
}
