
#include "data.hpp"

#include <iostream>
#include <iomanip>

int main(int argc, char ** argv)
{
  // Parse command-line arguments
  std::map<std::string, std::string> args = get_cmdline_args(argc, argv);

  if (args.count("-h")) {
    help();
    return 0;
  }

  const std::string output_name = args["-output"];
  const bool output = output_name != "";

  const bool noisy = args.count("-sigma");
  const double stddev = noisy ? std::stod(args["-sigma"]) : 10.0;

  std::string regularizer_name = "none";
  if (args.count("-regularization")) {
    const std::string arg = args["-regularization"];
    if (arg == "square-gradient" || arg == "total-variation")
      regularizer_name = arg;
  }

  std::string temp_profile = "parabolic";
  if (args.count("-temp-profile")) {
    const std::string arg = args["-temp-profile"];
    if (arg == "square" || arg == "rectangle")
      temp_profile = arg;
  }

  double tol = 1.0e-3;
  const std::string tol_name = args["-tol"];
  if (tol_name != "")
    tol = std::stod(tol_name);

  double length_scale = length;
  const std::string length_name = args["-length"];
  if (length_name != "")
    length_scale = 1000.0 * std::stod(length_name);


  /* -------------------------------------
   * Set up the geometry and glacier model */
  dealii::Triangulation<2> mesh;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  dealii::GridGenerator::hyper_rectangle(mesh, p1, p2);

  // Refine the mesh 5 times to get a decent resolution
  const unsigned int num_levels = 5;
  mesh.refine_global(num_levels);

  for (auto cell: mesh.cell_iterators())
    for (unsigned int face_number = 0;
         face_number < dealii::GeometryInfo<2>::faces_per_cell; ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id(1);

  const double area = dealii::GridTools::volume(mesh);

  icepack::IceShelf ice_shelf(mesh, 1);
  const auto& dsc = ice_shelf.get_discretization();


  /* -------------------
   * Make synthetic data */
  const Field<2>
    h = ice_shelf.interpolate(Thickness()),
    theta_guess = ice_shelf.interpolate(ConstantFunction<2>(temp)),
    sigma = ice_shelf.interpolate(ConstantFunction<2>(stddev));

  const VectorField<2>
    u_guess = ice_shelf.interpolate(Velocity());


  /* --------------------
   * Make the "true" data */
  std::unique_ptr<Function<2> > temperature_ptr =
    std::make_unique<ParabolicTemperature>();
  if (temp_profile == "rectangle")
    temperature_ptr = std::make_unique<AlongFlowTemperature>();
  if (temp_profile == "square")
    temperature_ptr = std::make_unique<DiscontinuousTemperature>();
  const Function<2>& temperature = *temperature_ptr;

  const Field<2>
    theta_true = ice_shelf.interpolate(temperature);

  const VectorField<2>
    u_true = ice_shelf.diagnostic_solve(h, theta_true, u_guess),
    u_obs = add_noise(u_true, noisy ? stddev : 0.0);

  if (output) {
    const std::string theta_filename = output_name + "_true.ucd";
    theta_true.write(theta_filename, "theta");

    const std::string u_true_filename = "u_true.ucd";
    u_true.write(u_true_filename, "u");

    const std::string u_obs_filename = "u_obs.ucd";
    u_obs.write(u_obs_filename, "u");
  }


  /* -------------------------
   * Solve the inverse problem */
  VectorField<2> u(u_guess);

  // Compute the error of our (crude) initial guess. In real life we don't get
  // to see what this is.
  double mean_error =
    dist(theta_guess, theta_true) / (std::sqrt(area) * std::abs(delta_temp));

  // Compute the residual of our initial guess, i.e. the misfit between the
  // computed and observed velocity. In real life this is all we get.
  double mean_residual = inverse::square_error(u, u_true, sigma) / area;

  std::cout << "Initial velocity misfit: " << mean_residual << std::endl
            << "Initial theta misfit:    " << mean_error << std::endl
            << "Measurement error:       " << (noisy * stddev) << std::endl
            << "Convergence tolerance:   " << tol << std::endl
            << "Smoothing length scale:  " << length_scale / 1.0e3 << " km"
            << std::endl;

  // Typical length and temperature scales for the problem. We need these to
  // non-dimensionalize everything and set the smoothing length for
  // regularization.
  const double theta_scale = 10.0;
  const double alpha = length_scale / theta_scale;

  // Create an object for regularizing the problem. This stores the matrices we
  // need to filter out any high-frequency oscillations from the putative
  // temperature field.
  std::unique_ptr<Regularizer<2> >
    regularizer = std::make_unique<NullRegularizer>();

  std::cout << "Regularization method:   ";
  if (regularizer_name == "square-gradient") {
    regularizer = std::make_unique<SquareGradient<2> >(dsc, alpha);
    std::cout << "square gradient";
  } else if (regularizer_name == "total-variation") {
    regularizer = std::make_unique<TotalVariation<2> >(dsc, alpha, 0.1);
    std::cout << "total variation";
  } else {
    std::cout << "none";
  }
  std::cout << std::endl;


  // The optimization routines expect to work with functions of one variable,
  // so we'll create some lambda functions that project out all the other
  // arguments:

  // First, the objective functional itself, which consists of both the misfit
  // with observations and regularization.
  const auto F =
    [&](const Field<2>& theta) -> double
    {
      u = ice_shelf.diagnostic_solve(h, theta, u);
      return inverse::square_error(u, u_obs, sigma) + (*regularizer)(theta);
    };

  // Next, we need the derivative of the objective functional in order to
  // optimize it.
  const auto dF =
    [&](const Field<2>& theta) -> DualField<2>
    {
      const DualField<2>
        dE = inverse::gradient(ice_shelf, h, theta, u_obs, sigma),
        dR = regularizer->derivative(theta);
      return dE + dR;
    };


  // Set a stopping criterion.
  const double tolerance = tol * area;

  // Solve the inverse problem using a descent algorithm
  Field<2> theta = numerics::lbfgs(F, dF, theta_guess, 6, tolerance);


  // Compute the final misfits in velocity and temperature.
  u = ice_shelf.diagnostic_solve(h, theta, u);


  theta.write("theta.ucd", "theta");
  u.write("u.ucd", "u");


  mean_residual = inverse::square_error(u, u_true, sigma) / area;
  mean_error =
    dist(theta, theta_true) / (std::sqrt(area) * std::abs(delta_temp));

  std::cout << "Final velocity misfit:      " << mean_residual << std::endl
            << "Final temperature misfit:   " << mean_error << std::endl;

  return 0;
}
