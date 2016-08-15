
#include <iostream>

#include <icepack/read_mesh.hpp>
#include <icepack/grid_data.hpp>
#include <icepack/glacier_models/ice_shelf.hpp>

using icepack::Field;
using icepack::VectorField;
using icepack::DualVectorField;
using icepack::IceShelf;
using icepack::readArcAsciiGrid;

void usage()
{
  std::cout << "This program solves for the velocity of the Ross " << std::endl
            << "Ice Shelf, Antarctica, using observational data  " << std::endl
            << "from MeASUREs and BEDMAP2.                       " << std::endl
            << "Usage:                                           " << std::endl
            << "  ./ross  <mesh>.msh  <h>.txt  <vx>.txt  <vy>.txt" << std::endl
            << "The outputs are written to an ArcInfo ASCII Grid." << std::endl;
}

int main(int argc, char ** argv)
{
  if (argc < 5) {
    usage();
    return 0;
  }

  const std::string mesh_filename = argv[1];
  const std::string h_filename = argv[2];
  const std::string vx_filename = argv[3];
  const std::string vy_filename = argv[4];

  dealii::Triangulation<2> tria = icepack::read_gmsh_grid<2>(mesh_filename);

  const auto h_obs = readArcAsciiGrid(h_filename);
  const auto vx_obs = readArcAsciiGrid(vx_filename);
  const auto vy_obs = readArcAsciiGrid(vy_filename);

  std::cout
      << "x range: " << h_obs.xrange[0] << ", " << h_obs.xrange[1] << std::endl
      << "y range: " << h_obs.yrange[0] << ", " << h_obs.yrange[1] << std::endl;

  IceShelf ice_shelf(tria, 1);

  const Field<2> h = ice_shelf.interpolate(h_obs);
  const VectorField<2> vo = ice_shelf.interpolate(vx_obs, vy_obs);

  const Field<2> theta = ice_shelf.interpolate(dealii::ConstantFunction<2>(263.15));
  const DualVectorField<2> tau_d = ice_shelf.driving_stress(h);
  const DualVectorField<2> r = ice_shelf.residual(h, theta, vo, tau_d);

  std::cout << "Initial residual: " << norm(r) << std::endl;

  const VectorField<2> v = ice_shelf.diagnostic_solve(h, theta, vo);

  std::cout << "Solution norm: " << norm(v) << std::endl;

  v.write("v.ucd", "v");
  vo.write("vo.ucd", "v");

  return 0;
}
