
// deal.II includes
#include <deal.II/base/function.h>

// our includes
#include "read_gridded_data.hpp"

// C++ includes
#include <fstream>
#include <iostream>


using dealii::Point;


int main (int argc, char **argv)
{
  bool verbose = false;
  if (strcmp(argv[argc-1], "-v") == 0) verbose = true;

  GridData jak = readGeoDat("/home/daniel/TSX_W69.10N_10Jun12_21Jun12.vx");

  if (verbose) std::cout << "Reading GeoDat file worked!" << std::endl;
  return 0;
}
