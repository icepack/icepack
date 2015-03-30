
#include <deal.II/grid/grid_generator.h>

#include "shallow_shelf.hpp"
#include "rhs.hpp"

using namespace dealii;

int main ()
{
  try
    {
      dealii::deallog.depth_console (0);

      Triangulation<2> tri;
      GridGenerator::hyper_cube(tri, -1, 1);
      RightHandSide<2> rhs;
      ShallowShelfApproximation::ShallowShelf ssa(tri, rhs);
      ssa.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
