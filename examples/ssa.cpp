
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

      for (auto cell: tri.active_cell_iterators())
        for (unsigned int face_number = 0;
             face_number < GeometryInfo<2>::faces_per_cell;
             ++face_number)
          if (cell->face(face_number)->center()(0) > 0.95)
            {
              // std::cout << "Yurp, got one!" << std::endl;
              cell->face(face_number)->set_boundary_indicator (1);
            }

      SurfaceElevation surface;
      BedElevation bed;
      BoundaryVelocity vb;

      ShallowShelfApproximation::ShallowShelf ssa(tri, surface, bed, vb);
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
