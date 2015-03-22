
#include "shallow_shelf.hpp"
#include "ice_thickness.hpp"
#include "driving_stress.hpp"

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/grid_generator.h>


using namespace dealii;



Triangulation<2> make_domain()
{
  Triangulation<2> tri;
  GridGenerator::hyper_cube (tri);

  for (auto cell: tri.active_cell_iterators())
  {
    for (unsigned int i = 0; i < GeometryInfo<2>::vertices_per_cell; ++i)
    {
      Point<2>& v = cell->vertex(i);
      Point<2> u = {500 * v(0), 25 * v(0) + 100 * v(1) - 50 * v(0) * v(1)};
      v = u;
    }
  }

  return tri;
}



int main(int argc, char **argv)
{

  Triangulation<2> tri = make_domain ();

  ScalarFunctionFromFunctionObject<2>
    bed ([](const Point<2>& x)
         {
           return -1000.0;
         });

  ScalarFunctionFromFunctionObject<2>
    surface ([](const Point<2>& x)
             {
               return 200.0 - 0.04 * x(0);
             });

  ScalarFunctionFromFunctionObject<2>
    beta ([](const Point<2>& x)
          {
            return 0.0;
          });


  IceThickness thickness (bed, surface);
  DrivingStress driving_stress (thickness, surface);


  return 0;
}
