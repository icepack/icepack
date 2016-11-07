
#include <set>
#include <string>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_refinement.h>

namespace icepack {
  namespace testing {

    std::set<std::string> get_cmdline_args(int argc, char ** argv)
    {
      std::set<std::string> args;
      for (int k = 0; k < argc; ++k)
        args.insert(std::string(argv[k]));

      return args;
    }


    dealii::Triangulation<2>
    rectangular_glacier(double length, double width, unsigned int num_levels)
    {
      dealii::Triangulation<2> triangulation;
      const dealii::Point<2> p1(0.0, 0.0), p2(length, width);
      dealii::GridGenerator::hyper_rectangle(triangulation, p1, p2);

      for (auto cell: triangulation.active_cell_iterators())
        for (unsigned int face_number = 0;
             face_number < dealii::GeometryInfo<2>::faces_per_cell;
             ++face_number)
          if (cell->face(face_number)->center()(0) > length - 1.0)
            cell->face(face_number)->set_boundary_id(1);

      triangulation.refine_global(num_levels);

      return triangulation;
    }


    bool is_decreasing(const std::vector<double>& seq)
    {
      for (size_t k = 1; k < seq.size(); ++k)
        if (seq[k] > seq[k - 1])
          return false;

      return true;
    }


  } // End of namespace testing
} // End of namespace icepack
