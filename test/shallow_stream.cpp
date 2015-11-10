
#include <deal.II/grid/grid_generator.h>

#include <icepack/glacier_models/shallow_stream.hpp>

using namespace dealii;
using namespace icepack;

const double length = 2000.0;
const double width = 500.0;

int main()
{
  Triangulation<2> triangulation;
  const Point<2> p1(0.0, 0.0), p2(length, width);
  GridGenerator::hyper_rectangle(triangulation, p1, p2);

  // Mark the right side of the rectangle as the ice front
  for (auto cell: triangulation.active_cell_iterators()) {
    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->center()(0) > length - 1.0)
        cell->face(face_number)->set_boundary_id(1);
  }

  triangulation.refine_global(3);

  ShallowStream ssa(triangulation, 1);
  const auto& vector_pde = ssa.get_vector_pde_skeleton();
  const FiniteElement<2>& fe = vector_pde.get_fe();

  const QGauss<2> quad(2);
  const QGauss<1> face_quad(2);

  FEValues<2> fe_values(fe, quad,  DefaultUpdateFlags::flags);
  FEFaceValues<2> fe_face_values(fe, face_quad, DefaultUpdateFlags::face_flags);

  for (auto cell: vector_pde.get_dof_handler().active_cell_iterators()) {
    fe_values.reinit(cell);

    const auto& xs = fe_values.get_quadrature_points();

    for (auto x: xs) std::cout << x << "  ";

    for (unsigned int face_number = 0;
         face_number < GeometryInfo<2>::faces_per_cell;
         ++face_number)
      if (cell->face(face_number)->at_boundary()
          and
          cell->face(face_number)->boundary_id() == 1) {
        fe_face_values.reinit(cell, face_number);
        const auto& face_xs = fe_face_values.get_quadrature_points();

        for (auto x: face_xs)
          std::cout << " " << x;
      }

    std::cout << std::endl;
  }

  return 0;
}
