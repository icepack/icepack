
#ifndef SHALLOW_SHELF_HPP
#define SHALLOW_SHELF_HPP


#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/lac/vector.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_q.h>

#include <fstream>
#include <iostream>

#include "../elliptic_systems.hpp"
#include "../physical_constants.hpp"
#include "../ice_thickness.hpp"


namespace icepack
{

  using dealii::Triangulation;
  using dealii::QGauss;
  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::DoFHandler;
  using dealii::FEValuesBase;
  using dealii::FESystem;
  using dealii::ConstraintMatrix;
  using dealii::SparsityPattern;
  using dealii::SparseMatrix;
  using dealii::Vector;
  using dealii::Tensor;
  using dealii::SymmetricTensor;
  using dealii::FullMatrix;

  using EllipticSystems::AssembleRHS;


  /**
   * Responsibility for assembling the cell stiffness matrix for the shallow
   * shelf model is delegated to this class.
   */

  class AssembleDrivingStress : public AssembleRHS<2>
  {
  public:
    AssembleDrivingStress (const unsigned int _n_q_points,
                           const unsigned int _dofs_per_cell,
                           const FESystem<2>& _fe,
                           const IceThickness& _ice_thickness,
                           const Function<2>& _surface);
    void operator() (const FEValuesBase<2>& fe_values,
                     Vector<double>&        cell_rhs);

  protected:
    const unsigned int n_q_points;
    const unsigned int dofs_per_cell;
    const FESystem<2>& fe;
    const IceThickness& thickness;
    const Function<2>&  surface;
    std::vector<double> thickness_values;
    std::vector< Tensor<1, 2> > surface_gradient_values;
  };


  class AssembleFrontalStress : public AssembleRHS<2>
  {
  public:
    AssembleFrontalStress (const unsigned int _n_face_q_points,
                           const unsigned int _dofs_per_cell,
                           const FESystem<2>& _fe,
                           const IceThickness& _ice_thickness,
                           const Function<2>& _surface);
    void operator () (const FEValuesBase<2>& fe_face_values,
                      Vector<double>&        cell_rhs);

  protected:
    const unsigned int n_face_q_points;
    const unsigned int dofs_per_cell;
    const FESystem<2>& fe;
    const IceThickness& thickness;
    const Function<2>&  surface;
    std::vector<double> thickness_values;
    std::vector<double> surface_values;
  };



  /**
   * The main class for the shallow shelf glacier model.
   */
  class ShallowShelf
  {
  public:
    ShallowShelf (Triangulation<2>&  _triangulation,
                  const Function<2>& _surface,
                  const Function<2>& _bed,
                  const Function<2>& _temperature,
                  const TensorFunction<1, 2>& _boundary_velocity);
    ~ShallowShelf ();
    void diagnostic_solve ();
    //void output (const std::string& filename);

    void setup_system (const bool initial_step);

    template <class ConstitutiveTensor>
    void assemble_system ();

    template <class ConstitutiveTensor>
    void assemble_matrix ();

    void assemble_rhs ();
    void solve ();
    void refine_grid ();
    void output_results (const unsigned int cycle) const;

    // Member variables
    const Function<2>& surface;
    const Function<2>& bed;
    const IceThickness thickness;
    const Function<2>& temperature;
    const TensorFunction<1, 2>& boundary_velocity;

    Triangulation<2>&  triangulation;
    DoFHandler<2>      dof_handler;

    FESystem<2>        fe;

    QGauss<2>          quadrature_formula;
    QGauss<1>          face_quadrature_formula;

    ConstraintMatrix     hanging_node_constraints;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    Vector<double>       solution;
    Vector<double>       system_rhs;
  };


} // End of icepack namespace

#endif
