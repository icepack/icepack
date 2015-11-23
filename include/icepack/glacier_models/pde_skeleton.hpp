
#ifndef ICEPACK_PDE_SKELETON_HPP
#define ICEPACK_PDE_SKELETON_HPP

#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

namespace icepack
{
  using dealii::Triangulation;
  using dealii::FE_Q;
  using dealii::FESystem;
  using dealii::DoFHandler;
  using dealii::SparsityPattern;
  using dealii::ConstraintMatrix;
  using dealii::UpdateFlags;


  // Default update flags for FEValues objects.
  namespace DefaultUpdateFlags
  {
    using dealii::update_values;
    using dealii::update_gradients;
    using dealii::update_quadrature_points;
    using dealii::update_JxW_values;
    using dealii::update_normal_vectors;

    const UpdateFlags flags =
      update_values            | update_gradients |
      update_quadrature_points | update_JxW_values;

    const UpdateFlags face_flags =
      update_values         | update_quadrature_points |
      update_normal_vectors | update_JxW_values;
  }


  /**
   * This is a utility class for setting up partial differential equations.
   * Every PDE is defined over some geometry; has some finite element basis;
   * has a mapping from geometric primitives to degrees of freedom in the FE
   * expansion; and has a sparsity pattern for the matrix of the resulting
   * linear system.
   *
   * This class does *not* contain a sparse matrix; two PDEs which share the
   * same tensor rank (scalar, vector, etc.) and underlying geometry may have
   * different linear systems. Likewise, it does not store right-hand sides,
   * solution vectors, etc.
   */
  template <int dim, class FE>
  class PDESkeleton
  {
  public:
    // Constructors & destructors
    PDESkeleton(const Triangulation<dim>& triangulation,
                const FE& _fe)
      :
      fe(_fe),
      dof_handler(triangulation)
    {
      dof_handler.distribute_dofs(fe);

      const unsigned int nn = dof_handler.n_dofs();
      sparsity.reinit(nn, nn, dof_handler.max_couplings_between_dofs());
      dealii::DoFTools::make_sparsity_pattern(dof_handler, sparsity);
      sparsity.compress();

      constraints.clear();
      dealii::DoFTools::make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();
    }

    ~PDESkeleton()
    {
      dof_handler.clear();
    }


    // Helper functions for making boundary value maps
    std::map<dealii::types::global_dof_index, double>
    zero_boundary_values(const unsigned int boundary_id = 0) const
    {
      std::map<dealii::types::global_dof_index, double> boundary_values;

      dealii::VectorTools::interpolate_boundary_values(
        dof_handler,
        boundary_id,
        dealii::ZeroFunction<dim>(fe.n_components()),
        boundary_values
      );

      return std::move(boundary_values);
    }


    template <int rank>
    std::map<dealii::types::global_dof_index, double>
    interpolate_boundary_values(const FieldType<rank, dim>& phi,
                                const unsigned int boundary_id = 0) const
    {
      const unsigned int n_components = std::pow(dim, rank);
      AssertDimension(fe.n_components(), n_components);

      // This perhaps requires some explanation.
      // Usually, one would take a deal.II Function object, interpolate it to
      // the boundary of the Triangulation, then keep track of the relevant
      // degress of freedom in a std::map object. One then uses this std::map
      // to fix the Dirichlet boundary conditions. Creating this std::map
      // is done in the function VectorTools::interpolate_boundary_values.
      //
      // In our case, however, we don't have a Function object for the boundary
      // values, just some FE field `phi`. Instead, we create the std::map by
      // interpolating 0 to the boundary...
      auto boundary_values = zero_boundary_values(boundary_id);

      // ...and, knowing the right degrees of freedom to fix for Dirichlet BCs,
      // we can get these directly from the coefficients of the input Field.
      const Vector<double>& Phi = phi.get_coefficients();
      for (auto& p: boundary_values) p.second = Phi(p.first);

      return std::move(boundary_values);
    }


    // Accessors
    const FE& get_fe() const
    {
      return fe;
    }

    const DoFHandler<dim>& get_dof_handler() const
    {
      return dof_handler;
    }

    const SparsityPattern& get_sparsity_pattern() const
    {
      return sparsity;
    }

    const ConstraintMatrix& get_constraints() const
    {
      return constraints;
    }

  protected:
    const FE fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity;
    ConstraintMatrix constraints;
  };


  template <int dim>
  using ScalarPDESkeleton = PDESkeleton<dim, FE_Q<dim> >;

  template <int dim>
  using VectorPDESkeleton = PDESkeleton<dim, FESystem<dim> >;

}

#endif
