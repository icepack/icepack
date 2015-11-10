
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
    }

    ~PDESkeleton()
    {
      dof_handler.clear();
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

  protected:
    const FE fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity;
  };


  template <int dim>
  using ScalarPDESkeleton = PDESkeleton<dim, FE_Q<dim> >;

  template <int dim>
  using VectorPDESkeleton = PDESkeleton<dim, FESystem<dim> >;

}

#endif
