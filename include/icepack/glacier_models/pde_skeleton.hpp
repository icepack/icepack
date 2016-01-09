
#ifndef ICEPACK_PDE_SKELETON_HPP
#define ICEPACK_PDE_SKELETON_HPP

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>

#include <icepack/field.hpp>

namespace icepack {

  using dealii::Triangulation;
  using dealii::FE_Q;
  using dealii::FESystem;
  using dealii::DoFHandler;
  using dealii::SparsityPattern;
  using dealii::ConstraintMatrix;

  /**
   * Default update flags for `dealii::FEValues` objects when iterating over
   * the degrees of freedom of a finite element field.
   */
  namespace DefaultUpdateFlags {
    using dealii::UpdateFlags;

    /**
     * Default update flags for finite element values on cells of the geometry
     */
    extern const UpdateFlags flags;

    /**
     * Default update flags for finite element values on faces of the geometry
     */
    extern const UpdateFlags face_flags;
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
   * solution vectors, etc. Nonetheless, two distinct PDEs may share the same
   * sparsity pattern or mapping from geometry to FE degrees of freedom.
   */
  template <int dim, class FE>
  class PDESkeleton
  {
  public:
    /**
     * Construct a PDE skeleton given the geometry and finite element basis;
     * from this data, the PDE skeleton constructs all other necessary data to
     * set up a PDE, e.g. degree-of-freedom handler, sparsity pattern,
     * quadrature rules, constraints on hanging nodes.
     */
    PDESkeleton(const Triangulation<dim>& triangulation,
                const FE& _fe)
      :
      fe(_fe),
      quad(fe.tensor_degree()),
      face_quad(fe.tensor_degree()),
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


    /**
     * Destructor; releases memory stored by the degree-of-freedom handler.
     */
    ~PDESkeleton()
    {
      dof_handler.clear();
    }


    /**
     * In order to enforce Dirichlet boundary conditions, deal.II often uses a
     * `std::map` of the Dirichlet boundary degrees of freedom to the boundary
     * value. This function constructs a map for homogeneous Dirichlet
     * conditions.
     */
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


    /**
     * Construct a boundary-value `map` (see `zero_boundary_values`) from a
     * field object, so that other fields can be made to have the same boundary
     * values as the input field.
     */
    template <int rank>
    std::map<dealii::types::global_dof_index, double>
    interpolate_boundary_values(
      const FieldType<rank, dim>& phi,
      const unsigned int boundary_id = 0
    ) const
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


    /**
     * Return the finite element object representing the basis functions used
     * to discretize PDEs with this skeleton.
     */
    const FE& get_fe() const
    {
      return fe;
    }

    /**
     * Return a reference to the degree-of-freedom handler for this skeleton.
     */
    const DoFHandler<dim>& get_dof_handler() const
    {
      return dof_handler;
    }

    /**
     * Return the quadrature rule for integrating over cells of the underlying
     * geometry, i.e. quads in 2D and hexes in 3D.
     */
    const QGauss<dim>& get_quadrature() const
    {
      return quad;
    }

    /**
     * Return the quadrature rule for integrating over faces of the underlying
     * geometry, i.e. lines in 2D and quads in 3D.
     */
    const QGauss<dim-1>& get_face_quadrature() const
    {
      return face_quad;
    }

    /**
     * Return a reference to the sparsity pattern for a PDE with the current
     * skeleton. Many sparse matrices can share the same sparsity pattern.
     */
    const SparsityPattern& get_sparsity_pattern() const
    {
      return sparsity;
    }

    /**
     * Return a reference to the constraints on the degrees of freedom for this
     * PDE, such as would occur through hanging nodes generated through adative
     * mesh refinement.
     */
    const ConstraintMatrix& get_constraints() const
    {
      return constraints;
    }

  protected:
    const FE fe;
    const QGauss<dim> quad;
    const QGauss<dim-1> face_quad;

    /**
     * Mapping of geometric primitives in the triangulation to finite element
     * degrees of freedom. The PDE skeleton is the owner of the DoF handler;
     * field objects representing the input data or solutions of PDEs with this
     * skeleton will contain references to this DoF handler.
     */
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
