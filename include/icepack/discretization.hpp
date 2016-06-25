
#ifndef ICEPACK_DISCRETIZATION_HPP
#define ICEPACK_DISCRETIZATION_HPP

#include <tuple>
#include <memory>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/synchronous_iterator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/lac/sparsity_pattern.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>

namespace icepack {

  using dealii::QGauss;
  using dealii::Triangulation;
  using dealii::FE_Q;
  using dealii::FESystem;
  using dealii::DoFHandler;
  using dealii::SparsityPattern;
  using dealii::DynamicSparsityPattern;
  using dealii::ConstraintMatrix;
  using dealii::SparseMatrix;
  using dealii::SmartPointer;
  using dealii::SynchronousIterators;
  using dealii::DoFTools::make_hanging_node_constraints;
  using dealii::DoFTools::make_sparsity_pattern;

  /**
   * Default update flags for `dealii::FEValues` objects when iterating over
   * the degrees of freedom of a finite element field.
   */
  namespace DefaultUpdateFlags {
    using dealii::UpdateFlags;

    extern const UpdateFlags flags;
    extern const UpdateFlags face_flags;
  }


  // This struct is just for getting the right finite element class for vector
  // or scalar discretizations as the case may be.
  namespace internal {
    template <int rank, int dim> struct fe_field;

    template <int dim> struct fe_field<0, dim>
    {
      using FE = FE_Q<dim>;
      static FE fe(const size_t p) { return FE(p); }
    };

    template <int dim> struct fe_field<1, dim>
    {
      using FE = FESystem<dim>;
      static FE fe(const size_t p) { return FE(FE_Q<dim>(p), dim); }
    };
  }


  /**
   * This class encapsulates all the data needed to discretize finite element
   * fields of a particular tensor rank, e.g. scalar, vector or tensor fields.
   */
  template <int rank, int dim>
  class FieldDiscretization
  {
  public:
    using FE = typename internal::fe_field<rank, dim>::FE;

    /*
     * Constructors & destructor
     */

    FieldDiscretization(const Triangulation<dim>& tria, const unsigned int p)
      :
      fe(internal::fe_field<rank, dim>::fe(p)),
      dof_handler(tria)
    {
      dof_handler.distribute_dofs(fe);

      constraints.clear();
      make_hanging_node_constraints(dof_handler, constraints);
      constraints.close();

      DynamicSparsityPattern dsp(dof_handler.n_dofs());
      make_sparsity_pattern(dof_handler, dsp, constraints, false);
      sparsity.copy_from(dsp);

      // TODO: lazy initialization
      mass_matrix.reinit(sparsity);
      dealii::MatrixCreator::create_mass_matrix(
        dof_handler,
        QGauss<dim>(p + 1),
        mass_matrix,
        (const dealii::Function<dim> * const)nullptr,
        constraints
      );
    }

    FieldDiscretization(const FieldDiscretization<rank, dim>&) = delete;

    ~FieldDiscretization()
    {
      dof_handler.clear();
    }


    /*
     * Boundary value maps
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

      return boundary_values;
    }


    /*
     * Accessors
     */

    const FE& get_fe() const { return fe; }
    const DoFHandler<dim>& get_dof_handler() const { return dof_handler; }
    const SparsityPattern& get_sparsity() const { return sparsity; }
    const ConstraintMatrix& get_constraints() const { return constraints; }
    const SparseMatrix<double>& get_mass_matrix() const { return mass_matrix; }

  protected:
    FE fe;
    DoFHandler<dim> dof_handler;
    SparsityPattern sparsity;
    ConstraintMatrix constraints;
    SparseMatrix<double> mass_matrix;
  };


  /**
   * This class encapsulates all the data needed to discretize finite element
   * fields of any tensor rank.
   *
   * In practice, we only need ranks 0 and 1. This class acts as an aggregate
   * of FieldDiscretization objects for each rank.
   */
  template <int dim>
  class Discretization : public dealii::Subscriptor
  {
  public:
    using Scalar = FieldDiscretization<0, dim>;
    using Vector = FieldDiscretization<1, dim>;

    Discretization(const Triangulation<dim>& tria, const unsigned int p)
      :
      p(p),
      tria(&tria),
      ranks(std::unique_ptr<Scalar>(new Scalar(tria, p)),
            std::unique_ptr<Vector>(new Vector(tria, p)))
    {}

    Discretization(const Discretization<dim>&) = delete;

    const Scalar& scalar() const
    {
      return *std::get<0>(ranks);
    }

    const Vector& vector() const
    {
      return *std::get<1>(ranks);
    }

    const Triangulation<dim>& get_triangulation() const
    {
      return *tria;
    }

    QGauss<dim> quad() const
    {
      return QGauss<dim>(p + 1);
    }

    QGauss<dim-1> face_quad() const
    {
      return QGauss<dim - 1>(p + 1);
    }

    unsigned int degree() const
    {
      return p;
    }

    template <int rank, int _dim> friend
    const FieldDiscretization<rank, _dim>& get(const Discretization<_dim>& dsc);


    /*
     * Iterating over both scalar and vector fields
     */

    using dof_iterator = typename DoFHandler<dim>::active_cell_iterator;
    using iterators = std::tuple<dof_iterator, dof_iterator>;
    using iterator = SynchronousIterators<iterators>;

    iterator begin() const
    {
      iterators its = {scalar().get_dof_handler().begin_active(),
                       vector().get_dof_handler().begin_active()};
      return iterator(its);
    }

    iterator end() const
    {
      iterators its = {scalar().get_dof_handler().end(),
                       vector().get_dof_handler().end()};
      return iterator(its);
    }

    const dof_iterator& scalar_cell_iterator(const iterators& its) const
    {
      return std::get<0>(its);
    }

    const dof_iterator& vector_cell_iterator(const iterators& its) const
    {
      return std::get<1>(its);
    }

  protected:
    const unsigned int p;
    SmartPointer<const Triangulation<dim>> tria;
    std::tuple<std::unique_ptr<Scalar>, std::unique_ptr<Vector>> ranks;
  };


  template <int rank, int dim>
  const FieldDiscretization<rank, dim>& get(const Discretization<dim>& dsc)
  {
    return *std::get<rank>(dsc.ranks);
  }

} // End of namespace icepack

#endif
