
#ifndef ICEPACK_FIELD_HPP
#define ICEPACK_FIELD_HPP

#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

namespace icepack
{
  using dealii::Tensor;
  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::Triangulation;
  using dealii::FiniteElement;
  using dealii::DoFHandler;
  using dealii::Vector;
  namespace VectorTools = dealii::VectorTools;


  namespace
  {
    /**
     * Base class for any physical field discretized by finite elements.
     */
    template <int rank, int dim>
    class FieldType
    {
    public:
      // Type aliases; these are for template magic.
      using value_type = typename Tensor<rank, dim>::tensor_type;
      using gradient_type = typename Tensor<rank + 1, dim>::tensor_type;


      // Constructors & destructors
      FieldType(const Triangulation<dim>& _triangulation,
                const FiniteElement<dim>& _fe)
        :
        triangulation(_triangulation),
        fe(_fe),
        dof_handler(new DoFHandler<dim>(triangulation))
      {
        dof_handler->distribute_dofs(fe);
        coefficients.reinit(dof_handler->n_dofs());

        // TODO: put in some asserts to make sure that the FiniteElement object
        // supplied is compatible with the field type (scalar vs. vector)
      }

      FieldType(FieldType<rank, dim>&& phi)
        :
        triangulation(phi.triangulation),
        fe(phi.fe),
        dof_handler(std::move(phi.dof_handler)),
        coefficients(phi.coefficients)
      {}

      virtual ~FieldType()
      {
        dof_handler->clear();
      }


      // Accessors to raw data
      const Vector<double>& get_coefficients() const
      {
        return coefficients;
      }

      Vector<double>& get_coefficients()
      {
        return coefficients;
      }

      const FiniteElement<dim>& get_fe() const
      {
        return fe;
      }

      const DoFHandler<dim>& get_dof_handler() const
      {
        return *dof_handler;
      }


    protected:
      const Triangulation<dim>& triangulation;
      const FiniteElement<dim>& fe;
      std::unique_ptr<DoFHandler<dim> > dof_handler;
      Vector<double> coefficients;
    };
  }


  // Scalar and vector fields are specializations of FieldType with ranks 0, 1.
  template <int dim> using Field = FieldType<0, dim>;
  template <int dim> using VectorField = FieldType<1, dim>;



  /**
   * Given a dealii::Function or TensorFunction object, return the function's
   * finite element interpolation as a Field or VectorField respectively.
   */
  template <int dim>
  Field<dim> interpolate(const Triangulation<dim>& triangulation,
                         const FiniteElement<dim>& finite_element,
                         const Function<dim>& phi)
  {
    Field<dim> psi(triangulation, finite_element);

    VectorTools::interpolate(
      psi.get_dof_handler(),
      phi,
      psi.get_coefficients()
    );

    return psi;
  }


  template <int dim>
  VectorField<dim> interpolate(const Triangulation<dim>& triangulation,
                               const FiniteElement<dim>& finite_element,
                               const TensorFunction<1, dim>& phi)
  {
    VectorField<dim> psi(triangulation, finite_element);

    VectorTools::interpolate(
      psi.get_dof_handler,
      VectorFunctionFromTensorFunction(phi),
      psi.get_coefficients()
    );

    return psi;
  }

}

#endif
