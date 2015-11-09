
#ifndef ICEPACK_FIELD_HPP
#define ICEPACK_FIELD_HPP

#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

namespace icepack
{
  using dealii::Tensor;
  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::VectorFunctionFromTensorFunction;
  using dealii::Triangulation;
  using dealii::FiniteElement;
  using dealii::DoFHandler;
  using dealii::Vector;
  using dealii::SmartPointer;

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
                const FiniteElement<dim>& _fe,
                const DoFHandler<dim>& _dof_handler)
        :
        triangulation(_triangulation),
        fe(_fe),
        dof_handler(&_dof_handler)
      {
        coefficients.reinit(dof_handler->n_dofs());

        // TODO: put in some asserts to make sure that the FiniteElement object
        // supplied is compatible with the field type (scalar vs. vector)
      }

      // Copy constructor
      FieldType(const FieldType<rank, dim>& phi)
        :
        triangulation(phi.triangulation),
        fe(phi.fe),
        dof_handler(phi.dof_handler),
        coefficients(phi.coefficients)
      {}

      // Move constructor
      FieldType(FieldType<rank, dim>&& phi)
        :
        triangulation(phi.triangulation),
        fe(phi.fe),
        dof_handler(std::move(phi.dof_handler)),
        coefficients(std::move(phi.coefficients))
      {}

      virtual ~FieldType()
      {}


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


      // File I/O
      bool write(const std::string& filename,
                 const std::string& field_name)
      {
        std::ofstream output(filename.c_str());

        dealii::DataOut<dim> data_out;
        data_out.attach_dof_handler(*dof_handler);

        std::vector<std::string> component_names;
        if (rank == 1)
          for (unsigned int k = 0; k < dim; ++k)
            component_names.push_back(field_name + "_" + std::to_string(k+1));
        else
          component_names.push_back(field_name);

        data_out.add_data_vector(coefficients, component_names);
        data_out.build_patches();
        data_out.write_ucd(output);

        return true;
      }


    protected:
      const Triangulation<dim>& triangulation;
      const FiniteElement<dim>& fe;
      SmartPointer<const DoFHandler<dim> > dof_handler;
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
                         const DoFHandler<dim>& dof_handler,
                         const Function<dim>& phi)
  {
    Field<dim> psi(triangulation, finite_element, dof_handler);

    dealii::VectorTools::interpolate(
      psi.get_dof_handler(),
      phi,
      psi.get_coefficients()
    );

    return psi;
  }


  template <int dim>
  VectorField<dim> interpolate(const Triangulation<dim>& triangulation,
                               const FiniteElement<dim>& finite_element,
                               const DoFHandler<dim>& dof_handler,
                               const TensorFunction<1, dim>& phi)
  {
    VectorField<dim> psi(triangulation, finite_element, dof_handler);

    const VectorFunctionFromTensorFunction<dim> vphi(phi);
    dealii::VectorTools::interpolate(
      psi.get_dof_handler(),
      vphi,
      psi.get_coefficients()
    );

    return psi;
  }

}

#endif
