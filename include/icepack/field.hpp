
#ifndef ICEPACK_FIELD_HPP
#define ICEPACK_FIELD_HPP

#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/data_out.h>

namespace icepack {

  using dealii::Tensor;
  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::VectorFunctionFromTensorFunction;

  using dealii::Triangulation;
  using dealii::QGauss;
  using dealii::FiniteElement;
  using dealii::DoFHandler;
  using dealii::FEValues;

  using dealii::update_values;
  using dealii::update_quadrature_points;
  using dealii::update_JxW_values;

  using dealii::Vector;
  using dealii::SmartPointer;


  // This is for template magic, nothing to see here, move along folks...
  namespace
  {
    template <int rank> struct ExtractorType
    {
      using type = dealii::FEValuesExtractors::Tensor<rank>;
    };

    template <> struct ExtractorType<0>
    {
      using type = dealii::FEValuesExtractors::Scalar;
    };

    template <> struct ExtractorType<1>
    {
      using type = dealii::FEValuesExtractors::Vector;
    };
  }


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
    using extractor_type = typename ExtractorType<rank>::type;

    // Constructors & destructors
    FieldType()
      :
      triangulation(nullptr),
      fe(nullptr),
      dof_handler(nullptr),
      coefficients(0)
    {}

    FieldType(
      const Triangulation<dim>& _triangulation,
      const FiniteElement<dim>& _fe,
      const DoFHandler<dim>& _dof_handler
    )
      :
      triangulation(&_triangulation),
      fe(&_fe),
      dof_handler(&_dof_handler)
    {
      coefficients.reinit(dof_handler->n_dofs());

      // TODO: put in some asserts to make sure that the FiniteElement object
      // supplied is compatible with the field type (scalar vs. vector)
    }

    /**
     * Delete the copy constructor.
     */
    FieldType(const FieldType<rank, dim>&) = delete;

    /**
     * Explicitly copy a field; this replaces the functionality of the copy
     * constructor.
     */
    void copy_from(const FieldType<rank, dim>& phi)
    {
      // These are all dealii::SmartPointers to the object in question, so the
      // assignment just copies the address and not the actual object.
      triangulation = phi.triangulation;
      fe = phi.fe;
      dof_handler = phi.dof_handler;

      // This actually copies the vector.
      coefficients = phi.coefficients;
    }

    /**
     * Move constructor. This allows fields to be returned from functions,
     * so that one can write things like
     *
     *     Field<dim> u = solve_pde(kappa, f);
     *
     * without an expensive and unnecessary copy operation by using C++11
     * move semantics.
     */
    FieldType(FieldType<rank, dim>&& phi)
      :
      triangulation(phi.triangulation),
      fe(phi.fe),
      dof_handler(phi.dof_handler),
      coefficients(std::move(phi.coefficients))
    {}

    /**
     * Move assignment operator. Like the move constructor, this allows fields
     * to be returned from functions, even after their declaration:
     *
     *     Field<dim> kappa = initial_guess();
     *     Field<dim> u = solve_pde(kappa, f);
     *     kappa = update_guess(u);
     *
     * This functionality is useful when solving nonlinear PDE iteratively.
     */
    FieldType<rank, dim>& operator=(FieldType<rank, dim>&& phi)
    {
      triangulation = phi.triangulation;
      fe = phi.fe;
      dof_handler = phi.dof_handler;

      coefficients = std::move(phi.coefficients);

      return *this;
    }

    // Destructor. FieldType doesn't directly own any heap-allocated memory,
    // although the Vector member coefficients does, so the dtor is trivial.
    virtual ~FieldType()
    {}


    /**
     * Return const access to the coefficients of the finite element expansion
     * of the field.
     */
    const Vector<double>& get_coefficients() const
    {
      return coefficients;
    }

    /**
     * Return non-const access to the coefficients of the field, e.g. so that a
     * PDE solver can set or update the field values.
     */
    Vector<double>& get_coefficients()
    {
      return coefficients;
    }

    /**
     * Return the `dealii::FiniteElement` object used to discretize the field.
     */
    const FiniteElement<dim>& get_fe() const
    {
      return *fe;
    }

    /**
     * Return the degree-of-freedom handler for the field. This object can be
     * used to iterate over all the degrees of freedom  of the field.
     */
    const DoFHandler<dim>& get_dof_handler() const
    {
      return *dof_handler;
    }


    /**
     * Write out the field to a file in the `.ucd` format. See `scripts/` for
     * Python modules to read meshes and data in this format.
     */
    bool write(const std::string& filename, const std::string& field_name)
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
    SmartPointer<const Triangulation<dim> > triangulation;
    SmartPointer<const FiniteElement<dim> > fe;
    SmartPointer<const DoFHandler<dim> > dof_handler;
    Vector<double> coefficients;
  };


  // Scalar and vector fields are specializations of FieldType with ranks 0, 1.
  template <int dim> using Field = FieldType<0, dim>;
  template <int dim> using VectorField = FieldType<1, dim>;



  /**
   * Given a dealii::Function or TensorFunction object, return the function's
   * finite element interpolation as a Field or VectorField respectively.
   */
  template <int dim>
  Field<dim> interpolate(
    const Triangulation<dim>& triangulation,
    const FiniteElement<dim>& finite_element,
    const DoFHandler<dim>& dof_handler,
    const Function<dim>& phi
  )
  {
    Field<dim> psi(triangulation, finite_element, dof_handler);

    dealii::VectorTools::interpolate(
      psi.get_dof_handler(),
      phi,
      psi.get_coefficients()
    );

    return psi;
  }


  /**
   * Overload of `interpolate` for VectorField
   */
  template <int dim>
  VectorField<dim> interpolate(
    const Triangulation<dim>& triangulation,
    const FiniteElement<dim>& finite_element,
    const DoFHandler<dim>& dof_handler,
    const TensorFunction<1, dim>& phi
  )
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


  /**
   * Compute the L2-norm of a finite element field
   */
  template <int rank, int dim>
  double norm(const FieldType<rank, dim>& phi)
  {
    const auto& dof_handler = phi.get_dof_handler();
    const auto& fe = phi.get_fe();
    const Vector<double>& Phi = phi.get_coefficients();

    const unsigned int p = fe.tensor_degree();
    const QGauss<dim> quad(p);

    FEValues<dim> fe_values(
      fe, quad, update_values | update_JxW_values | update_quadrature_points
    );

    const unsigned int n_q_points = quad.size();
    std::vector<typename FieldType<rank, dim>::value_type> phi_values(n_q_points);

    const typename FieldType<rank, dim>::extractor_type extractor(0);

    double N = 0.0;
    for (auto cell: dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      fe_values[extractor].get_function_values(Phi, phi_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = fe_values.JxW(q);
        const auto phi_q = phi_values[q];
        N += (phi_q * phi_q) * dx;
      }
    }

    return std::sqrt(N);
  }


  /**
   * Compute the L2-distance between two finite element fields
   */
  template <int rank, int dim>
  double dist(const FieldType<rank, dim>& phi1, const FieldType<rank, dim>& phi2)
  {
    const auto& dof_handler = phi1.get_dof_handler();
    const auto& fe = phi1.get_fe();
    const Vector<double>& Phi1 = phi1.get_coefficients();
    const Vector<double>& Phi2 = phi2.get_coefficients();

    //TODO: add some error handling to make sure both fields are defined with
    // the same FE discretization

    const unsigned int p = fe.tensor_degree();
    const QGauss<dim> quad(p);

    FEValues<dim> fe_values(
      fe, quad, update_values | update_JxW_values | update_quadrature_points
    );

    const unsigned int n_q_points = quad.size();
    using value_type = typename FieldType<rank, dim>::value_type;
    std::vector<value_type> phi1_values(n_q_points), phi2_values(n_q_points);

    const typename FieldType<rank, dim>::extractor_type extractor(0);

    double N = 0.0;
    for (auto cell: dof_handler.active_cell_iterators()) {
      fe_values.reinit(cell);
      fe_values[extractor].get_function_values(Phi1, phi1_values);
      fe_values[extractor].get_function_values(Phi2, phi2_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = fe_values.JxW(q);
        const auto diff_q = phi1_values[q] - phi2_values[q];
        N += (diff_q * diff_q) * dx;
      }
    }

    return std::sqrt(N);
  }

}

#endif
