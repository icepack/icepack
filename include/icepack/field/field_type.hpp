
#ifndef ICEPACK_FIELD_TYPE_HPP
#define ICEPACK_FIELD_TYPE_HPP

#include <fstream>

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>

#include <icepack/discretization.hpp>

namespace icepack {

  using dealii::Tensor;
  using dealii::Function;
  using dealii::TensorFunction;
  using dealii::VectorFunctionFromTensorFunction;

  using dealii::FiniteElement;
  using dealii::FEValues;

  using dealii::update_values;
  using dealii::update_quadrature_points;
  using dealii::update_JxW_values;

  using dealii::Vector;
  using dealii::SmartPointer;

  using dealii::ExcInternalError;


  // This is for template magic, nothing to see here, move along folks...
  namespace {
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


  // This is for the kind of template magic that gets you burned at the stake.
  template <int rank, int dim, class Expr>
  class FieldExpr
  {
  public:
    operator const Expr&() const
    {
      return static_cast<const Expr&>(*this);
    }

    double coefficient(const size_t i) const
    {
      return static_cast<const Expr&>(*this).coefficient(i);
    }
  };


  /**
   * This is a base class for any physical field discretized using a finite
   * element expansion. It is used as the return and argument types of all
   * glacier model objects (see `include/icepack/glacier_models`).
   */
  template <int rank, int dim>
  class FieldType : public FieldExpr<rank, dim, FieldType<rank, dim> >
  {
  public:
    // Type aliases; these are for template magic.

    /**
     * The `value_type` for a scalar field is a real number, for a vector field
     * a rank-1 tensor, and so on and so forth. The member class `tensor_type`
     * of the `dealii::Tensor` class template is aliases the right value type,
     * i.e. it reduces to `double` for rank 0.
     */
    using value_type = typename Tensor<rank, dim>::tensor_type;

    /**
     * Same considerations as for `value_type` but for the gradient, i.e. the
     * gradient type of a scalar field is rank-1 tensor, for a vector field
     * rank-2 tensor, etc.
     */
    using gradient_type = typename Tensor<rank + 1, dim>::tensor_type;

    /**
     * This typename aliases the right finite element values extractor for the
     * given field type, i.e. scalar fields need `FEValuesExtractors::Scalar`,
     * vector fields need `FEValuesExtractors::Vector`.
     */
    using extractor_type = typename ExtractorType<rank>::type;


    /**
     * Construct a field which is 0 everywhere given the data about its finite
     * element discretization.
     */
    FieldType(const Discretization<dim>& discretization)
      :
      discretization(&discretization),
      coefficients(get_field_discretization().get_dof_handler().n_dofs())
    {
      coefficients = 0;
    }


    /**
     * Copy the values of another field. Note that the discretization member is
     * just a `dealii::SmartPointer`, so this copies the address of the object
     * and not its contents.
     *
     * This method allocates memory and should be used sparingly, hence the
     * explicit keyword.
     */
    explicit FieldType(const FieldType<rank, dim>& phi)
      :
      discretization(phi.discretization),
      coefficients(phi.coefficients)
    {}


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
      discretization(phi.discretization),
      coefficients(std::move(phi.coefficients))
    {
      phi.discretization = nullptr;
      phi.coefficients.reinit(0);
    }


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
      discretization = phi.discretization;
      coefficients = std::move(phi.coefficients);

      phi.discretization = nullptr;
      phi.coefficients.reinit(0);

      return *this;
    }


    /**
     * Copy assignment operator
     */
    FieldType<rank, dim>& operator=(const FieldType<rank, dim>& phi)
    {
      discretization = phi.discretization;
      coefficients = phi.coefficients;

      return *this;
    }


    // Destructor. FieldType doesn't directly own any heap-allocated memory,
    // although the Vector member coefficients does, so the dtor is trivial.
    virtual ~FieldType()
    {}


    /**
     * Return the underlying discretization of this field.
     */
    const Discretization<dim>& get_discretization() const
    {
      return *discretization;
    }


    /**
     * Return the `FieldDiscretization` for this field, i.e. a scalar
     * discretization of a scalar field, etc.
     */
    const FieldDiscretization<rank, dim>& get_field_discretization() const
    {
      return (*discretization).field_discretization(fe_field<rank, dim>());
    }


    /**
     * Return whether or not another field uses the same discretization; the
     * comparison is at the level of pointer equality, i.e. they must point to
     * the same object in memory.
     */
    template <int rank_>
    bool has_same_discretization(const FieldType<rank_, dim>& phi) const
    {
      return (const Discretization<dim>*) discretization
        == (const Discretization<dim>*) phi.discretization;
    }


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
     * Return the underlying geometry of the field
     */
    const Triangulation<dim>& get_triangulation() const
    {
      return get_discretization().get_triangulation();
    }


    /**
     * Return the `dealii::FiniteElement` object used to discretize the field.
     */
    const FiniteElement<dim>& get_fe() const
    {
      return get_field_discretization().get_fe();
    }


    /**
     * Return the degree-of-freedom handler for the field. This object can be
     * used to iterate over all the degrees of freedom  of the field.
     */
    const DoFHandler<dim>& get_dof_handler() const
    {
      return get_field_discretization().get_dof_handler();
    }


    /**
     * Return the constraints on degrees of freedom imposed by mesh refinement
     */
    const ConstraintMatrix& get_constraints() const
    {
      return get_field_discretization().get_constraints();
    }


    /**
     * Implement coefficient access so that fields can trivially function as
     * as field expressions.
     */
    double coefficient(const size_t i) const
    {
      return coefficients(i);
    }


    /**
     * Assign a field from an algebraic expression
     */
    template <class Expr>
    FieldType<rank, dim>& operator=(const FieldExpr<rank, dim, Expr>& expr)
    {
      for (size_t k = 0; k < coefficients.size(); ++k)
        coefficients[k] = expr.coefficient(k);

      return *this;
    }


    /**
     * Write out the field to a file in the `.ucd` format. See `scripts/` for
     * Python modules to read meshes and data in this format.
     */
    void write(const std::string& filename, const std::string& name) const
    {
      std::ofstream output(filename.c_str());

      dealii::DataOut<dim> data_out;
      data_out.attach_dof_handler(get_dof_handler());

      std::vector<std::string> component_names;
      if (rank == 1)
        for (unsigned int k = 0; k < dim; ++k)
          component_names.push_back(name + "_" + std::to_string(k+1));
      else
        component_names.push_back(name);

      data_out.add_data_vector(coefficients, component_names);
      data_out.build_patches();
      data_out.write_ucd(output);
    }

  protected:
    /**
     * Reference to the Discretization object which aggregates all of the data
     * needed to define a finite element discretization.
     */
    SmartPointer<const Discretization<dim>> discretization;

    /**
     * Coefficients of the finite element expansion of the field.
     */
    Vector<double> coefficients;
  };


  // Scalar and vector fields are specializations of FieldType with ranks 0, 1.
  template <int dim> using Field = FieldType<0, dim>;
  template <int dim> using VectorField = FieldType<1, dim>;


  /**
   * Compute the L2-norm of a finite element field
   */
  template <int rank, int dim>
  double norm(const FieldType<rank, dim>& phi)
  {
    const auto& fe = phi.get_fe();
    const QGauss<dim> quad = phi.get_discretization().quad();

    FEValues<dim> fe_values(
      fe, quad, update_values|update_JxW_values|update_quadrature_points
    );

    const unsigned int n_q_points = quad.size();
    std::vector<typename FieldType<rank, dim>::value_type>
      phi_values(n_q_points);

    const typename FieldType<rank, dim>::extractor_type ex(0);

    double N = 0.0;
    for (auto cell: phi.get_dof_handler().active_cell_iterators()) {
      fe_values.reinit(cell);
      fe_values[ex].get_function_values(phi.get_coefficients(), phi_values);

      for (unsigned int q = 0; q < n_q_points; ++q) {
        const double dx = fe_values.JxW(q);
        const auto phi_q = phi_values[q];
        N += (phi_q * phi_q) * dx;
      }
    }

    return std::sqrt(N);
  }


  /**
   * Compute the RMS average of a finite-element field
   */
  template <int rank, int dim>
  double rms_average(const FieldType<rank, dim>& phi)
  {
    const double N = norm(phi);
    const double area = dealii::GridTools::volume(phi.get_triangulation());
    return std::sqrt(N * N / area);
  }


  /**
   * Compute the L2-distance between two finite element fields
   */
  template <int rank, int dim>
  double dist(const FieldType<rank, dim>& phi1, const FieldType<rank, dim>& phi2)
  {
    Assert(phi1.has_same_discretization(phi2), ExcInternalError());

    const auto& fe = phi1.get_fe();
    const QGauss<dim> quad = phi1.get_discretization().quad();

    FEValues<dim> fe_values(
      fe, quad, update_values | update_JxW_values | update_quadrature_points
    );

    const unsigned int n_q_points = quad.size();
    using value_type = typename FieldType<rank, dim>::value_type;
    std::vector<value_type> phi1_values(n_q_points), phi2_values(n_q_points);

    const typename FieldType<rank, dim>::extractor_type ex(0);

    double N = 0.0;
    for (auto cell: phi1.get_dof_handler().active_cell_iterators()) {
      fe_values.reinit(cell);
      fe_values[ex].get_function_values(phi1.get_coefficients(), phi1_values);
      fe_values[ex].get_function_values(phi2.get_coefficients(), phi2_values);

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
