
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


  /**
   * This is a base class for any physical field discretized using a finite
   * element expansion. It is used as the return and argument types of all
   * glacier model objects (see `include/icepack/glacier_models`).
   */
  template <int rank, int dim>
  class FieldType
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
     * Default constructor for an empty field object with no geometry, FE, etc.
     */
    FieldType()
      :
      discretization(nullptr),
      coefficients(0)
    {}


    /**
     * Construct a field given the geometry, discretization and mapping of
     * geometry to FE degrees of freedom. The field object does not own its
     * `dealii::DoFHandler`, which may be shared among several fields;
     * consequently, this data must be passed in to the constructor.
     * Initializes the field to 0.
     * You should not have to call this constructor in normal usage. Instead,
     * use the `interpolate` member of the model object for the problem.
     */
    FieldType(const Discretization<dim>& discretization)
      :
      discretization(&discretization),
      coefficients(get_field_discretization().get_dof_handler().n_dofs())
    {
      coefficients = 0;
    }

    /**
     * Delete the copy constructor; copying a field object should only be done
     * if the user asks for it explicitly (see `copy_from`).
     */
    FieldType(const FieldType<rank, dim>&) = delete;

    /**
     * Explicitly copy a field; this replaces the functionality of the copy
     * constructor.
     */
    void copy_from(const FieldType<rank, dim>& phi)
    {
      // This is a `dealii::SmartPointer` to the object in question, so the
      // assignment just copies the address and not the actual object.
      discretization = phi.discretization;

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
   * Given a dealii::Function or TensorFunction object, return the function's
   * finite element interpolation as a Field or VectorField respectively.
   */
  template <int dim>
  Field<dim> interpolate(
    const Discretization<dim>& discretization,
    const Function<dim>& phi
  )
  {
    Field<dim> psi(discretization);
    dealii::VectorTools::interpolate(
      psi.get_dof_handler(), phi, psi.get_coefficients()
    );
    return psi;
  }


  /**
   * Overload of `interpolate` for VectorField
   */
  template <int dim>
  VectorField<dim> interpolate(
    const Discretization<dim>& discretization,
    const TensorFunction<1, dim>& phi
  )
  {
    VectorField<dim> psi(discretization);
    const VectorFunctionFromTensorFunction<dim> vphi(phi);
    dealii::VectorTools::interpolate(
      psi.get_dof_handler(), vphi, psi.get_coefficients()
    );
    return psi;
  }


  /**
   * Create a map of degree-of-freedom indices to numbers describing how to
   * create a field with the same boundary values as `phi` on boundary vertices
   * with the id `boundary_id`
   */
  template <int rank, int dim>
  std::map<dealii::types::global_dof_index, double>
  interpolate_boundary_values(
    const FieldType<rank, dim>& phi,
    const unsigned int boundary_id = 0
  )
  {
    auto boundary_values =
      phi.get_field_discretization().zero_boundary_values(boundary_id);

    const Vector<double>& Phi = phi.get_coefficients();
    for (auto& p: boundary_values) p.second = Phi(p.first);

    return boundary_values;
  }


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
   * Compute the L2-distance between two finite element fields
   */
  template <int rank, int dim>
  double dist(const FieldType<rank, dim>& phi1, const FieldType<rank, dim>& phi2)
  {
    //TODO: add some error handling to make sure both fields are defined with
    // the same FE discretization

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
