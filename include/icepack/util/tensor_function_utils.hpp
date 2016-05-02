
#ifndef ICEPACK_TENSOR_FUNCTION_UTILS
#define ICEPACK_TENSOR_FUNCTION_UTILS

#include <array>

#include <deal.II/base/tensor.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>

namespace icepack {
  namespace internal {

    using dealii::Point;
    using dealii::Tensor;
    using dealii::Function;
    using dealii::TensorFunction;
    using dealii::SmartPointer;

    template <int dim>
    using CoordFunctions = std::array<SmartPointer<const Function<dim> >, dim>;

    /**
     * Fill an array of 2 smart pointers to scalar functions.
     */
    CoordFunctions<2> coord_functions(
      const Function<2>& phi0, const Function<2>& phi1
    );

    /**
     * Fill an array of 3 smart pointers to scalar functions.
     */
    CoordFunctions<3> coord_functions(
      const Function<3>& phi0, const Function<3>& phi1, const Function<3>& phi2
    );


    /**
     * This class creates a `dealii::TensorFunction` object from several scalar
     * `dealii::Function` objects, using each as one coordinate function. This
     * is useful for when we have measured velocity data stored as separate
     * components and we want to combine it into one vector field.
     */
    template <int dim>
    class TensorFunctionFromScalarFunctions : public TensorFunction<1, dim>
    {
    public:
      /**
       * Construct a `TensorFunction` object using several `Function` objects
       * as its coordinates. The template hackery is so we can write one
       * constructor for the cases where we pass in either 2/3 arguments,
       * depeneding on dimension.
       */
      template <typename... Args>
      TensorFunctionFromScalarFunctions(Args&&... args)
        :
        coordinate_functions(coord_functions(std::forward<Args>(args)...))
      {}

      /**
       * Return a `Tensor` value at a point by finding the values of each of
       * the coordinate functions at this point.
       */
      Tensor<1, dim> value(const Point<dim>& p) const
      {
        Tensor<1, dim> v;
        for (size_t k = 0; k < dim; ++k)
          v[k] = coordinate_functions[k]->value(p);
        return v;
      }

    private:
      /**
       * Store a `std::array` of `dealii::SmartPointer` to the coordinate
       * functions.
       */
      CoordFunctions<dim> coordinate_functions;
    };
  }
}


#endif
