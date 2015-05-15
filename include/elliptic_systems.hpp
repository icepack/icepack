
#ifndef ELLIPTIC_SYSTEMS_HPP
#define ELLIPTIC_SYSTEMS_HPP


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>


namespace EllipticSystems
{
  using dealii::Vector;
  using dealii::Tensor;
  using dealii::SymmetricTensor;
  using dealii::FESystem;
  using dealii::FEValues;
  using dealii::FEFaceValues;
  using dealii::FullMatrix;

  template <int dim>
  SymmetricTensor<4, dim>
  stress_strain_tensor (const double lambda, const double mu)
  {
    SymmetricTensor<4,dim> C;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int k = 0; k < dim; ++k)
          for (unsigned int l = 0; l < dim; ++l)
            C[i][j][k][l] = (((i==k) && (j==l) ? mu : 0.0) +
                             ((i==l) && (j==k) ? mu : 0.0) +
                             ((i==j) && (k==l) ? lambda : 0.0));
    return C;
  }


  template <int dim>
  inline
  SymmetricTensor<2, dim>
  get_strain (const FEValues<dim>& fe_values,
              const unsigned int   shape_func,
              const unsigned int   q_point)
  {
    SymmetricTensor<2, dim> strain;

    for (unsigned int i = 0; i < dim; ++i)
      strain[i][i] = fe_values.shape_grad_component (shape_func, q_point, i)[i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i+1; j < dim; ++j)
        strain[i][j]
          = (fe_values.shape_grad_component (shape_func, q_point, i)[j] +
             fe_values.shape_grad_component (shape_func, q_point, j)[i]) / 2;

    return strain;
  }


  template <int dim>
  inline
  SymmetricTensor<2, dim>
  get_strain (const std::vector<Tensor<1, dim> >& grad)
  {
    // Put an Assert in here

    SymmetricTensor<2, dim> strain;
    for (unsigned int i = 0; i < dim; ++i) strain[i][i] = grad[i][i];

    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = i + 1; j < dim; ++j)
        strain[i][j] = (grad[i][j] + grad[j][i]) / 2;

    return strain;
  }



  template <int dim>
  inline
  void fill_cell_matrix (FullMatrix<double>& cell_matrix,
                         const SymmetricTensor<4, dim>& stress_strain,
                         const FEValues<dim>& fe_values,
                         const unsigned int q_point,
                         const unsigned int dofs_per_cell)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          const SymmetricTensor<2, dim>
          eps_phi_i = get_strain (fe_values, i, q_point),
          eps_phi_j = get_strain (fe_values, j, q_point);

          cell_matrix(i, j)
            += (eps_phi_i * stress_strain * eps_phi_j)
               *
               fe_values.JxW (q_point);
        }
  }



  template <int dim>
  inline
  void fill_cell_rhs_field (Vector<double>& cell_rhs,
                            const Tensor<1, dim>& field_value,
                            const FESystem<dim>& fe,
                            const FEValues<dim>& fe_values,
                            const unsigned int q_point,
                            const unsigned int dofs_per_cell)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;
        cell_rhs(i) += fe_values.shape_value(i, q_point) *
                       field_value[component_i] *
                       fe_values.JxW(q_point);
      }
  }



  template <int dim>
  inline
  void fill_cell_rhs_neumann (Vector<double>& cell_rhs,
                              const Tensor<1, dim>& neumann_value,
                              const FESystem<dim>& fe,
                              const FEFaceValues<dim>& fe_face_values,
                              const unsigned int q_point,
                              const unsigned int dofs_per_cell)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        const unsigned int component_i = fe.system_to_component_index(i).first;
        cell_rhs(i) += neumann_value[component_i] *
                       fe_face_values.shape_value(i, q_point) *
                       fe_face_values.JxW(q_point);
      }
  }

} // End of EllipticSystems namespace



#endif
