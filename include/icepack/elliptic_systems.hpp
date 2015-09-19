
#ifndef ELLIPTIC_SYSTEMS_HPP
#define ELLIPTIC_SYSTEMS_HPP


#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/symmetric_tensor.h>

#include <deal.II/lac/full_matrix.h>

#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>

#include "physical_constants.hpp"

namespace EllipticSystems
{
  using dealii::Vector;
  using dealii::Tensor;
  using dealii::SymmetricTensor;
  using dealii::FESystem;
  using dealii::FEValuesBase;
  using dealii::FullMatrix;
  using dealii::SparseMatrix;
  namespace FEValuesExtractors = dealii::FEValuesExtractors;


  template <int dim>
  class AssembleRHS
  {
  public:
    virtual void operator() (const FEValuesBase<dim>& fe_values,
                             Vector<double>&          cell_rhs) = 0;
    virtual ~AssembleRHS () {};
  };


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
                             ((i==j) && (k==l) ? lambda : 0.0));
    return C;
  }


  struct LinearSSATensor
  {
    SymmetricTensor<4, 2> operator()(const double temperature,
                                     const double h,
                                     const SymmetricTensor<2, 2>) const
    {
      const double nu_q = h * viscosity(temperature, 0.2);
      return stress_strain_tensor<2>(2 * nu_q, 2 * nu_q);
    }
  };


  struct SSATensor
  {
    inline
    SymmetricTensor<4, 2> operator()(const double temperature,
                                     const double h,
                                     const SymmetricTensor<2, 2> eps) const
    {
      const double tr = first_invariant (eps);
      const double eps_e = sqrt(0.5 * (eps * eps + tr * tr));
      const double nu_q = h * viscosity(temperature, eps_e);
      return stress_strain_tensor<2>(2 * nu_q, 2 * nu_q);
    }
  };


  template <int dim>
  inline
  void fill_cell_rhs (Vector<double>& cell_rhs,
                      const Tensor<1, dim>& field_value,
                      const FESystem<dim>& fe,
                      const FEValuesBase<dim>& fe_values,
                      const unsigned int q_point,
                      const unsigned int dofs_per_cell)
  {
    for (unsigned int i = 0; i < dofs_per_cell; ++i) {
      const unsigned int component_i = fe.system_to_component_index(i).first;
      cell_rhs(i) += fe_values.shape_value(i, q_point) *
                     field_value[component_i] *
                     fe_values.JxW(q_point);
    }
  }



  inline
  void cell_to_global (const FullMatrix<double>& cell_matrix,
                       const std::vector<unsigned int>& local_dof_indices,
                       SparseMatrix<double>& system_matrix)
  {
    const unsigned int dofs_per_cell = local_dof_indices.size();

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      for (unsigned int j = 0; j < dofs_per_cell; ++j)
        system_matrix.add (local_dof_indices[i],
                           local_dof_indices[j],
                           cell_matrix(i,j));

  }


  inline
  void cell_to_global (const Vector<double>& cell_rhs,
                       const std::vector<unsigned int>& local_dof_indices,
                       Vector<double>& system_rhs)
  {
    const unsigned int dofs_per_cell = local_dof_indices.size();

    for (unsigned int i = 0; i < dofs_per_cell; ++i)
      system_rhs(local_dof_indices[i]) += cell_rhs(i);
  }

} // End of EllipticSystems namespace



#endif
