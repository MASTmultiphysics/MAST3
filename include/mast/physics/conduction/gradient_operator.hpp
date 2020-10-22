/*
* MAST: Multidisciplinary-design Adaptation and Sensitivity Toolkit
* Copyright (C) 2013-2020  Manav Bhatia and MAST authors
*
* This library is free software; you can redistribute it and/or
* modify it under the terms of the GNU Lesser General Public
* License as published by the Free Software Foundation; either
* version 2.1 of the License, or (at your option) any later version.
*
* This library is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
* Lesser General Public License for more details.
*
* You should have received a copy of the GNU Lesser General Public
* License along with this library; if not, write to the Free Software
* Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
*/

#ifndef __mast_conduction_gradient_operator_h__
#define __mast_conduction_gradient_operator_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/fem_operator_matrix.hpp>


namespace MAST {
namespace Physics {
namespace Conduction {
namespace GradientOperator {


template <typename NodalScalarType,
          typename VarScalarType,
          typename FEVarType,
          uint_t Dim>
inline void
gradient_operator(const FEVarType                                    &fe_var,
                  const uint_t                                        qp,
                  typename Eigen::Matrix<VarScalarType, Dim, 1>      &epsilon,
                  MAST::Numerics::FEMOperatorMatrix<NodalScalarType> &Bmat) {
    
    epsilon.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(epsilon.size() == Dim,
            epsilon.size(),
            "Invalid gradient dimension");
    Assert1(Bmat.m() == Dim,
            Bmat.m(),
            "Invalid gradient dimension");
    Assert2(Bmat.n() == fe.n_basis(),
            Bmat.n(), fe.n_basis(),
            "Incompatible Operator size.");
    
    
    for (uint_t i=0; i<Dim; i++) {

        Bmat.set_shape_function(i, 0, fe.dphi_dx(qp, i)); //  du/dxi
        epsilon(i) = fe_var.du_dx(qp, 0, i);  // du/dxi
    }
}

}  // namespace GradientOperator
}  // namespace Conduction
}  // namespace Physics
}  // namespace MAST


#endif // __mast_linear_elastic_strain_operator_h__
