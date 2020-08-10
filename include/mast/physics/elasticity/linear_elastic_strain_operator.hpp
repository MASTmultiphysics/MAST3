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

#ifndef __mast_linear_elastic_strain_operator_h__
#define __mast_linear_elastic_strain_operator_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/fem_operator_matrix.hpp>


namespace MAST {
namespace Physics {
namespace Elasticity {
namespace LinearContinuum {

template <uint_t D> struct NStrainComponents { };
template <> struct NStrainComponents<1> { static const uint_t value = 1; };
template <> struct NStrainComponents<2> { static const uint_t value = 3; };
template <> struct NStrainComponents<3> { static const uint_t value = 6; };


template <typename NodalScalarType, typename VarScalarType, typename FEVarType, uint_t Dim>
inline
typename std::enable_if<Dim == 2, void>::type
strain(const FEVarType&                                    fe_var,
       const uint_t                                        qp,
       typename Eigen::Matrix<VarScalarType, 3, 1>&        epsilon,
       MAST::Numerics::FEMOperatorMatrix<NodalScalarType>& Bmat) {
    
    epsilon.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(epsilon.size() == 3,
            epsilon.size(),
            "Strain vector for 2D continuum strain should be 3");
    Assert1(Bmat.m() == 3,
            Bmat.m(),
            "Strain vector for 2D continuum strain should be 3");
    Assert2(Bmat.n() == 2*fe.n_basis(),
            Bmat.n(), 2*fe.n_basis(),
            "Incompatible Operator size.");
    
    
    // linear strain operator
    Bmat.set_shape_function(0, 0, fe.dphi_dx(qp, 0)); //  epsilon_xx = du/dx
    Bmat.set_shape_function(2, 1, fe.dphi_dx(qp, 0)); //  gamma_xy = dv/dx + ...
        
    // linear strain operator
    Bmat.set_shape_function(1, 1, fe.dphi_dx(qp, 1)); //  epsilon_yy = dv/dy
    Bmat.set_shape_function(2, 0, fe.dphi_dx(qp, 1)); //  gamma_xy = du/dy + ...
    
    epsilon(0) = fe_var.du_dx(qp, 0, 0);  // du/dx
    epsilon(1) = fe_var.du_dx(qp, 1, 1);  // dv/dy
    epsilon(2) = fe_var.du_dx(qp, 0, 1) + fe_var.du_dx(qp, 1, 0);  // du/dy + dv/dx
}



template <typename NodalScalarType, typename VarScalarType, typename FEVarType, uint_t Dim>
inline
typename std::enable_if<Dim == 3, void>::type
strain(const FEVarType&                                    fe_var,
       const uint_t                                        qp,
       typename Eigen::Matrix<VarScalarType, 6, 1>&        epsilon,
       MAST::Numerics::FEMOperatorMatrix<NodalScalarType>& Bmat) {
    
    epsilon.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(epsilon.size() == 6,
            epsilon.size(),
            "Strain vector for 3D continuum strain should be 6");
    Assert1(Bmat.m() == 6,
            Bmat.m(),
            "Strain vector for 3D continuum strain should be 6");
    Assert2(Bmat.n() == 3*fe.n_basis(),
            Bmat.n(), 3*fe.n_basis(),
            "Incompatible Operator size.");
    
    
    // linear strain operator
    Bmat.set_shape_function(0, 0, fe.dphi_dx(qp, 0)); //  epsilon_xx = du/dx
    Bmat.set_shape_function(3, 1, fe.dphi_dx(qp, 0)); //  gamma_xy = dv/dx + ...
    Bmat.set_shape_function(5, 2, fe.dphi_dx(qp, 0)); //  gamma_zx = dw/dx + ...

    // linear strain operator
    Bmat.set_shape_function(1, 1, fe.dphi_dx(qp, 1)); //  epsilon_yy = dv/dy
    Bmat.set_shape_function(3, 0, fe.dphi_dx(qp, 1)); //  gamma_xy = du/dy + ...
    Bmat.set_shape_function(4, 2, fe.dphi_dx(qp, 1)); //  gamma_yz = dw/dy + ...

    Bmat.set_shape_function(2, 2, fe.dphi_dx(qp, 2)); //  epsilon_zz = dw/dz
    Bmat.set_shape_function(4, 1, fe.dphi_dx(qp, 2)); //  gamma_yz = dv/dz + ...
    Bmat.set_shape_function(5, 0, fe.dphi_dx(qp, 2)); //  gamma_zx = du/dz + ...

    epsilon(0) = fe_var.du_dx(qp, 0, 0);  // du/dx
    epsilon(1) = fe_var.du_dx(qp, 1, 1);  // dv/dy
    epsilon(2) = fe_var.du_dx(qp, 2, 2);  // dv/dy
    epsilon(3) = fe_var.du_dx(qp, 0, 1) + fe_var.du_dx(qp, 1, 0);  // du/dy + dv/dx
    epsilon(4) = fe_var.du_dx(qp, 1, 2) + fe_var.du_dx(qp, 2, 1);  // dv/dz + dw/dy
    epsilon(5) = fe_var.du_dx(qp, 0, 2) + fe_var.du_dx(qp, 2, 0);  // du/dz + dw/dx
}

}  // namespace LinearContinuum
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST


#endif // __mast_linear_elastic_strain_operator_h__
