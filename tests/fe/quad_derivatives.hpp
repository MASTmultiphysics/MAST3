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

#ifndef __mast_test_fe_quad_derivatrives_h__
#define __mast_test_fe_quad_derivatrives_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Test {
namespace FEBasis {

namespace Edge2 {

template <typename ScalarType>
inline void fe_lagrange(const ScalarType xi,
                        Eigen::Matrix<ScalarType, 2, 1>& Nvec,
                        Eigen::Matrix<ScalarType, 2, 1>& dNvec) {
    
    Nvec     = Eigen::Matrix<ScalarType, 2, 1>::Zero();
    Nvec(0)  = (1.-xi)/2.;
    Nvec(1)  = (1.+xi)/2.;
    dNvec(0) = -1./2.;
    dNvec(1) = +1./2.;
}

} // namespace Edge2


namespace Quad4 {

template <typename ScalarType>
inline void tensor_product(const Eigen::Matrix<ScalarType, 2, 1>& Nxi,
                           const Eigen::Matrix<ScalarType, 2, 1>& Neta,
                           Eigen::Matrix<ScalarType, 4, 1>&       Nvec) {

    Nvec = Eigen::Matrix<ScalarType, 4, 1>::Zero();
    Nvec(0) = Nxi(0)*Neta(0); // node 1
    Nvec(1) = Nxi(1)*Neta(0); // node 2
    Nvec(2) = Nxi(1)*Neta(1); // node 3
    Nvec(3) = Nxi(0)*Neta(1); // node 4
}


template <typename ScalarType>
inline void fe_basis(const ScalarType xi,
                     const ScalarType eta,
                     Eigen::Matrix<ScalarType, 4, 1>& Nvec,
                     Eigen::Matrix<ScalarType, 4, 1>& dNvecdxi,
                     Eigen::Matrix<ScalarType, 4, 1>& dNvecdeta) {
    
    
    Eigen::Matrix<ScalarType, 2, 1>
    Nxi,
    Neta,
    dNdxi,
    dNdeta;
    
    MAST::Test::FEBasis::Edge2::fe_lagrange(xi,   Nxi,  dNdxi);
    MAST::Test::FEBasis::Edge2::fe_lagrange(eta, Neta, dNdeta);
    
    MAST::Test::FEBasis::Quad4::tensor_product(  Nxi,   Neta,      Nvec);
    MAST::Test::FEBasis::Quad4::tensor_product(dNdxi,   Neta,  dNvecdxi);
    MAST::Test::FEBasis::Quad4::tensor_product(  Nxi, dNdeta, dNvecdeta);
}



template <typename ScalarType>
inline void compute_fe_quad_derivatives(const ScalarType xi,
                                        const ScalarType eta,
                                        const Eigen::Matrix<ScalarType, 4, 1>& x_vec,
                                        const Eigen::Matrix<ScalarType, 4, 1>& y_vec,
                                        const Eigen::Matrix<ScalarType, 4, 1>& u_vec,
                                        const uint_t mode,
                                        ScalarType& u,
                                        ScalarType& du_dx,
                                        ScalarType& du_dy,
                                        Eigen::Matrix<ScalarType, 4, 1>& Nvec,
                                        Eigen::Matrix<ScalarType, 4, 1>& dNvec_dxi,
                                        Eigen::Matrix<ScalarType, 4, 1>& dNvec_deta,
                                        Eigen::Matrix<ScalarType, 4, 1>& dNvec_dx,
                                        Eigen::Matrix<ScalarType, 4, 1>& dNvec_dy,
                                        Eigen::Matrix<ScalarType, 2, 2>& Jac,
                                        Eigen::Matrix<ScalarType, 2, 2>& Jac_inv,
                                        ScalarType& J_det,
                                        Eigen::Matrix<ScalarType, 3, 1>& nvec,
                                        Eigen::Matrix<ScalarType, 3, 1>& tvec) {

    //
    //  mode = 0:    J_det is defined for the 2D domain
    //  mode = 1:    J_det is defined for the bottom edge
    //  mode = 2:    J_det is defined for the right edge
    //  mode = 3:    J_det is defined for the top edge
    //  mode = 4:    J_det is defined for the left edge
    //
    
    MAST::Test::FEBasis::Quad4::fe_basis(xi, eta, Nvec, dNvec_dxi, dNvec_deta);
    
    ScalarType
    dx_dxi    = dNvec_dxi.dot(x_vec),       // dx/dxi
    dx_deta   = dNvec_deta.dot(x_vec),      // dx/deta
    dy_dxi    = dNvec_dxi.dot(y_vec),       // dy/dxi
    dy_deta   = dNvec_deta.dot(y_vec);      // dy/deta

    // Jacobian
    Jac(0, 0) = dx_dxi;
    Jac(0, 1) = dx_deta;
    Jac(1, 0) = dy_dxi;
    Jac(1, 1) = dy_deta;
    
    // inverse of Jacobian
    Jac_inv   = Jac.inverse();

    dNvec_dx   =  Jac_inv(0, 0) * dNvec_dxi + Jac_inv(1, 0) * dNvec_deta;     // dN/dx
    dNvec_dy   =  Jac_inv(0, 1) * dNvec_dxi + Jac_inv(1, 1) * dNvec_deta;     // dN/dy

    u          =  Nvec.dot(u_vec);            // value of u
    du_dx      =  dNvec_dx.dot(u_vec);        // du/dx
    du_dy      =  dNvec_dy.dot(u_vec);        // du/dy
    
    Eigen::Matrix<ScalarType, 3, 1>
    khat, ds;
    khat << 0, 0, 1;

    switch (mode) {
        case 0: {
            
            J_det   = Jac.determinant();
        }
            break;
            
        case 1: {

            // for constant eta = -1
            // Hence, measure J_det using (dx/dxi  dy/dxi)
            J_det   = Jac.row(0).norm();
            ds      << dx_dxi, dy_dxi, 0;
        }
            break;
            
        case 2: {
            // for constant xi  =  1
            // Hence, measure J_det using (dx/deta  dy/deta)
            J_det   = Jac.row(1).norm();
            ds      << dx_deta, dy_deta, 0;
        }
            break;
            
        case 3: {
            // for constant eta =  1
            // Hence, measure J_det using (dx/dxi  dy/dxi)
            J_det   = Jac.row(0).norm();
            ds      << -dx_dxi, -dy_dxi, 0;
        }
            break;
            
        case 4: {
            // for constant xi  = -1
            // Hence, measure J_det using (dx/deta  dy/deta)
            J_det   = Jac.row(1).norm();
            ds      << -dx_deta, -dy_deta, 0;
        }
            break;
            
        default:
            Error(false, "Invalid mode number");
            
    }
    tvec    = ds/ds.norm();
    nvec    = ds.cross(khat);
    nvec    /= nvec.norm();
}
} // namespace Quad4
} // namespace FEBasis
} // namespace Test
} // namespace MAST

#endif // __mast_test_fe_quad_derivatrives_h__
