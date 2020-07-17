
#ifndef __mast_mindlin_strain_operator_h__
#define __mast_mindlin_strain_operator_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/fem_operator_matrix.hpp>


namespace MAST {
namespace Physics {
namespace Elasticity {
namespace MindlinPlate {


template <typename NodalScalarType, typename VarScalarType, typename FEVarType>
inline void
inplane_strain(const FEVarType&                                    fe_var,
               const uint_t                                        qp,
               const VarScalarType                                 z,
               typename Eigen::Matrix<VarScalarType, 3, 1>        &epsilon,
               MAST::Numerics::FEMOperatorMatrix<NodalScalarType>& Bmat) {
    
    epsilon.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(epsilon.size() == 3,
            epsilon.size(),
            "Strain vector dimension for inplane strain of Mindlin plate should be 3");
    Assert1(Bmat.m() == 3,
            Bmat.m(),
            "Strain vector dimension for inplane strain of Mindlin plate should be 3");
    Assert2(Bmat.n() == 3*fe.n_basis(),
            Bmat.n(), 3*fe.n_basis(),
            "Incompatible Operator size.");
    
    
    // linear strain operator
    Bmat.set_shape_function(0, 2,  z, fe.dphi_dx(qp, 0)); //  epsilon_xx =  z * dthetay/dx
    Bmat.set_shape_function(2, 1, -z, fe.dphi_dx(qp, 0)); //  gamma_xy   = -z * dthetax/dx + ...
        
    // linear strain operator
    Bmat.set_shape_function(1, 1, -z, fe.dphi_dx(qp, 1)); //  epsilon_yy = -z * dthetax/dy
    Bmat.set_shape_function(2, 2,  z, fe.dphi_dx(qp, 1)); //  gamma_xy   =  z * dthetay/dy
    
    epsilon(0) =  z * fe_var.du_dx(qp, 2, 0);  //  z * dthetay/dx
    epsilon(1) = -z * fe_var.du_dx(qp, 1, 1);  // -z * dthetax/dy
    epsilon(2) =  z * (fe_var.du_dx(qp, 2, 1) - fe_var.du_dx(qp, 1, 0));  // z * (dty/dy - dtx/dx)
}



template <typename NodalScalarType, typename VarScalarType, typename FEVarType>
inline void
transverse_shear_strain(const FEVarType&                                    fe_var,
                        const uint_t                                        qp,
                        typename Eigen::Matrix<VarScalarType, 2, 1>        &epsilon,
                        MAST::Numerics::FEMOperatorMatrix<NodalScalarType>& Bmat) {
    
    epsilon.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(epsilon.size() == 2,
            epsilon.size(),
            "Strain vector dimension for transverse shear strain of Mindlin plate should be 2");
    Assert1(Bmat.m() == 2,
            Bmat.m(),
            "Strain vector dimension for transverse shear strain of Mindlin plate should be 2");
    Assert2(Bmat.n() == 3*fe.n_basis(),
            Bmat.n(), 3*fe.n_basis(),
            "Incompatible Operator size.");

    Bmat.set_shape_function(0, 2,  1., fe.phi(qp)); // gamma-xz:  thetay
    Bmat.set_shape_function(1, 1, -1., fe.phi(qp)); // gamma-yz : thetax

    epsilon(0) = fe_var.du_dx(qp, 0, 0) + fe_var.u(qp, 2);  // gamma-xz = dw/dx + ty
    epsilon(1) = fe_var.du_dx(qp, 0, 1) - fe_var.u(qp, 1);  // gamma-xz = dw/dy - tx
}

}  // namespace MindlinPlate
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST


#endif // __mast_mindlin_strain_operator_h__
