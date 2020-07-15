
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
            "Strain vector for 2D continuum strain should be 3");
    Assert1(Bmat.m() == 3,
            Bmat.m(),
            "Strain vector for 2D continuum strain should be 3");
    Assert2(Bmat.n() == 2*fe.n_basis(),
            Bmat.n(), 2*fe.n_basis(),
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
                        const VarScalarType                                 z,
                        typename Eigen::Matrix<VarScalarType, 3, 1>        &epsilon,
                        MAST::Numerics::FEMOperatorMatrix<NodalScalarType>& Bmat) {
    
    // initialize the strain operator
    for ( unsigned int i_nd=0; i_nd<phi.size(); i_nd++ )
        phi_vec(i_nd) = dphi[i_nd][qp](0);  // dphi/dx
    
    Bmat.set_shape_function(0, 0, fe.dphi_dx(qp, 0)); // gamma-xz:  dw/dx
    
    for ( unsigned int i_nd=0; i_nd<phi.size(); i_nd++ )
        phi_vec(i_nd) = dphi[i_nd][qp](1);  // dphi/dy
    
    Bmat.set_shape_function(1, 0, fe.dphi_dx(qp, 1)); // gamma-yz : dw/dy
    
    for ( unsigned int i_nd=0; i_nd<phi.size(); i_nd++ )
        phi_vec(i_nd) = phi[i_nd][qp];  // phi
    
    Bmat.set_shape_function(0, 4, phi_vec); // gamma-xz:  thetay
    phi_vec  *= -1.0;
    Bmat.set_shape_function(1, 3, phi_vec); // gamma-yz : thetax
    
    
    // now add the transverse shear component
    Bmat.vector_mult(vec_2, _structural_elem.local_solution());
    vec_2 = material * vec_2;
    Bmat.vector_mult_transpose(vec_n2, vec_2);
    local_f += JxW[qp] * vec_n2;
    
    if (request_jacobian) {
        
        // now add the transverse shear component
        Bmat.left_multiply(mat_2n2, material);
        Bmat.right_multiply_transpose(mat_n2n2, mat_2n2);
        local_jac += JxW[qp] * mat_n2n2;
    }
}


}  // namespace MindlinPlate
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST


#endif // __mast_mindlin_strain_operator_h__
