
#ifndef __mast_linear_elastic_strain_operator_h__
#define __mast_linear_elastic_strain_operator_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/fem_operator_matrix.hpp>


namespace MAST {
namespace Physics {
namespace Elasticity {


template <typename NodalScalarType, typename VarScalarType, typename FEVarType, uint_t Dim>
inline
typename std::enable_if<Dim == 2, void>::type
linear_continuum_strain(const FEVarType&                                    fe_var,
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

}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST


#endif // __mast_linear_elastic_strain_operator_h__
