
#ifndef __mast_green_lagrange_strain_operator_h__
#define __mast_green_lagrange_strain_operator_h__

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
void green_lagrange_strain_operator(const FEVarType& fe_var,
                                    const uint_t qp,
                                    typename Eigen::Matrix<VarScalarType, 3, 1>::type& epsilon,
                                    typename Eigen::Matrix<VarScalarType, 3, 1>::type& epsilon,
                                    EigenVector<ScalarType>& epsilon,
                                    EigenMatrix<ScalarType>& mat_x,
                                    EigenMatrix<ScalarType>& mat_y,
                                    MAST::FEMOperatorMatrixBase<ScalarType>& Bmat_lin,
                                    MAST::FEMOperatorMatrixBase<ScalarType>& Bmat_nl_x,
                                    MAST::FEMOperatorMatrixBase<ScalarType>& Bmat_nl_y,
                                    MAST::FEMOperatorMatrixBase<ScalarType>& Bmat_nl_u,
                                    MAST::FEMOperatorMatrixBase<ScalarType>& Bmat_nl_v) {
    
    
    epsilon.setZero();
    mat_x.setZero();
    mat_y.setZero();

    const typename FEVarType::fe_shape_data_type
    &fe = fe_var.get_fe_shape_object();

    // make sure all matrices are the right size
    Assert1(epsilon.size() == 3,
            epsilon.size(),
            "Strain vector for 2D continuum strain should be 3.");
    Assert1(mat_x.rows() == 3,
            mat_x.rows(),
            "Incompatible matrix size.");
    Assert1(mat_x.cols() == 2,
            mat_x.cols(),
            "Incompatible matrix size.");
    Assert1(mat_y.rows() == 3,
            mat_y.rows(),
            "Incompatible matrix size.");
    Assert1(mat_y.cols() == 2,
            mat_y.cols(),
            "Incompatible matrix size.");

    Assert1(Bmat_lin.m() == 3,
            Bmat_lin.m(),
            "Strain vector for 2D continuum strain should be 3");
    Assert2(Bmat_lin.n() == 2*fe.n_basis(),
            Bmat_lin.n(), 2*fe.n_basis(),
            "Incompatible Operator size.");
    Assert1(Bmat_nl_x.m() == 2,
            Bmat_nl_x.m(),
            "Incompatible matrix size.");
    Assert2(Bmat_nl_x.n() == 2*fe.n_basis(),
            Bmat_nl_x.n(), 2*fe.n_basis(),
            "Incompatible matrix size.");
    Assert1(Bmat_nl_y.m() == 2,
            Bmat_nl_y.m(),
            "Incompatible matrix size.");
    Assert2(Bmat_nl_y.n() == 2*fe.n_basis(),
            Bmat_nl_y.n(), 2*fe.n_basis(),
            "Incompatible matrix size.");
    Assert1(F.cols() == 2,
            F.cols(),
            "Incompatible matrix size.");
    Assert1(F.rows() == 2,
            F.rows(),
            "Incompatible matrix size.");
    Assert1(E.cols() == 2,
            E.cols(),
            "Incompatible matrix size.");
    Assert1(E.rows() == 2,
            E.rows(),
            "Incompatible matrix size.");

    
    // now set the shape function values
    Bmat_lin.set_shape_function(0, 0, fe.dphi_dx(qp, 0)); //  epsilon_xx = du/dx
    Bmat_lin.set_shape_function(2, 1, fe.dphi_dx(qp, 0)); //  gamma_xy = dv/dx + ...
    
    // nonlinear strain operator in x
    Bmat_nl_x.set_shape_function(0, 0, fe.dphi_dx(qp, 0)); // du/dx
    Bmat_nl_x.set_shape_function(1, 1, fe.dphi_dx(qp, 0)); // dv/dx
    
    // nonlinear strain operator in u
    Bmat_nl_u.set_shape_function(0, 0, fe.dphi_dx(qp, 0)); // du/dx
    Bmat_nl_v.set_shape_function(0, 1, fe.dphi_dx(qp, 0)); // dv/dx
    
    // dN/dy
    Bmat_lin.set_shape_function(1, 1, fe.dphi_dx(qp, 1)); //  epsilon_yy = dv/dy
    Bmat_lin.set_shape_function(2, 0, fe.dphi_dx(qp, 1)); //  gamma_xy = du/dy + ...
    
    // nonlinear strain operator in y
    Bmat_nl_y.set_shape_function(0, 0, fe.dphi_dx(qp, 1)); // du/dy
    Bmat_nl_y.set_shape_function(1, 1, fe.dphi_dx(qp, 1)); // dv/dy
    
    // nonlinear strain operator in v
    Bmat_nl_u.set_shape_function(1, 0, fe.dphi_dx(qp, 1)); // du/dy
    Bmat_nl_v.set_shape_function(1, 1, fe.dphi_dx(qp, 1)); // dv/dy
    
    // prepare the deformation gradient matrix
    F.row(0) = fe_var.du_dx(qp, 0);
    F.row(1) = fe_var.du_dx(qp, 1);
    
    // this calculates the Green-Lagrange strain in the reference config
    E = 0.5*(F + F.transpose() + F.transpose() * F);
    
    // now, add this to the strain vector
    epsilon(0) = E(0,0);
    epsilon(1) = E(1,1);
    epsilon(2) = E(0,1) + E(1,0);
    
    // now initialize the matrices with strain components
    // that multiply the Bmat_nl terms
    mat_x(0, 0) =     fe_var.du_dx(qp, 0, 0);
    mat_x(0, 1) =     fe_var.du_dx(qp, 1, 0);
    mat_x(2, 0) =     fe_var.du_dx(qp, 0, 1);
    mat_x(2, 1) =     fe_var.du_dx(qp, 1, 1);
    
    mat_y(1, 0) =     fe_var.du_dx(qp, 0, 1);
    mat_y(1, 1) =     fe_var.du_dx(qp, 1, 1);
    mat_y(2, 0) =     fe_var.du_dx(qp, 0, 0);
    mat_y(2, 1) =     fe_var.du_dx(qp, 1, 0);
}

}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST


#endif // __mast_green_lagrange_strain_operator_h__
