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

#ifndef __mast_elasticity_plate_linear_acceleration_kernel_h__
#define __mast_elasticity_plate_linear_acceleration_kernel_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/fem_operator_matrix.hpp>


namespace MAST {
namespace Physics {
namespace Elasticity {
namespace Plate {


template <typename NodalScalarType,
          typename VarScalarType,
          typename FEVarType>
inline void
displacement(const FEVarType&                                    fe_var,
             const uint_t                                        qp,
             typename Eigen::Matrix<VarScalarType, 3, 1>&        u,
             MAST::Numerics::FEMOperatorMatrix<NodalScalarType> &Bmat) {
    
    u.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(Bmat.m() == 3,
            Bmat.m(),
            "Incompatible operator size");
    Assert2(Bmat.n() == 3*fe.n_basis(),
            Bmat.n(), 3*fe.n_basis(),
            "Incompatible Operator size.");
    
    
    // set the strain displacement relation
    for (uint_t i=0; i<3; i++) {
        Bmat.set_shape_function(i, i, fe.phi(qp));
        u(i) = fe_var.u(qp, i);
    }
}

/*!
 * This class implements the discrete evaluation of the acceleration kernel defined as
 * \f[ - \int_{\Omega_e}   \left(\delta w \rho h \frac{\partial^t u}{\partial t^2} + \delta \theta_x \frac{\rho h^3}{12} \frac{\partial^t \theta_x}{\partial t^2} + \delta \theta_y \frac{\rho h^3}{12} \frac{\partial^t \theta_y}{\partial t^2}  \right) ~d\Omega, \f]
 * where, \f$ \phi\f$ is the variation, \f$ \rho \f$ is the material density, and
 * \f$ h \f$ is the section thickness.
 *
 * Template parameter:
 *    - \p FEVarType : Class that provides the interpolation and spatial derivative of solution at quadrature points.
 *    - \p SectionPropertyType : Class that provides the factors for multiplying translation and rotational terms at
 *    quadrature points.
 *    - \p ContextType : Class that provides the context object where member variable \p qp
 *    is set to the current quadrature point during the quadrature loop.
 */
template <typename FEVarType,
          typename SectionPropertyType,
          typename ContextType>
class LinearAcceleration {

public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    LinearAcceleration():
    _property      (nullptr),
    _fe_var_data   (nullptr)
    { }
    
    virtual ~LinearAcceleration() { }
    
    inline void
    set_section_property(const SectionPropertyType& p) {
        
        Assert0(!_property, "Property already initialized.");
        
        _property = &p;
    }

    inline void set_fe_var_data(const FEVarType& fe) { _fe_var_data = &fe;}

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return 3*_fe_var_data->get_fe_shape_data().n_basis();
    }

    /*!
     * Computes the residual of variational term
     * \f[ - \int_{\Omega_e}   \left(\delta w \rho h \frac{\partial^t u}{\partial t^2} + \delta \theta_x \frac{\rho h^3}{12} \frac{\partial^t \theta_x}{\partial t^2} + \delta \theta_y \frac{\rho h^3}{12} \frac{\partial^t \theta_y}{\partial t^2}  \right) ~d\Omega, \f]
     *  and returns it in \p res. The Jacobian is returned in \p jac if it is a non-null pointer. Note that this method
     *  does not zero these two quantities and adds the contribution from this element to the
     *  vector and matrix provided in the function arguments.
     */
    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");

        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        const uint_t
        n_basis = fe.n_basis();

        typename Eigen::Matrix<scalar_t, 3, 1>
        u;

        vector_t
        vec     = vector_t::Zero(3*n_basis);

        scalar_t
        w_factor     = 0.,
        theta_factor = 0.;

        matrix_t
        mat = matrix_t::Zero(3*n_basis, 3*n_basis);

        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bmat;
        Bmat.reinit(3, 3, n_basis);

        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            
            _property->translation_inertia(c, w_factor);
            _property->rotation_inertia(c, theta_factor);
            
            MAST::Physics::Elasticity::Plate::displacement
            <scalar_t, scalar_t, FEVarType>(*_fe_var_data, i, u, Bmat);

            u(0) *= w_factor;
            u(1) *= theta_factor;
            u(2) *= theta_factor;
            
            Bmat.vector_mult_transpose(vec, u);
            res += fe.detJxW(i) * vec;
            
            if (jac) {
                
                Bmat.right_multiply_transpose(mat, Bmat);

                mat.topLeftCorner(n_basis, n_basis)           *= w_factor;
                mat.block(n_basis, n_basis, n_basis, n_basis) *= theta_factor;
                mat.bottomRightCorner(n_basis, n_basis)       *= theta_factor;

                (*jac) += fe.detJxW(i) * mat;
            }
        }
    }
    
    
    /*!
     * Computes the derivative of residual of variational term with respect to parameter \f$ \alpha \f$
     * \f[ - \int_{\Omega_e}   \left(\delta w  \frac{\partial \rho h}{\partial \alpha} \frac{\partial^t u}{\partial t^2} + \delta \theta_x \frac{\partial (\rho h^3/12)}{\partial \alpha} \frac{\partial^t \theta_x}{\partial t^2} + \delta \theta_y \frac{\partial (\rho h^3/12)}{\partial \alpha} \frac{\partial^t \theta_y}{\partial t^2}  \right) ~d\Omega, \f]
     *  and returns it in \p res. The Jacobian and its derivative for this term is zero.
     *  Note that this method does not zero these two quantities and adds the contribution from this
     *  element to the vector and matrix provided in the function arguments.
     */
    template <typename ScalarFieldType>
    inline void derivative(ContextType& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");

        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        const uint_t
        n_basis = fe.n_basis();
        
        typename Eigen::Matrix<scalar_t, 3, 1>
        u;
        
        vector_t
        vec     = vector_t::Zero(3*n_basis);

        scalar_t
        dw_factor     = 0.,
        dtheta_factor = 0.;

        matrix_t
        mat = matrix_t::Zero(3*n_basis, 3*n_basis);

        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bmat;
        Bmat.reinit(3, 3, n_basis);

        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            
            _property->translation_inertia_derivative(c, f, dw_factor);
            _property->rotation_inertia_derivative(c, f, dtheta_factor);
            
            MAST::Physics::Elasticity::Plate::displacement
            <scalar_t, scalar_t, FEVarType>(*_fe_var_data, i, u, Bmat);
            
            u(0) *= dw_factor;
            u(1) *= dtheta_factor;
            u(2) *= dtheta_factor;

            Bmat.vector_mult_transpose(vec, u);
            res += fe.detJxW(i) * vec;
            
            if (jac) {
                
                Bmat.right_multiply_transpose(mat, Bmat);

                mat.topLeftCorner(n_basis, n_basis)           *= dw_factor;
                mat.block(n_basis, n_basis, n_basis, n_basis) *= dtheta_factor;
                mat.bottomRightCorner(n_basis, n_basis)       *= dtheta_factor;

                (*jac) += fe.detJxW(i) * mat;
            }
        }
    }
    
private:

    const SectionPropertyType  *_property;
    const FEVarType            *_fe_var_data;
};


} // namespace Plate
} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_elasticity_plate_linear_acceleration_kernel_h__
