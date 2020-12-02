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

#ifndef __mast_elasticity_continuum_linear_acceleration_kernel_h__
#define __mast_elasticity_continuum_linear_acceleration_kernel_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {
namespace LinearContinuum {


template <typename NodalScalarType,
          typename VarScalarType,
          typename FEVarType,
          uint_t Dim>
inline void
displacement(const FEVarType&                                    fe_var,
             const uint_t                                        qp,
             typename Eigen::Matrix<VarScalarType, Dim, 1>&      u,
             MAST::Numerics::FEMOperatorMatrix<NodalScalarType>& Bmat) {
    
    u.setZero();
    
    const typename FEVarType::fe_shape_deriv_t
    &fe = fe_var.get_fe_shape_data();
    
    // make sure all matrices are the right size
    Assert1(Bmat.m() == Dim,
            Bmat.m(),
            "Incompatible operator size");
    Assert2(Bmat.n() == Dim*fe.n_basis(),
            Bmat.n(), Dim*fe.n_basis(),
            "Incompatible Operator size.");
    
    
    // set the strain displacement relation
    for (uint_t i=0; i<Dim; i++) {
        Bmat.set_shape_function(i, i, fe.phi(qp));
        u(i) = fe_var.u(qp, i);
    }
}

/*!
 * This class implements the discrete evaluation of the acceleration kernel defined as
 * \f[ - \int_{\Omega_e}  \phi \rho a \frac{\partial^t u}{\partial t^2} ~d\Gamma, \f]
 * where, \f$ \phi\f$ is the variation, \f$ \rho \f$ is the material density, and
 * \f$ a \f$ is the section thickness for 2D elements or section area.
 *
 * Template parameter:
 *    - \p FEVarType : Class that provides the interpolation and spatial derivative of solution at quadrature points.
 *    - \p DensityFieldType : Class that provides the density value at quadrature point
 *    - \p SectionAreaType : Class that provides the section thickness for 1D elements and section area for 2D
 *    elements at quadrature points.
 *    - \p Dim : Spatial dimension of the element.
 *    - \p ContextType : Class that provides the context object where member variable \p qp
 *    is set to the current quadrature point during the quadrature loop.
 */
template <typename FEVarType,
          typename DensityFieldType,
          typename SectionAreaType,
          uint_t Dim,
          typename ContextType>
class LinearAcceleration {

public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    LinearAcceleration():
    _section       (nullptr),
    _traction      (nullptr),
    _fe_var_data   (nullptr)
    { }
    
    virtual ~LinearAcceleration() { }
    
    inline void set_section_area(const SectionAreaType& s) { _section = &s;}
    
    inline void set_density(const DensityFieldType& rho) { _density = &rho;}
    
    inline void set_fe_var_data(const FEVarType& fe) { _fe_var_data = &fe;}

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return Dim*_fe_var_data->get_fe_shape_data().n_basis();
    }

    /*!
     * Computes the residual of variational term
     * \f[ - \int_{\Omega_e}  \phi \rho a \frac{\partial^t u}{\partial t^2} ~d\Gamma, \f] and returns it
     *  in \p res. The Jacobian is returned in \p jac if it is a non-null pointer. Note that this method does
     *  not zero these two quantities and adds the contribution from this element to the
     *  vector and matrix provided in the function arguments.
     */
    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_section, "Section property not initialized");
        Assert0(_density, "Density not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        typename Eigen::Matrix<scalar_t, Dim, 1>
        u;
        vector_t
        vec     = vector_t::Zero(Dim*fe.n_basis());

        scalar_t
        rho = 0.,
        sec = 0.;

        matrix_t
        mat = matrix_t::Zero(Dim*fe.n_basis(), Dim*fe.n_basis());

        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bmat;
        Bmat.reinit(Dim, Dim, fe.n_basis());

        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            
            _density->value(c, rho);
            _sec->value(c, sec)
            
            MAST::Physics::Elasticity::LinearContinuum::displacement
            <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, u, Bmat);
            Bmat.vector_mult_transpose(vec, u);
            res += fe.detJxW(i) * vec * rho * sec;
            
            if (jac) {
                
                Bmat.right_multiply_transpose(mat, Bmat);
                (*jac) += fe.detJxW(i) * mat * rho * sec;
            }
        }
    }
    
    
    /*!
     * Computes the derivative of residual of variational term with respect to parameter \f$ \alpha \f$
     *  \f[ - \int_{\Gamma_e} \phi \left( \frac{\partial a}{\partial \alpha}  t +
     *                          \frac{\partial t}{\partial \alpha} a \right) \cdot \hat{n} ~d\Gamma, \f]
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
        Assert0(_section, "Section property not initialized");
        Assert0(_density, "Density not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        typename Eigen::Matrix<scalar_t, Dim, 1>
        u;
        vector_t
        vec     = vector_t::Zero(Dim*fe.n_basis());

        scalar_t
        rho  = 0.,
        drho = 0.,
        sec  = 0.;
        dsec = 0.;

        matrix_t
        mat = matrix_t::Zero(Dim*fe.n_basis(), Dim*fe.n_basis());

        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bmat;
        Bmat.reinit(Dim, Dim, fe.n_basis());

        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            
            _density->value(c, rho);
            _density->derivative(c, f, rho);
            _sec->value(c, sec)
            _sec->derivative(c, f, sec)

            MAST::Physics::Elasticity::LinearContinuum::displacement
            <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, u, Bmat);
            Bmat.vector_mult_transpose(vec, u);
            res += fe.detJxW(i) * vec * (drho * sec + rho * dsec);
            
            if (jac) {
                
                Bmat.right_multiply_transpose(mat, Bmat);
                (*jac) += fe.detJxW(i) * mat * (drho * sec + rho * dsec);
            }
        }
    }
    
private:

    const SectionAreaType      *_section;
    const DensityFieldType     *_density;
    const FEVarType            *_fe_var_data;
};


} // namespace LinearContinuum
} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_elasticity_continuum_linear_acceleration_kernel_h__
