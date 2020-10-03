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

#ifndef __mast_conduction_kernel_h__
#define __mast_conduction_kernel_h__

// MAST includes
#include <mast/physics/conduction/gradient_operator.hpp>

namespace MAST {
namespace Physics {
namespace Conduction {
//namespace ConductionKernel {


template <typename FEVarType,
typename SectionPropertyType,
uint_t Dim,
typename ContextType,
bool IsotropicMaterial = SectionPropertyType::is_isotropic,
bool LinearMaterial    = SectionPropertyType::is_linear>
class ConductionKernel { };


/*!
 * This class implements the discrete evaluation of the conduction (Laplace operator) kernel defined as
 * \f[ \int_{\Omega_e} \frac{\partial \phi}{\partial x_i} k \frac{\partial T}{\partial x_i}, \f]
 * where, \f$ \phi\f$ is the variation and \f$ T \f$ is the temperature. Note that currently this assumes
 * a linear isotropic coefficient of conductance.
 *
 * Template parameter:
 *    - \p FEVarType : Class that provides the interpolation and spatial derivative of solution at quadrature points.
 *    - \p SectionPropertyType : Class that provides material property
 *    - \p Dim : Spatial dimension of the element
 *    - \p ContextType : Class that provides the context object where member variable \p qp
 *    is set to the current quadrature point during the quadrature loop.
 */
template <typename FEVarType,
typename SectionPropertyType,
uint_t Dim,
typename ContextType>
class ConductionKernel<FEVarType,
SectionPropertyType,
Dim,
ContextType,
true,
true> {
    
public:
    
    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using fe_shape_deriv_t = typename FEVarType::fe_shape_deriv_t;
    
    ConductionKernel():
    _property    (nullptr),
    _fe_var_data (nullptr)
    { }
    
    virtual ~ConductionKernel() { }
    
    inline void
    set_section_property(const SectionPropertyType& p) {
        
        Assert0(!_property, "Property already initialized.");
        
        _property = &p;
    }
    
    inline void set_fe_var_data(const FEVarType& fe_data)
    {
        Assert0(!_fe_var_data, "FE data already initialized.");
        _fe_var_data = &fe_data;
    }
    
    inline uint_t n_dofs() const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        
        return _fe_var_data->get_fe_shape_data().n_basis();
    }
    
    /*!
     * Computes the residual of variational term
     *  \f[ \int_{\Omega_e} \frac{\partial \phi}{\partial x_i} k \frac{\partial T}{\partial x_i}. \f] and returns it
     *  in \p res, and its Jacobian in \p jac if it is not a \p nullptr. Note that this method does
     *  not zero these two quantities and adds the contribution from this element to the
     *  vector and matrix provided in the function arguments.
     */
    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        typename Eigen::Matrix<scalar_t, Dim, 1>
        grad;
        vector_t
        vec     = vector_t::Zero(fe.n_basis());
        
        typename SectionPropertyType::value_t
        mat;
        
        matrix_t
        mat2 = matrix_t::Zero(fe.n_basis(), fe.n_basis());
        
        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bxmat;
        Bxmat.reinit(Dim, 1, fe.n_basis());
        
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp = i;
            
            _property->value(c, mat);
            MAST::Physics::Conduction::GradientOperator::gradient_operator
            <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, grad, Bxmat);
            Bxmat.vector_mult_transpose(vec, grad);
            res += fe.detJxW(i) * mat * vec;
            
            if (jac) {
                
                Bxmat.right_multiply_transpose(mat2, Bxmat);
                (*jac) += fe.detJxW(i) * mat * mat2;
            }
        }
    }
    
    /*!
     * Computes the derivative of residual of variational term with respect to parameter \f$ \alpha \f$
     *  \f[ \int_{\Omega_e} \frac{\partial \phi}{\partial x_i} \frac{\partial k}{\partial \alpha} \frac{\partial T}{\partial x_i}. \f] and returns it
     *  in \p res, and the derivative of Jacobian in \p jac if it is not a \p nullptr.
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
        
        typename Eigen::Matrix<scalar_t, Dim, 1>
        grad;
        vector_t
        vec     = vector_t::Zero(fe.n_basis());
        
        typename SectionPropertyType::value_t
        mat;
        matrix_t
        mat2 = matrix_t::Zero(fe.n_basis(), fe.n_basis());
        
        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bxmat;
        Bxmat.reinit(Dim, 1, fe.n_basis());
        
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp = i;
            
            _property->derivative(c, f, mat);
            MAST::Physics::Conduction::GradientOperator::gradient_operator
            <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, grad, Bxmat);
            Bxmat.vector_mult_transpose(vec, grad);
            res += fe.detJxW(i) * mat * vec;
            
            if (jac) {
                
                Bxmat.right_multiply_transpose(mat2, Bxmat);
                (*jac) += fe.detJxW(i) * mat * mat2;
            }
        }
    }
    
    
private:
    
    
    const SectionPropertyType       *_property;
    const FEVarType                 *_fe_var_data;
};

//}  // namespace ConductionKernel
}  // namespace Conduction
}  // namespace Physics
}  // namespace MAST

#endif // __mast_conduction_kernel_h__
