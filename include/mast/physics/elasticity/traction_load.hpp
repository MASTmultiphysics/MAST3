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

#ifndef __mast_traction_load_h__
#define __mast_traction_load_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {

/*!
 * This class implements the discrete evaluation of the surface traction kernel defined as
 * \f[ - \int_{\Gamma_e}  a \phi t \cdot \hat{n} ~d\Gamma, \f]
 * where, \f$ \phi\f$ is the variation, \f$ t \f$ is the surface traction, \f$ a \f$ is the section thickness for 2D
 * elements or section area for 1D elements, and \f$ \hat{n} \f$ is the surface normal.
 *
 * Template parameter:
 *    - \p FEVarType : Class that provides the interpolation and spatial derivative of solution at quadrature points.
 *    - \p TractionFieldType : Class that provides the flux value at quadrature point
 *    - \p SectionAreaType : Class that provides the section thickness for 2D elements and section area for 1D
 *    elements at quadrature points.
 *    - \p Dim : Spatial dimension of the element.
 *    - \p ContextType : Class that provides the context object where member variable \p qp
 *    is set to the current quadrature point during the quadrature loop.
 */
template <typename FEVarType,
          typename TractionFieldType,
          typename SectionAreaType,
          uint_t Dim,
          typename ContextType>
class SurfaceTractionLoad {

public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    SurfaceTractionLoad():
    _section       (nullptr),
    _traction      (nullptr),
    _fe_var_data   (nullptr)
    { }
    
    virtual ~SurfaceTractionLoad() { }
    
    inline void set_section_area(const SectionAreaType& s) { _section = &s;}
    
    inline void set_traction(const TractionFieldType& t) { _traction = &t;}
    
    inline void set_fe_var_data(const FEVarType& fe) { _fe_var_data = &fe;}

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return Dim*_fe_var_data->get_fe_shape_data().n_basis();
    }

    /*!
     * Computes the residual of variational term
     * \f[ - \int_{\Gamma_e} \phi a t \cdot \hat{n} ~d\Gamma, \f] and returns it
     *  in \p res. The Jacobian for this term is zero. Note that this method does
     *  not zero these two quantities and adds the contribution from this element to the
     *  vector and matrix provided in the function arguments.
     */
    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_section, "Section property not initialized");
        Assert0(_traction, "Traction not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        typename TractionFieldType::value_t
        trac;
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            
            _traction->value(c, trac);
            
            trac *= _section->value(c);
            
            for (uint_t j=0; j<Dim; j++) {
                
                // j-th component of normal vector at ith quadrature point
                scalar_t nj = fe.normal(i, j);
                
                if (nj != 0.) {
                    for (uint_t k=0; k<fe.n_basis(); k++)
                        res(j*fe.n_basis() + k) -= fe.detJxW(i) * fe.phi(i, k) * trac(j) * nj;
                }
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
        Assert0(_traction, "Traction not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        typename TractionFieldType::value_t
        trac,
        dtrac;

        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            
            _traction->value(c, trac);
            _traction->derivative(f, c, dtrac);
            
            dtrac *= _traction->value(c);
            trac  *= _section->derivative(c, f);
            trac += dtrac;
            
            for (uint_t j=0; j<Dim; j++) {
                
                // j-th component of normal vector at ith quadrature point
                scalar_t nj = fe.normal(i, j);
                
                if (nj != 0.) {
                    for (uint_t k=0; k<fe.n_basis(); k++)
                        res(j*fe.n_basis() + k) -= fe.detJxW(i) * fe.phi(i, k) * trac(j) * nj;
                }
            }
        }
    }
    
private:

    const SectionAreaType      *_section;
    const TractionFieldType    *_traction;
    const FEVarType            *_fe_var_data;
};


} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_traction_load_h__
