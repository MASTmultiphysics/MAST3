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

#ifndef __mast_conduction_source_load_h__
#define __mast_conduction_source_load_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Conduction {

template <typename ScalarType,
          typename SectionAreaType,
          typename SourceLoadFieldType,
          typename ContextType,
          uint_t Dim>
typename std::enable_if<Dim<3, ScalarType>::type
source_load_multiplier(const SourceLoadFieldType    *f,
                       const SectionAreaType  *s,
                       ContextType            &c) {
    Assert0(f, "Invalid pointer");
    Assert0(s, "Invalid pointer");
    return f->value(c) * s->value(c);
}

template <typename ScalarType,
          typename SectionAreaType,
          typename SourceLoadFieldType,
          typename ContextType,
          uint_t Dim>
typename std::enable_if<Dim==3, ScalarType>::type
source_load_multiplier(const SourceLoadFieldType    *f,
                       const SectionAreaType  *s,
                       ContextType            &c) {
    Assert0(f, "Invalid pointer");
    Assert0(!s, "Pointer must be nullptr");
    return f->value(c);
}

template <typename ScalarType,
          typename SectionAreaType,
          typename SourceLoadFieldType,
          typename ContextType,
          typename ScalarFieldType,
          uint_t Dim>
typename std::enable_if<Dim<3, ScalarType>::type
source_load_derivative_multiplier(const SourceLoadFieldType    *f,
                                  const SectionAreaType  *s,
                                  ContextType            &c,
                                  const ScalarFieldType  &p) {
    
    Assert0(f, "Invalid pointer");
    Assert0(s, "Invalid pointer");
    return (f->value(c) * s->derivative(c, p) +
            s->value(c) * f->derivative(c, p));
}

template <typename ScalarType,
          typename SectionAreaType,
          typename SourceLoadFieldType,
          typename ContextType,
          typename ScalarFieldType,
          uint_t Dim>
typename std::enable_if<Dim==3, ScalarType>::type
source_load_derivative_multiplier(const SourceLoadFieldType    *f,
                                  const SectionAreaType  *s,
                                  ContextType            &c,
                                  const ScalarFieldType  &p) {
    
    Assert0(f, "Invalid pointer");
    Assert0(!s, "Pointer must be nullptr");
    return f->derivative(c);
}


/*!
 * This class implements the discrete evaluation of the conduction (Laplace operator) kernel defined as
 * \f[ - \int_{\Omega_e} \phi q_v~d\Omega, \f]
 * where, \f$ \phi\f$ is the variation and \f$ q_v \f$ is the source load value.
 *
 * Template parameter:
 *    - \p FEVarType : Class that provides the interpolation and spatial derivative of solution at quadrature points.
 *    - \p SourceLoadFieldType : Class that provides the load value at quadrature point
 *    - \p SectionAreaType : Class that provides the section thickness for 2D elements and section area for 1D
 *    elements at quadrature points. 3D elements do not require a section area objcet, and this template parameter
 *    - can be \p void.
 *    - \p Dim : Spatial dimension of the element.
 *    - \p ContextType : Class that provides the context object where member variable \p qp
 *    is set to the current quadrature point during the quadrature loop.
 */
template <typename FEVarType,
          typename SourceLoadFieldType,
          typename SectionAreaType,
          uint_t Dim,
          typename ContextType>
class SourceHeatLoad {

public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    SourceHeatLoad():
    _section       (nullptr),
    _load          (nullptr),
    _fe_var_data   (nullptr)
    { }
    
    virtual ~SourceHeatLoad() { }
    
    /*!
     * Provides the section area through the object \p s. Not needed for 3D elements.
     */
    inline void set_section_area(const SectionAreaType& s) {
        
        Assert1(Dim < 3, Dim, "SectionAreaType only used for 1D and 2D elements");
        
        _section = &s;
    }
    
    inline void set_source(const SourceLoadFieldType& s) { _load = &s;}
    
    inline void set_fe_var_data(const FEVarType& fe) { _fe_var_data = &fe;}

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return _fe_var_data->get_fe_shape_data().n_basis();
    }

    /*!
     * Computes the residual of variational term
     * \f[ - \int_{\Omega_e} \phi q_v~d\Omega, \f] and returns it
     *  in \p res. The Jacobian for this term is zero. Note that this method does
     *  not zero these two quantities and adds the contribution from this element to the
     *  vector and matrix provided in the function arguments.
     */
    inline void
    compute(ContextType& c,
            vector_t& res,
            matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(Dim==3 || _section, "Section property not initialized");
        Assert0(_load, "Source load not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            scalar_t p = source_load_multiplier<scalar_t,
                                                SectionAreaType,
                                                SourceLoadFieldType,
                                                ContextType,
                                                Dim>(_load, _section, c);
            
            for (uint_t k=0; k<fe.n_basis(); k++)
            res(k) -= fe.detJxW(i) * fe.phi(i, k) * p;
        }
    }

    
    /*!
     * Computes the derivative of residual of variational term with respect to parameter \f$ \alpha \f$
     *  \f[ - \int_{\Omega_e} \phi \frac{\partial q_v}{\partial \alpha}~d\Omega, \f] and returns it
     *  in \p res. The Jacobian and its derivative for this term is zero.
     *  Note that this method does not zero these two quantities and adds the contribution from this
     *  element to the vector and matrix provided in the function arguments.
     */
    template <typename ScalarFieldType>
    inline void derivative(ContextType& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(Dim==3 || _section, "Section property not initialized");
        Assert0(_load, "Source load not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            scalar_t p =
            source_load_derivative_multiplier<scalar_t,
                                              SectionAreaType,
                                              SourceLoadFieldType,
                                              ContextType,
                                              ScalarFieldType,
                                              Dim>(_load, _section, c, f);
            
            for (uint_t k=0; k<fe.n_basis(); k++)
            res(k) -= fe.detJxW(i) * fe.phi(i, k) * p;
        }
    }
    
private:

    const SectionAreaType      *_section;
    const SourceLoadFieldType  *_load;
    const FEVarType            *_fe_var_data;
};

} // namespace Conduction
} // namespace Physics
} // namespace MAST


#endif // __mast_conduction_source_load_h__
