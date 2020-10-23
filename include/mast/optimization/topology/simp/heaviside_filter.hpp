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

#ifndef __mast_simp_heaviside_filter_h__
#define __mast_simp_heaviside_filter_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {

/*!
 * This class implements the Heaviside filter defined as
 * \f[ \tilde{v) =  \frac{\tanh(\beta \eta) + \tanh(\beta(v-\eta)) }{\tanh (\beta \eta) + \tanh (\beta(1-\eta)) } \f]
 */
template <typename ScalarType, typename FieldType>
class HeavisideFilter {
    
public:
    
    HeavisideFilter():
    _beta   (0.),
    _eta    (0.),
    _v    (nullptr)
    { }
    
    virtual ~HeavisideFilter() {}
    
    inline void set_field(const FieldType& v) { _v = &v;}
    
    inline void set_parameters(const real_t beta, const real_t eta) {
        
        _beta = beta;
        _eta  = eta;
    }
    
    template <typename ContextType>
    inline ScalarType value(const ContextType& c) const {
        
        Assert0(_v, "Scalar field not initialized");
        
        return ((tanh(_beta*_eta)+tanh(_beta*(_v->value(c)-_eta))) /
                (tanh(_beta*_eta)+tanh(_beta*(1.-_eta))));
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType&     c,
                                 const ScalarFieldType& f) const {
        
        Assert0(_v, "Scalar field not initialized");

        return ((1.-pow(tanh(_beta*(_v->value(c)-_eta)),2))*
                _beta*_v->derivative(c, f) /
                (tanh(_beta*_eta)+tanh(_beta*(1.-_eta))));
    }

    
private:
    
    real_t                   _beta;
    real_t                   _eta;
    const FieldType         *_v;
};
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif  // __mast_simp_heaviside_filter_h__
