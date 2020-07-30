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

#ifndef __mast_simp_penalized_youngs_modulus_h__
#define __mast_simp_penalized_youngs_modulus_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {

template <typename ScalarType, typename PenalizedDensityType>
class PenalizedYoungsModulus {
    
public:
    
    using scalar_t = ScalarType;
    
    PenalizedYoungsModulus():
    _E0     (0.),
    _E_min  (0.),
    _d      (nullptr)
    { }
    
    virtual ~PenalizedYoungsModulus() {}
    
    inline void set_density(const PenalizedDensityType &d) { _d = &d;}
    
    inline void set_modulus(const ScalarType E0,
                            const ScalarType Emin) {
        _E0    = E0;
        _E_min = Emin;
    }
    
    template <typename ContextType>
    inline ScalarType value(const ContextType& c) const {
        
        Assert0(_d,  "Density field not initialized");
        
        return _E_min + _E0 * _d->value(c);
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType&     c,
                                 const ScalarFieldType& f) const {
        
        Assert0( _d, "Density field not initialized");
        
        return _E0 * _d->derivative(c, f);
    }

    
private:
    
    ScalarType                   _E0;
    ScalarType                   _E_min;
    const PenalizedDensityType  *_d;
};
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif  // __mast_simp_penalized_youngs_modulus_h__
