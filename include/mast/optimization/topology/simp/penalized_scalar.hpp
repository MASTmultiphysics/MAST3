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

#ifndef __mast_simp_penalized_scalar_h__
#define __mast_simp_penalized_scalar_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {

template <typename ScalarType, typename PenalizedDensityType>
class PenalizedScalar {
    
public:
    
    using scalar_t = ScalarType;
    
    PenalizedScalar():
    _v0     (0.),
    _v_min  (0.),
    _d      (nullptr)
    { }
    
    virtual ~PenalizedScalar() {}
    
    inline void set_density(const PenalizedDensityType &d) { _d = &d;}
    
    inline void set_scalar(const ScalarType v0,
                           const ScalarType vmin) {
        _v0    = v0;
        _v_min = vmin;
    }
    
    template <typename ContextType>
    inline ScalarType value(const ContextType& c) const {
        
        Assert0(_d,  "Density field not initialized");
        
        return _v_min + _v0 * _d->value(c);
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType&     c,
                                 const ScalarFieldType& f) const {
        
        Assert0( _d, "Density field not initialized");
        
        return _v0 * _d->derivative(c, f);
    }

    
private:
    
    ScalarType                   _v0;
    ScalarType                   _v_min;
    const PenalizedDensityType  *_d;
};
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif  // __mast_simp_penalized_scalar_h__
