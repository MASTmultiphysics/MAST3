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

#ifndef __mast_simp_penalized_density_h__
#define __mast_simp_penalized_density_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {

template <typename ScalarType, typename DensityFieldType>
class PenalizedDensity {
    
public:
    
    PenalizedDensity():
    _p   (0),
    _d   (nullptr)
    { }
    
    virtual ~PenalizedDensity() {}
    
    inline void set_density_field(const DensityFieldType& d) { _d = &d;}
    
    inline void set_penalty(const real_t p) { _p = p;}
    
    template <typename ContextType>
    inline ScalarType value(const ContextType& c) const {
        
        Assert0(_p, "Penalty value not initialized");
        Assert0(_d, "Density field not initialized");
        
        return pow(_d->value(c), _p);
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType&     c,
                                 const ScalarFieldType& f) const {
        
        Assert0(_p, "Penalty value not initialized");
        Assert0(_d, "Density field not initialized");
        
        return _p * pow(_d->value(c), _p-1.) * _d->derivative(c, f);
    }

    
private:
    
    real_t                   _p;
    const DensityFieldType  *_d;
};
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif  // __mast_simp_penalized_density_h__ 
