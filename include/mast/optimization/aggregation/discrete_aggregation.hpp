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

#ifndef __mast_optimization_discrete_aggregation_h__
#define __mast_optimization_discrete_aggregation_h__

// C++ includes
#include <algorithm>

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/numerics/utility.hpp>


namespace MAST {
namespace Optimization {
namespace Aggregation {

template <typename ScalarType>
ScalarType
aggregate_minimum(const std::vector<ScalarType> &vec,
                  const real_t                   p) {
    
    ScalarType
    v      = ScalarType(),
    v_min  = ScalarType();
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);
    
    for (uint_t i=0; i<vec.size(); i++) {
        
        v += exp(-p * (vec[i] - v_min));
    }
    
    v = v_min - log(v) / p;
    
    return v;
}


template <typename ScalarType>
ScalarType
aggregate_minimum_sensitivity(const std::vector<ScalarType> &vec,
                              const uint_t                   i,
                              const ScalarType              &dv_i,
                              const real_t                   p) {
    
    ScalarType
    v      = ScalarType(),
    v_min  = ScalarType();
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);

    for (uint_t i=0; i<vec.size(); i++) {
        
        v  += exp(-p * (vec[i] - v_min));
    }
    
    v = exp(-p * (vec[i] - v_min)) * dv_i / v;
    
    return v;
}



template <typename ScalarType>
ScalarType
aggregate_minimum_sensitivity(const std::vector<ScalarType> &vec,
                              const std::vector<ScalarType> &dvec,
                              const real_t                   p) {
    
    ScalarType
    dv     = ScalarType(),
    v      = ScalarType(),
    v_min  = ScalarType();
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);

    for (uint_t i=0; i<vec.size(); i++) {
        
        dv += exp(-p * (vec[i] - v_min)) * dvec[i];
        v  += exp(-p * (vec[i] - v_min));
    }
    
    v = dv / v;
    
    return v;
}



template <typename ScalarType>
ScalarType
aggregate_maximum(const std::vector<ScalarType> &vec,
                  const real_t                   p) {
    
    ScalarType
    v      = ScalarType(),
    v_max  = ScalarType();

    v_max = MAST::Numerics::Utility::real_maximum(vec);

    for (uint_t i=0; i<vec.size(); i++) {
        
        v += exp(p * (vec[i] - v_max));
    }
    
    v = v_max + log(v) / p;
    
    return v;
}


} // Aggregation
} // Optimization
} // MAST

#endif // __mast_optimization_discrete_aggregation_h__
