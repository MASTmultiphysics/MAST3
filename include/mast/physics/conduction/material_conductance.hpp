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

#ifndef __mast_isotropic_material_conductance_h__
#define __mast_isotropic_material_conductance_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Conduction {


template <typename ScalarType, typename ConductanceType, typename ContextType>
class IsotropicMaterialConductance {
public:
    
    using scalar_t     = typename ConductanceType::scalar_t;
    using value_t      = scalar_t;
    using is_isotropic = std::true_type;
    using is_linear    = std::true_type;
    
    IsotropicMaterialConductance():
    _k     (nullptr) { }
    
    virtual ~IsotropicMaterialConductance() {}
    
    inline void set_conductance(const ConductanceType& k) {
        
        _k  = &k;
    }
    
    
    inline const ConductanceType& get_k() const {return *_k;}
    
    inline void value(ContextType& c, value_t& m) const {
                
        Assert0(_k, "Material values not provided");
        
        m = _k->value(c);
    }
    
    
    template <typename ScalarFieldType>
    inline void derivative(ContextType&           c,
                           const ScalarFieldType& f,
                           value_t&               m) const {
        
        Assert0(_k, "Material values not provided");

        m = _k->derivative(c, f);
    }

private:
    
    const ConductanceType*  _k;
};




} // namespace Conductance
} // namespace Physics
} // namespace MAST


#endif // __mast_isotropic_material_conductance_h__
