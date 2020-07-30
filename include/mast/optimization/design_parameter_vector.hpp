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

#ifndef __mast_optimization_design_parameter_vector_h__
#define __mast_optimization_design_parameter_vector_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/base/parameter_data.hpp>
#include <mast/optimization/design_parameter.hpp>

namespace MAST {
namespace Optimization {


template <typename ScalarType>
class DesignParameterVector {
    
public:
    DesignParameterVector() { }
    
    virtual ~DesignParameterVector() {
        
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::iterator
        it   = _data.begin(),
        end  = _data.end();

        for (; it != end; it++)
            delete it->second;
    }
    
    inline uint_t size() const { return _parameters.size(); }
    
    inline MAST::Optimization::DesignParameter<ScalarType>&
    operator[](uint_t i) {
        
        Assert2(i < _parameters.size(),
                i, _parameters.size(),
                "Invalid parameter index");
        
        return *_parameters[i];
    }

    
    inline bool
    is_design_parameter_index(const uint_t i) const {
        
        return _dv_index.count(i);
    }
    
    
    inline const MAST::Optimization::DesignParameter<ScalarType>&
    operator[](uint_t i) const {
        
        Assert2(i < _parameters.size(),
                i, _parameters.size(),
                "Invalid parameter index");
        
        return *_parameters[i];
    }

    
    template <typename T>
    inline T get_parameter_for_dv(uint_t i, const std::string& nm) const {
        
        return this->get_data_for_parameter((*this)[i]).template get<T>(nm);
    }
    
    
    inline MAST::Base::ParameterData&
    add_parameter(MAST::Optimization::DesignParameter<ScalarType>& p) {

        // make sure this does not exist
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::iterator
        it   = _data.find(&p);
        
        Assert0(it == _data.end(), "Parameter already exists");
        
        _parameters.push_back(&p);
        
        MAST::Base::ParameterData* d = new MAST::Base::ParameterData;
        _data[&p] = d;
        
        return *d;
    }


    /*!
     * \param id is the location of the coefficient in the global vector that stores the scalar field (density or level-set)
     *  used to define the topology.
     */
    inline MAST::Base::ParameterData&
    add_topology_parameter(MAST::Optimization::DesignParameter<ScalarType>& p,
                           const uint_t id) {

        // make sure this does not exist
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::iterator
        it   = _data.find(&p);
        
        Assert0(it == _data.end(), "Parameter already exists");
        
        _parameters.push_back(&p);
        
        MAST::Base::ParameterData* d = new MAST::Base::ParameterData;
        _data[&p] = d;

        d->add<int>("dof_id") = id;
        _dv_index.insert(id);
        
        return *d;
    }

    
    inline const MAST::Base::ParameterData&
    get_data_for_parameter(const MAST::Optimization::DesignParameter<ScalarType>& p) const {

        // make sure this does not exist
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::const_iterator
        it   = _data.find(&p);
        
        Assert0(it != _data.end(), "Parameter does not exists in vector");
        
        return *it->second;
    }


    inline MAST::Base::ParameterData&
    get_data_for_parameter(const MAST::Optimization::DesignParameter<ScalarType>& p) {

        // make sure this does not exist
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::iterator
        it   = _data.find(&p);
        
        Assert0(it != _data.end(), "Parameter does not exists in vector");
        
        return *it->second;
    }

    
private:
    
    std::vector<MAST::Optimization::DesignParameter<ScalarType>*>    _parameters;
    std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
             MAST::Base::ParameterData*> _data;
    std::set<uint_t>                     _dv_index;
};

} // namespace Optimization
} // namespace MAST


#endif // __mast_optimization_design_parameter_vector_h__
