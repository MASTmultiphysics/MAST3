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

// TIMPI includes
#include <libmesh/parallel.h>

namespace MAST {
namespace Optimization {


template <typename ScalarType>
class DesignParameterVector {
    
public:

    DesignParameterVector(const libMesh::Parallel::Communicator  &comm):
    _comm    (comm)
    { }
    
    
    virtual ~DesignParameterVector() {
        
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::iterator
        it   = _data.begin(),
        end  = _data.end();

        for (; it != end; it++)
            delete it->second;
    }
    
    
    inline uint_t size() const {
        
        Assert0(_rank_begin_index.size(),
                "Data must be synchronized before call to size()");
        
        return _rank_end_index[_comm.size()-1];
    }
    
    
    inline uint_t local_begin() const {

        Assert0(_rank_begin_index.size(),
                "Data must be synchronized before call to size()");
        
        return _rank_begin_index[_comm.rank()];
    }

    
    inline uint_t local_end() const {

        Assert0(_rank_begin_index.size(),
                "Data must be synchronized before call to size()");
        
        return _rank_end_index[_comm.rank()];
    }

    
    inline MAST::Optimization::DesignParameter<ScalarType>&
    operator[](uint_t i) {
        
        Assert2(i >= _rank_begin_index[_comm.rank()],
                i, _rank_begin_index[_comm.rank()],
                "Invalid parameter index for rank");

        Assert2(i < _rank_end_index[_comm.rank()],
                i, _rank_end_index[_comm.rank()],
                "Invalid parameter index for rank");

        return *_parameters[i-_rank_begin_index[_comm.rank()]];
    }

    
    inline bool
    is_design_parameter_index(const uint_t i) const {
        
        return _dv_index.count(i);
    }
    
    
    inline const MAST::Optimization::DesignParameter<ScalarType>&
    operator[](uint_t i) const {

        Assert2(i >= _rank_begin_index[_comm.rank()],
                i, _rank_begin_index[_comm.rank()],
                "Invalid parameter index for rank");

        Assert2(i < _rank_end_index[_comm.rank()],
                i, _rank_end_index[_comm.rank()],
                "Invalid parameter index for rank");
        
        return *_parameters[i-_rank_begin_index[_comm.rank()]];
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

    inline void synchronize() {
        
        Assert0(!_rank_begin_index.size(), "Data already synchronized");
        Assert0(_parameters.size(), "Parameters not initialized on this rank");

        std::vector<int_t>
        rank_dvs(_comm.size(), 0);
        
        _rank_begin_index.resize(_comm.size());
        _rank_end_index.resize(_comm.size());

        // initialize the number of DVs for the current rank
        rank_dvs[_comm.rank()] = _parameters.size();
        
        // now obtain these values from each rank
        _comm.sum(rank_dvs);
        
        // now identify the beginning IDs for each processor
        _rank_begin_index[0] = 0;
        _rank_end_index[0]   = rank_dvs[0];
        
        for (uint_t i=1; i<_comm.size(); i++) {

            _rank_begin_index[i]  = _rank_begin_index[i-1] + rank_dvs[i-1];
            _rank_end_index[i]    = _rank_end_index[i-1]   + rank_dvs[i];
        }
    }
    
private:
    
    const libMesh::Parallel::Communicator&                           _comm;
    std::vector<MAST::Optimization::DesignParameter<ScalarType>*>    _parameters;
    std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
             MAST::Base::ParameterData*> _data;
    std::set<uint_t>                     _dv_index;
    std::vector<uint_t>                  _rank_begin_index;
    std::vector<uint_t>                  _rank_end_index;
};

} // namespace Optimization
} // namespace MAST


#endif // __mast_optimization_design_parameter_vector_h__
