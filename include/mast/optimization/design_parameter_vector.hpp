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

// libMesh includes
#include <libmesh/parallel.h>
#include <libmesh/dof_map.h>

namespace MAST {
namespace Optimization {


template <typename ScalarType>
class DesignParameterVector {
    
public:
    
    using dv_id_param_map_t = std::map<uint_t, MAST::Optimization::DesignParameter<ScalarType>*>;
    

    DesignParameterVector(const libMesh::Parallel::Communicator  &comm):
    _comm    (comm)
    { }
    
    
    virtual ~DesignParameterVector() {
        
        {
            typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
            MAST::Base::ParameterData*>::iterator
            it   = _data.begin(),
            end  = _data.end();
            
            for (; it != end; it++)
                delete it->second;
        }
        
        {
            typename std::map<uint_t, MAST::Optimization::DesignParameter<ScalarType>*>::iterator
            it   = _parameters.begin(),
            end  = _parameters.end();
            
            for (; it != end; it++)
                delete it->second;
        }
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
        
        typename std::map<uint_t, MAST::Optimization::DesignParameter<ScalarType>*>::iterator
        it   = _parameters.find(i);
        
        Assert0(it != _parameters.end(), "Invalid parameter index for rank");

        return *it->second;
    }

    
    
    inline const MAST::Optimization::DesignParameter<ScalarType>&
    operator[](uint_t i) const {

        typename std::map<uint_t, MAST::Optimization::DesignParameter<ScalarType>*>::const_iterator
        it   = _parameters.find(i);
        
        Assert0(it != _parameters.end(), "Invalid parameter index for rank");

        return *it->second;
    }

    
    inline const dv_id_param_map_t& get_dv_map() const {
        
        return _parameters;
    }
    

    inline uint_t get_dv_id_for_topology_dof(const uint_t id) const {
    
        std::map<uint_t, uint_t>::const_iterator
        it  = _dof_id_to_dv_id_map.find(id);
        
        Assert1(it != _dof_id_to_dv_id_map.end(),
                id, "dof ID not in this vector");
        
        return it->second;
    }
    
    
    
    inline bool
    is_design_parameter_dof_id(const uint_t i) const {
        
        return _dof_id_to_dv_id_map.count(i);
    }


    inline bool
    is_design_parameter_index(const uint_t i) const {
        
        return _dv_index.count(i);
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
        
        _local_parameters.push_back(&p);
        
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
        
        _local_parameters.push_back(&p);
        
        MAST::Base::ParameterData* d = new MAST::Base::ParameterData;
        _data[&p] = d;

        d->add<int>("dof_id") = id;
        _dv_index.insert(id);

        return *d;
    }

    
    inline MAST::Base::ParameterData&
    add_ghosted_topology_parameter(MAST::Optimization::DesignParameter<ScalarType>& p,
                                   const uint_t id) {

        // make sure this does not exist
        typename std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
                          MAST::Base::ParameterData*>::iterator
        it   = _data.find(&p);
        
        Assert0(it == _data.end(), "Parameter already exists");
        
        _ghosted_parameters.push_back(&p);
        
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

    inline void synchronize(const libMesh::DofMap &dof_map) {
        
        Assert0(!_rank_begin_index.size(), "Data already synchronized");
        Assert0(_local_parameters.size(), "Parameters not initialized on this rank");

        std::vector<int_t>
        rank_dvs(_comm.size(), 0);
        
        _rank_begin_index.resize(_comm.size());
        _rank_end_index.resize(_comm.size());

        // initialize the number of DVs for the current rank
        rank_dvs[_comm.rank()] = _local_parameters.size();
        
        // now obtain these values from each rank
        _comm.sum(rank_dvs);
        
        // now identify the beginning IDs for each processor
        _rank_begin_index[0] = 0;
        _rank_end_index[0]   = rank_dvs[0];
        
        for (uint_t i=1; i<_comm.size(); i++) {

            _rank_begin_index[i]  = _rank_begin_index[i-1] + rank_dvs[i-1];
            _rank_end_index[i]    = _rank_end_index[i-1]   + rank_dvs[i];
        }

        // assign the DV Ids and populate the DofID map
        uint_t
        dof_id = 0,
        owner  = 0;

        std::map<uint_t, MAST::Optimization::DesignParameter<ScalarType>*>
        dof_to_ghost_param_map;
        
        for (uint_t i=0; i<_local_parameters.size(); i++) {
            
            dof_id = this->get_data_for_parameter(*_local_parameters[i]).template get<int>("dof_id");
            _local_parameters[i]->set_id(_rank_begin_index[_comm.rank()]+i);
            _dof_id_to_dv_id_map[dof_id] = _local_parameters[i]->id();
        }
            
        
        // ghosted dofs indices needed from each rank
        std::vector<std::vector<uint_t>>
        ghosted_indices_on_rank_send(_comm.size()),
        ghosted_indices_on_rank_recv(_comm.size()),
        ghosted_dv_id_on_rank_recv(_comm.size());
        
        // figure out the DV IDs for the ghosted parameters
        for (uint_t i=0; i<_ghosted_parameters.size(); i++) {
            
            dof_id = this->get_data_for_parameter(*_ghosted_parameters[i]).template get<int>("dof_id");
            owner  = dof_map.dof_owner(dof_id);
            
            dof_to_ghost_param_map[dof_id] = _ghosted_parameters[i];
            ghosted_indices_on_rank_send[owner].push_back(dof_id);
        }
        
        // now ask respective processors for the indices
        for (uint_t i=0; i<_comm.size(); i++) {
            
            for (uint_t j=0; j<_comm.size(); j++) {
                
                if ( i != j) {
                    
                    // send to the j^th processor if it is my turn,
                    // else, receive from the ith processor
                    if (i == _comm.rank())
                        _comm.send(j, ghosted_indices_on_rank_send[j]);
                    else if (j == _comm.rank())
                        _comm.receive(i, ghosted_indices_on_rank_recv[i]);
                }
            }
        }
        
        // now that we have received all the dof indices, we are going to send
        // back to the respective processor the DV id that corresponds to the
        // dof indices.
        

        for (uint_t i=0; i<_comm.size(); i++) {
            
            for (uint_t k=0; k<ghosted_indices_on_rank_recv[i].size(); k++) {
                
                // make sure that the indices are all local
                Assert2(ghosted_indices_on_rank_recv[i][k]
                        >= dof_map.first_dof(_comm.rank()),
                        ghosted_indices_on_rank_recv[i][k],
                        dof_map.first_dof(_comm.rank()),
                        "Requested dof does not belong to this processor");
                Assert2(ghosted_indices_on_rank_recv[i][k]
                        < dof_map.end_dof(_comm.rank()),
                        ghosted_indices_on_rank_recv[i][k],
                        dof_map.end_dof(_comm.rank()),
                        "Requested dof does not belong to this processor");
                
                // now identify the DV Id number for these dofs
                std::map<uint_t, uint_t>::const_iterator
                it  = _dof_id_to_dv_id_map.find(ghosted_indices_on_rank_recv[i][k]);
                
                Assert0(it != _dof_id_to_dv_id_map.end(),
                        "No DV Id found for this dof id");
                
                ghosted_indices_on_rank_recv[i][k] = it->second;
            }
        }
        
        // now we communicate this information back to the processors
        for (uint_t i=0; i<_comm.size(); i++) {
            
            for (uint_t j=0; j<_comm.size(); j++) {
                
                if (( i == _comm.rank() && ghosted_indices_on_rank_recv[j].size()) ||
                    ( j == _comm.rank() && ghosted_indices_on_rank_send[i].size())) {
                    
                    // send to the i^th processor if it is my turn,
                    // else, receive from the j^th processor
                    if (i == _comm.rank())
                        _comm.send(j, ghosted_indices_on_rank_recv[j]);
                    else if (j == _comm.rank()) {
                        
                        _comm.receive(i, ghosted_dv_id_on_rank_recv[i]);

                        Assert2(ghosted_indices_on_rank_send[i].size() ==
                                ghosted_dv_id_on_rank_recv[i].size(),
                                ghosted_indices_on_rank_send[i].size(),
                                ghosted_dv_id_on_rank_recv[i].size(),
                                "Dof and DV ID map sizes must be same");
                        
                        uint_t
                        dof_id = 0,
                        dv_id  = 0;
                        
                        for (uint_t k=0; k<ghosted_dv_id_on_rank_recv[i].size(); k++) {

                            dof_id = ghosted_indices_on_rank_send[i][k];
                            dv_id  = ghosted_dv_id_on_rank_recv[i][k];
                            dof_to_ghost_param_map[dof_id]->set_id(dv_id);
                            
                            // also store this information in the map
                            _dof_id_to_dv_id_map[dof_id] = dv_id;
                        }
                    }
                }
            }
        }
        
        // now, populate the parameters map
        for (uint_t i=0; i<_local_parameters.size(); i++)
            _parameters[_local_parameters[i]->id()] = _local_parameters[i];
        
        for (uint_t i=0; i<_ghosted_parameters.size(); i++)
            _parameters[_ghosted_parameters[i]->id()] = _ghosted_parameters[i];

        
        // we don't need these any more, so we clear them.
        _local_parameters.clear();
        _ghosted_parameters.clear();
    }
    
private:
    
    const libMesh::Parallel::Communicator&                             _comm;
    std::map<uint_t, MAST::Optimization::DesignParameter<ScalarType>*> _parameters;
    std::vector<MAST::Optimization::DesignParameter<ScalarType>*>      _local_parameters;
    std::vector<MAST::Optimization::DesignParameter<ScalarType>*>      _ghosted_parameters;
    std::map<const MAST::Optimization::DesignParameter<ScalarType>*,
             MAST::Base::ParameterData*> _data;
    std::set<uint_t>                     _dv_index;
    std::map<uint_t, uint_t>             _dof_id_to_dv_id_map;
    std::vector<uint_t>                  _rank_begin_index;
    std::vector<uint_t>                  _rank_end_index;
};

} // namespace Optimization
} // namespace MAST


#endif // __mast_optimization_design_parameter_vector_h__ 
