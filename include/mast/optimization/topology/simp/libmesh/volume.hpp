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

#ifndef __mast_optimization_topology_simp_libmesh_volume_h__
#define __mast_optimization_topology_simp_libmesh_volume_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/utility.hpp>

// libMesh includes
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/dof_map.h>
#include <libmesh/elem.h>


namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {
namespace libMeshWrapper {

template <typename ScalarType>
class Volume {
    
public:
    
    Volume() {}
    virtual ~Volume() {}
    
    template <typename ContextType, typename VecType>
    inline ScalarType compute(ContextType& c,
                              const VecType &density) const {
        
        ScalarType
        volume = 0.,
        rho    = 0.;
        
        const uint_t
        sys_num = c.rho_sys->number();
        
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        density_accessor (*c.rho_sys, density);
        
        libMesh::MeshBase::element_iterator
        it    =  c.mesh->active_local_elements_begin(),
        end   =  c.mesh->active_local_elements_end();
        
        for ( ; it != end; it++) {
            
            const libMesh::Elem& e = **it;
            
            // compute the average element density value
            rho = 0.;
            
            for (uint_t i=0; i<e.n_nodes(); i++) {
                
                const libMesh::Node& n = *e.node_ptr(i);
                
                rho +=
                MAST::Numerics::Utility::get(density,
                                             n.dof_number(sys_num, 0, 0));
            }
            
            rho /= (1. * e.n_nodes());
            
            // use this density value to compute the volume
            volume  +=  e.volume() * rho;
        }
        
        MAST::Numerics::Utility::comm_sum(c.rho_sys->comm(), volume);
        
        return volume;
    }
    
    
    template <typename VecType,
              typename ContextType,
              typename FilterType>
    inline void derivative(ContextType             &c,
                           const VecType           &density,
                           const FilterType        &filter,
                           const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
                           std::vector<ScalarType> &sens) {
        
        Assert2(dvs.size() == sens.size(),
                dvs.size(), sens.size(),
                "DV and sensitivity vectors must have same size");

        const uint_t
        n_density_dofs = c.rho_sys->n_dofs();

        MAST::Numerics::Utility::setZero(sens);
        std::vector<ScalarType>
        v (n_density_dofs, ScalarType()),
        v_filtered (n_density_dofs, ScalarType());

        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        density_accessor (*c.rho_sys, density);

        std::vector<uint_t>
        param_dof_ids(dvs.size());
        
        // cache values for later use
        for (uint_t i=0; i<dvs.size(); i++)
            param_dof_ids[i] = dvs.get_data_for_parameter(dvs[i]).template get<int>("dof_id");
        
        std::set<uint_t> density_dofs;
        
        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        // first compute the sensitivity information assuming unfiltered
        // density variables.
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for
            // the initialization routines
            c.elem = *el;
            
            density_accessor.init(*c.elem);
            density_accessor.init_dof_id_set(density_dofs);

            for (uint_t i=0; i<dvs.size(); i++) {
                
                // this assumes that if the DV (which is associated with a node)
                // is connected to this element, then the dof_indices for this
                // element will contain this index. If not, then the contribution
                // of this element to the sensitivity is zero.
                if (density_dofs.count(param_dof_ids[i])) {
                
                    // each density coefficient shoudl appear only once
                    // for each element. So, if the dof was found for this
                    // element, then we will simply set the element sensitivity
                    // to be the averaged value
                    v[param_dof_ids[i]]  +=  c.elem->volume()/(1. * c.elem->n_nodes());
                }
            }
        }
        
        // Now, combine the sensitivty with the filtering data
        filter.compute_reverse_filtered_values(dvs, v, v_filtered);

        // copy the results back to sense
        for (uint_t i=0; i<param_dof_ids.size(); i++)
            sens[i] = v_filtered[param_dof_ids[i]];

        MAST::Numerics::Utility::comm_sum(c.rho_sys->comm(), sens);
    }
    
private:
    
};

}  // namespace libMeshWrapper
}  // namespace SIMP
}  // namespace Topology
}  // namespace Optimization
}  // namespace MAST


#endif // __mast_optimization_topology_simp_libmesh_volume_h__
