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
#include <mast/mesh/libmesh/utility.hpp>

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
    
    template <typename VecType, typename ContextType>
    inline ScalarType compute(ContextType& c,
                              const VecType &density) const {
        
        ScalarType
        volume = 0.,
        rho    = 0.;
        
        uint_t
        sys_num = c.rho_sys->number(),
        n_nodes = 0;
        
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        density_accessor (*c.rho_sys, density);
        
        libMesh::MeshBase::element_iterator
        it    =  c.mesh->active_local_elements_begin(),
        end   =  c.mesh->active_local_elements_end();
        
        for ( ; it != end; it++) {
            
            const libMesh::Elem& e = **it;
            
            // compute the average element density value
            rho = 0.;
            
            n_nodes = MAST::Mesh::libMeshWrapper::Utility::n_linear_basis_nodes_on_elem(e);
            
            for (uint_t i=0; i<n_nodes; i++) {
                
                const libMesh::Node& n = *e.node_ptr(i);
                
                rho +=
                MAST::Numerics::Utility::get(density,
                                             n.dof_number(sys_num, 0, 0));
            }
            
            rho /= (1. * n_nodes);
            
            // use this density value to compute the volume
            volume  +=  e.volume() * rho;
        }
        
        MAST::Numerics::Utility::comm_sum(c.rho_sys->comm(), volume);
        
        return volume;
    }

    
    template <typename VecType, typename ContextType, typename DensityFilterType>
    inline ScalarType compute(ContextType             &c,
                              const VecType           &density,
                              const DensityFilterType &filter) const {
        
        ScalarType
        volume = 0.,
        rho    = 0.;
        
        uint_t
        sys_num = c.rho_sys->number(),
        n_nodes = 0;
        
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        density_accessor (*c.rho_sys, density);
        
        libMesh::MeshBase::element_iterator
        it    =  c.mesh->active_local_elements_begin(),
        end   =  c.mesh->active_local_elements_end();
        
        for ( ; it != end; it++) {
            
            const libMesh::Elem& e = **it;
            
            // compute the average element density value
            rho = 0.;
            
            n_nodes = MAST::Mesh::libMeshWrapper::Utility::n_linear_basis_nodes_on_elem(e);
            
            for (uint_t i=0; i<n_nodes; i++) {
                
                const libMesh::Node& n = *e.node_ptr(i);
                
                rho += filter.filter(MAST::Numerics::Utility::get
                                     (density, n.dof_number(sys_num, 0, 0)));
            }
            
            rho /= (1. * n_nodes);
            
            // use this density value to compute the volume
            volume  +=  e.volume() * rho;
        }
        
        MAST::Numerics::Utility::comm_sum(c.rho_sys->comm(), volume);
        
        return volume;
    }

    
    
    
    template <typename VecType,
              typename ContextType,
              typename GeometricFilterType>
    inline void derivative(ContextType                      &c,
                           const VecType                    &density,
                           const GeometricFilterType        &filter,
                           const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
                           std::vector<ScalarType> &sens) {
        
        Assert2(dvs.size() == sens.size(),
                dvs.size(), sens.size(),
                "DV and sensitivity vectors must have same size");

        uint_t
        n_density_dofs = c.rho_sys->n_dofs(),
        n_nodes        = 0;

        MAST::Numerics::Utility::setZero(sens);
        std::unique_ptr<VecType>
        v (MAST::Numerics::Utility::build<VecType>(*c.rho_sys).release()),
        v_filtered (MAST::Numerics::Utility::build<VecType>(*c.rho_sys).release());

        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        density_accessor (*c.rho_sys, density);

        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        // first compute the sensitivity information assuming unfiltered
        // density variables.
        real_t
        e_vol = 0.;
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for
            // the initialization routines
            c.elem = *el;
            e_vol  = c.elem->volume();
            
            density_accessor.init(*c.elem);

            const std::vector<libMesh::dof_id_type>
            &density_dof_ids = density_accessor.dof_indices();

            n_nodes = MAST::Mesh::libMeshWrapper::Utility::n_linear_basis_nodes_on_elem(**el);
            
            for (uint_t i=0; i<n_nodes; i++) {

                // each density coefficient shoudl appear only once
                // for each element. So, if the dof was found for this
                // element, then we will simply set the element sensitivity
                // to be the averaged value
                MAST::Numerics::Utility::add(*v,
                                             density_dof_ids[i],
                                             e_vol/(1. * n_nodes));
            }
        }
        
        MAST::Numerics::Utility::finalize(*v);

        // Now, combine the sensitivty with the filtering data
        filter.compute_reverse_filtered_values(*v, *v_filtered);

        uint_t
        idx  = 0;

        // copy the results back to sens
        for (uint_t i=dvs.local_begin(); i<dvs.local_end(); i++) {

            idx = dvs.get_data_for_parameter(dvs[i]).template get<int>("dof_id");
            sens[i] = MAST::Numerics::Utility::get(*v_filtered, idx);
        }

        MAST::Numerics::Utility::comm_sum(c.rho_sys->comm(), sens);
    }

    
    
    template <typename VecType,
              typename ContextType,
              typename DensityFilterType,
              typename GeometricFilterType>
    inline void derivative(ContextType                      &c,
                           const VecType                    &density,
                           const DensityFilterType          &density_filter,
                           const GeometricFilterType        &geom_filter,
                           const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
                           std::vector<ScalarType> &sens) {
        
        Assert2(dvs.size() == sens.size(),
                dvs.size(), sens.size(),
                "DV and sensitivity vectors must have same size");

        uint_t
        sys_num        = c.rho_sys->number(),
        n_density_dofs = c.rho_sys->n_dofs(),
        n_nodes        = 0;

        MAST::Numerics::Utility::setZero(sens);
        
        std::unique_ptr<VecType>
        v (MAST::Numerics::Utility::build<VecType>(*c.rho_sys).release()),
        v_filtered (MAST::Numerics::Utility::build<VecType>(*c.rho_sys).release());

        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        density_accessor (*c.rho_sys, density);

        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        // first compute the sensitivity information assuming unfiltered
        // density variables.
        real_t
        e_vol = 0.;
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for
            // the initialization routines
            c.elem = *el;
            e_vol  = c.elem->volume();
            
            density_accessor.init(*c.elem);

            const std::vector<libMesh::dof_id_type>
            &density_dof_ids = density_accessor.dof_indices();

            n_nodes = MAST::Mesh::libMeshWrapper::Utility::n_linear_basis_nodes_on_elem(**el);
            
            for (uint_t i=0; i<n_nodes; i++) {

                const libMesh::Node& n = *c.elem->node_ptr(i);

                // each density coefficient shoudl appear only once
                // for each element. So, if the dof was found for this
                // element, then we will simply set the element sensitivity
                // to be the averaged value
                MAST::Numerics::Utility::add
                (*v,
                 density_dof_ids[i],
                 e_vol/(1. * n_nodes) *
                 density_filter.filter_derivative(MAST::Numerics::Utility::get
                                                  (density, n.dof_number(sys_num, 0, 0)),
                                                  1.));
            }
        }
        
        MAST::Numerics::Utility::finalize(*v);
        
        // Now, combine the sensitivty with the filtering data
        geom_filter.compute_reverse_filtered_values(*v, *v_filtered);
        
        uint_t
        idx  = 0;

        // copy the results back to sens
        for (uint_t i=dvs.local_begin(); i<dvs.local_end(); i++) {

            idx = dvs.get_data_for_parameter(dvs[i]).template get<int>("dof_id");
            sens[i] = MAST::Numerics::Utility::get(*v_filtered, idx);
        }

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
