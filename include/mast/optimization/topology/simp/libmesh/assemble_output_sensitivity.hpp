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

#ifndef __mast_optimization_topology_simp_libmesh_output_sensitivity_h__
#define __mast_optimization_topology_simp_libmesh_output_sensitivity_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/base/assembly/libmesh/utility.hpp>
#include <mast/base/assembly/libmesh/accessor.hpp>
#include <mast/numerics/utility.hpp>
#include <mast/optimization/design_parameter_vector.hpp>

// libMesh includes
#include <libmesh/nonlinear_implicit_system.h>
#include <libmesh/dof_map.h>


namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {
namespace libMeshWrapper {

template <typename ScalarType,
          typename ResidualElemOpsType,
          typename OutputElemOpsType>
class AssembleOutputSensitivity {
    
public:

    static_assert(std::is_same<ScalarType,
                  typename ResidualElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");
    static_assert(std::is_same<ScalarType,
                  typename OutputElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");

    
    AssembleOutputSensitivity():
    _e_ops        (nullptr),
    _output_e_ops (nullptr)
    { }
    
    virtual ~AssembleOutputSensitivity() {}
    
    inline void set_elem_ops(ResidualElemOpsType &e_ops,
                             OutputElemOpsType   &output_ops) {
        
        _e_ops        = &e_ops;
        _output_e_ops = &output_ops;
    }

    /*!
     *  output derivative is defined as a
     * \f[ \frac{dQ}{d\alpha} = \frac{\partial Q}{\partial \alpha} + \lambda^T \frac{\partial R}{\partial \alpha} \f]
     */
    template <typename Vec1Type,
              typename Vec2Type,
              typename ContextType,
              typename FilterType>
    inline void assemble(ContextType               &c,
                         const Vec1Type            &X,
                         const Vec2Type            &density,
                         const Vec1Type            &X_adj,
                         const FilterType          &filter,
                         const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
                         std::vector<ScalarType>   &sens) {
                
        Assert0(_e_ops && _output_e_ops, "Elem Operation objects not initialized");
        Assert2(density.size() == c.rho_sys->n_dofs(),
                density.size(), c.rho_sys->n_dofs(),
                "Density coefficients must be provided for whole mesh");
        Assert2(dvs.size() == sens.size(),
                dvs.size(), sens.size(),
                "DV and sensitivity vectors must have same size");
           
        const uint_t
        n_density_dofs = c.rho_sys->n_dofs();

        MAST::Numerics::Utility::setZero(sens);

        std::unique_ptr<Vec2Type>
        v (MAST::Numerics::Utility::build<Vec2Type>(*c.rho_sys).release()),
        v_filtered (MAST::Numerics::Utility::build<Vec2Type>(*c.rho_sys).release());

        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, Vec1Type>
        sol_accessor     (*c.sys, X),
        adj_accessor     (*c.sys, X_adj);
        
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, Vec2Type>
        density_accessor (*c.rho_sys, density);

        using elem_vector_t = typename ResidualElemOpsType::vector_t;
        using elem_matrix_t = typename ResidualElemOpsType::matrix_t;
        
        elem_vector_t
        dres_e,
        drho;

        uint_t
        idx = 0;

        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        // first compute the sensitivity information assuming unfiltered
        // density variables.
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for
            // the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            density_accessor.init(*c.elem);
            adj_accessor.init(*c.elem);
            
            const std::vector<libMesh::dof_id_type>
            &density_dof_ids = density_accessor.dof_indices();

            for (uint_t i=0; i<c.elem->n_nodes(); i++) {
                
                if (!dvs.is_design_parameter_dof_id(density_dof_ids[i]))
                    continue;

                // this assumes that if the DV (which is associated with a node)
                // is connected to this element, then the dof_indices for this
                // element will contain this index. If not, then the contribution
                // of this element to the sensitivity is zero.
                
                idx    = dvs.get_dv_id_for_topology_dof(density_dof_ids[i]);
                const MAST::Optimization::DesignParameter<ScalarType>
                &dv    = dvs[idx];
                
                // set a unit value of density sensitivity
                // for this dof
                drho.setZero(density_dof_ids.size());
                drho(i) = 1.;
                
                dres_e.setZero(sol_accessor.n_dofs());
                
                // first we compute the partial derivative of the
                // residual wrt the parameter.
                _e_ops->derivative(c,
                                   dv,
                                   sol_accessor,
                                   density_accessor,
                                   drho,
                                   dres_e,
                                   nullptr);
                
                // next, we compute the partial derivative derivative of
                // the output functional and add it to the sensitivity result
                MAST::Numerics::Utility::add
                (*v,
                 density_dof_ids[i],
                 _output_e_ops->derivative(c, // partial derivative of output
                                           dv,
                                           sol_accessor,
                                           density_accessor,
                                           drho)
                 + adj_accessor.dot(dres_e)); // the adjoint vector combined w/ res sens
            }
        }
        
        MAST::Numerics::Utility::finalize(*v);

        // Now, combine the sensitivty with the filtering data
        filter.compute_reverse_filtered_values(*v, *v_filtered);

        // copy the results back to sens
        for (uint_t i=dvs.local_begin(); i<dvs.local_end(); i++) {

            idx = dvs.get_data_for_parameter(dvs[i]).template get<int>("dof_id");
            sens[i] = MAST::Numerics::Utility::get(*v_filtered, idx);
        }
        
        MAST::Numerics::Utility::comm_sum(c.rho_sys->comm(), sens);
    }

private:
  
    ResidualElemOpsType  *_e_ops;
    OutputElemOpsType    *_output_e_ops;
};

} // namespace libMeshWrapper
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST


#endif // __mast_optimization_topology_simp_libmesh_output_sensitivity_h__
