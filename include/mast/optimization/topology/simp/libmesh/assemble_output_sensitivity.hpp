
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
        std::vector<ScalarType>
        v (n_density_dofs, ScalarType()),
        v_filtered (n_density_dofs, ScalarType());
        
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
            
            sol_accessor.init(*c.elem);
            density_accessor.init(*c.elem);
            adj_accessor.init(*c.elem);
            
            density_accessor.init_dof_id_set(density_dofs);
            
            dres_e.setZero(sol_accessor.n_dofs());

            for (uint_t i=0; i<dvs.size(); i++) {
                
                // this assumes that if the DV (which is associated with a node)
                // is connected to this element, then the dof_indices for this
                // element will contain this index. If not, then the contribution
                // of this element to the sensitivity is zero.
                if (density_dofs.count(param_dof_ids[i])) {
                
                    const std::vector<libMesh::dof_id_type>
                    &density_dof_ids = density_accessor.dof_indices();
                    
                    // set a unit value of density sensitivity
                    // for this dof
                    drho.setZero(density_dof_ids.size());
                    for (uint_t i=0; i<density_dof_ids.size(); i++) {
                        
                        if (density_dof_ids[i] == param_dof_ids[i]) {
                            drho(i) = 1.;
                            break;
                        }
                    }
                    
                    // perform the element level calculations
                    _e_ops->derivative(c,
                                       dvs[i],
                                       sol_accessor,
                                       density_accessor,
                                       drho,
                                       dres_e,
                                       nullptr);
                    v[param_dof_ids[i]] += _output_e_ops->derivative(c,
                                                                     dvs[i],
                                                                     sol_accessor,
                                                                     density_accessor,
                                                                     drho);
                    v[param_dof_ids[i]] += adj_accessor.dot(dres_e);
                }
            }
        }
        
        // Now, combine the sensitivty with the filtering data
        filter.compute_filtered_values(dvs, v, v_filtered);
        
        // copy the results back to sense
        for (uint_t i=0; i<param_dof_ids.size(); i++)
            sens[i] = v_filtered[param_dof_ids[i]];
        
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
