
#ifndef __mast_optimization_topology_simp_libmesh_output_sensitivity_h__
#define __mast_optimization_topology_simp_libmesh_output_sensitivity_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/base/assembly/libmesh/utility.hpp>
#include <mast/base/assembly/libmesh/accessor.hpp>
#include <mast/numerics/utility.hpp>

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

    static_assert(std::is_same<ScalarType, typename ElemOpsType::scalar_t>::value,
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
    template <typename VecType,
              typename ContextType,
              typename ScalarFieldType,
              typename DVType,
              typename FilterType>
    inline void assemble(ContextType               &c,
                         const ScalarFieldType     &f,
                         const VecType             &X,
                         const VecType             &X_adj,
                         const FilterType          &filter,
                         const std::vector<DVType> &dvs,
                         std::vector<ScalarType>   &sens) {
                
        Assert0(_e_ops && _output_e_ops, "Elem Operation objects not initialized");
        Assert2(dvs.size() == sens.size(),
                dvs.size(), sens.size(),
                "DV and sensitivity vectors must have same size");
        
        MAST::Numerics::Utility::setZero(sens);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor(*c.sys, X),
        adj_accessor(*c.sys, X_adj);

        using elem_vector_t = typename ElemOpsType::vector_t;
        using elem_matrix_t = typename ElemOpsType::matrix_t;
        
        elem_vector_t  dres_e;
        
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
            adj_accessor.init(*c.elem);

            dres_e.setZero(sol_accessor.n_dofs());

            for (uint_t i=0; i<dvs.size(); i++) {
                
                // perform the element level calculations
                _e_ops->derivative(c, *dvs[i], sol_accessor, dres_e, nullptr);
                sens[i]  = _output_e_ops->derivative(c, *dvs[i], sol_accessor);
                sens[i] += adj_accessor.dot(dres_e);
            }
        }
        
        // Now, combine the sensitivty with the filtering data
        filter.compute_filtered_values
        <ScalarType, >
();
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
