
#ifndef __mast_optimization_topology_simp_libmesh_residual_and_jacobian_h__
#define __mast_optimization_topology_simp_libmesh_residual_and_jacobian_h__

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
          typename ElemOpsType>
class ResidualAndJacobian {

public:
    
    static_assert(std::is_same<ScalarType, typename ElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");
    
    ResidualAndJacobian():
    _finalize_jac (true),
    _e_ops        (nullptr)
    { }
    
    virtual ~ResidualAndJacobian() { }
    
    inline void set_finalize_jac(bool f) { _finalize_jac = f;}
    
    inline void set_elem_ops(ElemOpsType& e_ops) { _e_ops = &e_ops; }

    template <typename VecType, typename MatType, typename ContextType>
    inline void assemble(ContextType   &c,
                         const VecType &X,
                         const VecType &density,
                         VecType       *R,
                         MatType       *J) {
        
        Assert0( R || J, "Atleast one assembled quantity should be specified.");
        
        if (R) MAST::Numerics::Utility::setZero(*R);
        if (J) MAST::Numerics::Utility::setZero(*J);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor     (*c.sys, X),
        density_accessor (*c.density_sys, density);

        using elem_vector_t = typename ElemOpsType::vector_t;
        using elem_matrix_t = typename ElemOpsType::matrix_t;
        
        elem_vector_t res_e, density_e;
        elem_matrix_t jac_e;
        
        
        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            density_accessor.init(*c.elem);
            
            res_e.setZero(sol_accessor.n_dofs());
            if (J) jac_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            
            // perform the element level calculations
            _e_ops->compute(c,
                            sol_accessor,
                            density_accessor,
                            res_e,
                            J?&jac_e:nullptr);
                        
            // constrain the quantities to account for hanging dofs,
            // Dirichlet constraints, etc.
            if (R && J)
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add_matrix_and_vector
                <ScalarType, VecType, MatType, elem_vector_t, elem_matrix_t>
                (*R, *J, c.sys->get_dof_map(), sol_accessor.dof_indices(), res_e, jac_e);
            else if (R)
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add_vector
                <ScalarType, VecType, elem_vector_t>
                (*R, c.sys->get_dof_map(), sol_accessor.dof_indices(), res_e);
            else
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add_matrix
                <ScalarType, MatType, elem_matrix_t>
                (*J, c.sys->get_dof_map(), sol_accessor.dof_indices(), jac_e);
        }

        // parallel matrix/vector require finalization of communication
        if (R) MAST::Numerics::Utility::finalize(*R);
        if (J && _finalize_jac) MAST::Numerics::Utility::finalize(*J);
    }
    
private:

    bool         _finalize_jac;
    ElemOpsType  *_e_ops;
};

} // namespace libMeshWrapper
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif // __mast_optimization_topology_simp_libmesh_residual_and_jacobian_h__
