
#ifndef __mast_libmesh_residual_and_jacobian_h__
#define __mast_libmesh_residual_and_jacobian_h__

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
namespace Base {
namespace Assembly {
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
                         VecType       *R,
                         MatType    *J) {
        
        Assert0( R || J, "Atleast one assembled quantity should be specified.");
        
        if (R) MAST::Numerics::Utility::setZero(*R);
        if (J) MAST::Numerics::Utility::setZero(*J);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor(*c.sys, X);

        typename ElemOpsType::vector_t
        res_e;
        typename ElemOpsType::matrix_t
        jac_e;
        
        
        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            
            res_e.setZero(sol_accessor.n_dofs());
            if (J) jac_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            
            // perform the element level calculations
            _e_ops->compute(c, sol_accessor, res_e, J?&jac_e:nullptr);
            
            // copy to the libMesh matrix for further processing
            using sub_vec_t = libMesh::DenseVector<ScalarType>;
            using sub_mat_t = libMesh::DenseMatrix<ScalarType>;
            sub_vec_t v;
            sub_mat_t m;
            if (R) MAST::Numerics::Utility::copy(res_e, v);
            if (J) MAST::Numerics::Utility::copy(jac_e, m);
            
            // constrain the quantities to account for hanging dofs,
            // Dirichlet constraints, etc.
            if (R && J)
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add
                <ScalarType, VecType, MatType, sub_vec_t, sub_mat_t>
                (*R, *J, c.sys->get_dof_map(), sol_accessor.dof_indices(), v, m);
            else if (R)
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add
                <ScalarType, VecType, sub_vec_t>
                (*R, c.sys->get_dof_map(), sol_accessor.dof_indices(), v);
            else
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add
                <ScalarType, MatType, sub_mat_t>
                (*J, c.sys->get_dof_map(), sol_accessor.dof_indices(), m);
        }

        // parallel matrix/vector require finalization of communication
        if (R) MAST::Numerics::Utility::finalize(*R);
        if (J && _finalize_jac) MAST::Numerics::Utility::finalize(*J);
    }
    
private:

    bool         _finalize_jac;
    ElemOpsType  *_e_ops;
};

} // namespace libMesh
} // namespace Assembly
} // namespace Base
} // namespace MAST

#endif // __mast_libmesh_residual_and_jacobian_h__
