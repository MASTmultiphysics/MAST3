
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
          typename ElemOpsType,
          typename ContextType>
class ResidualAndJacobian:
public libMesh::NonlinearImplicitSystem::ComputeResidualandJacobian {

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

    template <typename VecType, typename MatrixType>
    inline void assemble(ContextType   &c,
                         const VecType &X,
                         VecType       *R,
                         MatrixType    *J) {
        
        Assert0( R || J, "Atleast one assembled quantity should be specified.");
        
        libMesh::NonlinearImplicitSystem& sys = c.system();
        
        if (R) MAST::Numerics::Utility::setZero(*R);
        if (J) MAST::Numerics::Utility::setZero(*J);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor(sys, *X);

        typename ElemOpsType::vec_t
        res_e;
        typename ElemOpsType::mat_t
        jac_e;
        
        
        libMesh::MeshBase::const_element_iterator       el     =
        sys.get_mesh().active_local_elements_begin();
        const libMesh::MeshBase::const_element_iterator end_el =
        sys.get_mesh().active_local_elements_end();
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            _e_ops->init(c, sol_accessor);
            
            res_e.setZero(sol_accessor.n_dofs());
            if (J) jac_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            
            // perform the element level calculations
            _e_ops->compute(c, res_e, J?&jac_e:nullptr);
            
            _e_ops->clear();
            
            // copy to the libMesh matrix for further processing
            libMesh::DenseVector<ScalarType> v;
            libMesh::DenseMatrix<ScalarType> m;
            if (R) MAST::Numerics::Utility::copy(v, res_e);
            if (J) MAST::Numerics::Utility::copy(m, jac_e);
            
            // constrain the quantities to account for hanging dofs,
            // Dirichlet constraints, etc.
            if (R && J)
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add
                (*R, *J, sys.get_dof_map(), sol_accessor.dof_indices(), v, m);
            else if (R)
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add
                (*R, sys.get_dof_map(), sol_accessor.dof_indices(), v);
            else
                MAST::Base::Assembly::libMeshWrapper::constrain_and_add
                (*J, sys.get_dof_map(), sol_accessor.dof_indices(), m);
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
