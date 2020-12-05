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

#ifndef __mast_libmesh_eigenproblem_assembly_h__
#define __mast_libmesh_eigenproblem_assembly_h__

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
class EigenProblemAssembly {

public:
    
    static_assert(std::is_same<ScalarType, typename ElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");
    
    EigenProblemAssembly():
    _finalize_jac (true),
    _e_ops        (nullptr)
    { }
    
    virtual ~EigenProblemAssembly() { }
    
    inline void set_finalize_jac(bool f) { _finalize_jac = f;}
    
    inline void set_elem_ops(ElemOpsType& e_ops) { _e_ops = &e_ops; }

    template <typename VecType, typename MatType, typename ContextType>
    inline void assemble(ContextType   &c,
                         const VecType &X,
                         MatType       &A,
                         MatType       &B) {
                
        MAST::Numerics::Utility::setZero(A);
        MAST::Numerics::Utility::setZero(B);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor(*c.sys, X);

        using elem_vector_t = typename ElemOpsType::vector_t;
        using elem_matrix_t = typename ElemOpsType::matrix_t;
        
        elem_matrix_t
        A_e,
        B_e;
        
        
        
        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            
            A_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            B_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            
            // perform the element level calculations
            _e_ops->compute(c, sol_accessor, A_e, B_e);
                        
            // constrain the quantities to account for hanging dofs,
            // Dirichlet constraints, etc.
            MAST::Base::Assembly::libMeshWrapper::constrain_and_add_matrix
            <ScalarType, MatType, elem_matrix_t>
            (A, c.sys->get_dof_map(), sol_accessor.dof_indices(), A_e);
            
            MAST::Base::Assembly::libMeshWrapper::constrain_and_add_matrix
            <ScalarType, MatType, elem_matrix_t>
            (B, c.sys->get_dof_map(), sol_accessor.dof_indices(), B_e);
        }

        // parallel matrix require finalization of communication
        if (_finalize_jac) {
            
            MAST::Numerics::Utility::finalize(A);
            MAST::Numerics::Utility::finalize(B);
        }
    }

    
    template <typename VecType,
              typename MatType,
              typename ContextType,
              typename ScalarFieldType>
    inline void sensitivity_assemble(ContextType            &c,
                                     const ScalarFieldType  &f,
                                     const VecType          &X,
                                     MatType                &A,
                                     MatType                &B) {
                
        MAST::Numerics::Utility::setZero(A);
        MAST::Numerics::Utility::setZero(B);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor(*c.sys, X);

        using elem_vector_t = typename ElemOpsType::vector_t;
        using elem_matrix_t = typename ElemOpsType::matrix_t;
        
        elem_matrix_t
        A_e,
        B_e;
        
        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            
            A_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            B_e.setZero(sol_accessor.n_dofs(), sol_accessor.n_dofs());
            
            // perform the element level calculations
            _e_ops->derivative(c, f, sol_accessor, A_e, B_e);
                        
            // constrain the quantities to account for hanging dofs,
            // Dirichlet constraints, etc.
            MAST::Base::Assembly::libMeshWrapper::constrain_and_add_matrix
            <ScalarType, MatType, elem_matrix_t>
            (A, c.sys->get_dof_map(), sol_accessor.dof_indices(), A_e);
            
            MAST::Base::Assembly::libMeshWrapper::constrain_and_add_matrix
            <ScalarType, MatType, elem_matrix_t>
            (B, c.sys->get_dof_map(), sol_accessor.dof_indices(), B_e);
        }

        // parallel matrix require finalization of communication
        if (_finalize_jac) {
            
            MAST::Numerics::Utility::finalize(A);
            MAST::Numerics::Utility::finalize(B);
        }
    }

private:

    bool         _finalize_jac;
    ElemOpsType  *_e_ops;
};

} // namespace libMeshWrapper
} // namespace Assembly
} // namespace Base
} // namespace MAST

#endif // __mast_libmesh_eigenproblem_assembly_h__
