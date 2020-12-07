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

#ifndef __mast_libmesh_output_derivative_h__
#define __mast_libmesh_output_derivative_h__

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

/*!
 * provides a method for global assembly of derivative of output functional with respect to the solution
 * vector. This is typically used for solution of adjoint problems.
 */
template <typename ScalarType,
          typename ElemOpsType>
class OutputDerivative {

public:
    
    static_assert(std::is_same<ScalarType, typename ElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");
    
    OutputDerivative():
    _e_ops        (nullptr)
    { }
    
    virtual ~OutputDerivative() { }
        
    inline void set_elem_ops(ElemOpsType& e_ops) { _e_ops = &e_ops; }

    template <typename VecType, typename ContextType>
    inline void assemble(ContextType            &c,
                         const VecType          &X,
                         VecType                &dqdX) {
                
        MAST::Numerics::Utility::setZero(dqdX);
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, VecType>
        sol_accessor(*c.sys, X);

        using elem_vector_t = typename ElemOpsType::vector_t;
        
        elem_vector_t dqdX_e;
        
        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for the initialization routines
            c.elem = *el;
            
            sol_accessor.init(*c.elem);
            
            dqdX_e.setZero(sol_accessor.n_dofs());
            
            // perform the element level calculations
            _e_ops->derivativeX(c, sol_accessor, dqdX_e);
            
            // constrain the quantities to account for hanging dofs,
            // Dirichlet constraints, etc.
            MAST::Base::Assembly::libMeshWrapper::constrain_and_add_vector
            <ScalarType, VecType, elem_vector_t>
            (dqdX, c.sys->get_dof_map(), sol_accessor.dof_indices(), dqdX_e);
        }

        // parallel matrix/vector require finalization of communication
        MAST::Numerics::Utility::finalize(dqdX);
    }
    
private:

    bool         _finalize_jac;
    ElemOpsType  *_e_ops;
};

} // namespace libMeshWrapper
} // namespace Assembly
} // namespace Base
} // namespace MAST

#endif // __mast_libmesh_output_derivative_h__
