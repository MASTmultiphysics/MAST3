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

#ifndef __mast_libmesh_wrapper_elasticity_null_space_h__
#define __mast_libmesh_wrapper_elasticity_null_space_h__

// C++ includes
#include <limits>

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// libMesh includes
#include <libmesh/system.h>
#include <libmesh/dof_map.h>
#include <libmesh/mesh_base.h>
#include <libmesh/petsc_vector.h>
#include <libmesh/node.h>

// PETSc includes
#include <petscmat.h>

namespace MAST {
namespace Physics {
namespace Conduction {
namespace libMeshWrapper {

class NullSpace {
  
public:
    
    NullSpace(libMesh::System& sys, uint_t dim):
    _sys    (sys),
    _dim    (dim) {

        Assert1(_dim > 0, _dim, "Invalid dimension");
        Assert2(_sys.n_vars() == 1,
                _sys.n_vars(), 1,
                "Number of unknowns should be equal to one");

        // make sure all variable types are LAGRANGE
        Assert0(_sys.variable_type(0).family == libMesh::LAGRANGE,
                "Variables are expected to be LAGRANGE");

        // now initialize
        _init();
    }
    
    virtual ~NullSpace() {
        
        PetscErrorCode ierr = 0;
        
        ierr = MatNullSpaceDestroy(&_mns);
        CHKERRABORT(_sys.comm().get(), ierr);
    }
    

    inline MatNullSpace get() {

        Assert0(_mns, "Object not initialized");

        return _mns;
    }
    
    inline void attach_to_matrix(Mat m) {
        
        PetscErrorCode ierr = 0;
        
        ierr = MatSetNearNullSpace(m, _mns);
        CHKERRABORT(_sys.comm().get(), ierr);
    }
    
    
private:
    
    inline void _init() {

        PetscErrorCode
        ierr = 0;
        
        std::unique_ptr<libMesh::NumericVector<real_t>>
        vec(_sys.solution->zero_clone().release());
                        
        libMesh::MeshBase::const_node_iterator
        it   = _sys.get_mesh().local_nodes_begin(),
        end  = _sys.get_mesh().local_nodes_end();
        
        for ( ; it != end; it++)
            // constant temperature
            vec->set((*it)->dof_number(0, 0, 0), 1.);
        
        // close the vector and scale it to unit norm
        vec->close();
        
        real_t
        l2 = vec->l2_norm();
        
        vec->scale(1./l2);
        
        l2 = vec->l2_norm();

        // vector of vectors to be sent to PETSc
        std::vector<Vec> v(1);
        v[0] = dynamic_cast<libMesh::PetscVector<real_t>*>(vec.get())->vec();
        
        ierr = MatNullSpaceCreate(_sys.comm().get(),
                                  PETSC_FALSE,
                                  1,
                                  &v[0],
                                  &_mns);
        CHKERRABORT(_sys.comm().get(), ierr);
    }
    
    
    
    
    libMesh::System    &_sys;
    const uint_t        _dim;
    MatNullSpace        _mns;

};
}  // namespace MAST
}  // namespace Conduction
}  // namespace Physics
}  // namespace libMeshWrapper


#endif // __mast_libmesh_wrapper_elasticity_null_space_h__
