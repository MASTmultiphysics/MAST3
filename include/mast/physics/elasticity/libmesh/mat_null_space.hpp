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
namespace Elasticity {
namespace libMeshWrapper {

class NullSpace {
  
public:
    
    NullSpace(libMesh::System& sys, uint_t dim):
    _sys    (sys),
    _dim    (dim) {

        Assert1(_dim > 0, _dim, "Invalid dimension");
        Assert2(_sys.n_vars() == _dim,
                _sys.n_vars(), _dim,
                "Number of unknowns should be equal to spatial dimension");

        // make sure all variable types are LAGRANGE
        for (uint_t i=0; i<_dim; i++)
            Assert0(_sys.variable_type(i).family == libMesh::LAGRANGE,
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
        
        uint_t
        n_translation = _dim,
        n_rotation    = 0;
        
        if (_dim == 2)
            // an inplane problem has one single rotation about the z-axis
            n_rotation = 1;
        else if (_dim == 3)
            // a three-dimensional problem can rotate about all three axes.
            n_rotation = 3;
        
        
        std::vector<libMesh::NumericVector<real_t>*>
        vecs(n_translation+n_rotation, nullptr);
        
        std::vector<libMesh::dof_id_type>
        dof_ids(_dim, 0);
        
        // create and initialize the vectors
        for (uint_t i=0; i<vecs.size(); i++)
            vecs[i] = _sys.solution->zero_clone().release();
                
        libMesh::MeshBase::const_node_iterator
        it   = _sys.get_mesh().local_nodes_begin(),
        end  = _sys.get_mesh().local_nodes_end();
        
        for ( ; it != end; it++) {
            
            const libMesh::Node& n = **it;

            // assuming that the dofs are sequentially numbered
            dof_ids.clear();
            _sys.get_dof_map().dof_indices(&n, dof_ids);
            
            // make sure that the correct number of dofs is found
            Assert2(dof_ids.size() == n_translation,
                    dof_ids.size(), n_translation,
                    "Invalid number of dofs on node");
            
            // rigid-body translation
            for (uint_t i=0; i<n_translation; i++)
                vecs[i]->set(dof_ids[i], 1.);
            
            // rigid-body rotation, assumed to be about point (0,0,0)
            for (uint_t i=0; i<n_rotation; i++) {
                
                libMesh::NumericVector<real_t>
                &v = *vecs[n_translation + i];
                
                switch (i) {

                        // first rotation about z-axis since this is the only one
                        // necessary for 2D elasticity
                    case 0: {
                        
                        real_t th = 1.e-5;
                        // set deformation assuming rotation about (x,y,z)=0
                        // theta-z is only going to lead to changes in x,y
                        v.set(dof_ids[0],
                              (cos(th) * n(0) - sin(th) * n(1)) - n(0));
                        v.set(dof_ids[1],
                              (sin(th) * n(0) + cos(th) * n(1)) - n(1));
                    }
                        break;
                        
                        // next rotation about x-axis
                    case 1: {
                        
                        real_t th = 1.e-5;
                            // also set deformation assuming rotation about (x,y,z)=0
                            // theta-x is only going to lead to changes in y,z
                        v.set(dof_ids[1],
                              (cos(th) * n(1) - sin(th) * n(2)) - n(1));
                        v.set(dof_ids[2],
                              (sin(th) * n(1) + cos(th) * n(2)) - n(2));
                    }
                        break;
                        
                        // finally, rotation about y-axis
                    case 2: {
                        
                        real_t th = 1.e-5;
                        // also set deformation assuming rotation about (x,y,z)=0
                        // theta-y is only going to lead to changes in x,z
                        v.set(dof_ids[0],
                              (cos(th) * n(0) + sin(th) * n(2)) - n(0));
                        v.set(dof_ids[2],
                              (-sin(th) * n(0) + cos(th) * n(2)) - n(2));
                    }
                        
                    default:
                        Error(false, "Invalid rotation vector index");
                }
            }
        }
        
        // close the vectors and scale them to unit norm
        for (uint_t i=0; i<vecs.size(); i++) {
            
            vecs[i]->close();
            
            // the vectors should form an orthonormal basis
            // We use the Gram-Schmidt orthogonalization
            for (uint_t j=0; j<i; j++) {
                
                real_t
                v = vecs[i]->dot(*vecs[j]);
                
                vecs[i]->add(-v, *vecs[j]);
            }

            real_t
            l2 = vecs[i]->l2_norm();
            
            vecs[i]->scale(1./l2);

            l2 = vecs[i]->l2_norm();

            // the vectors should now be orthonormal basis
            for (uint_t j=0; j<i; j++) {
                
                // use the Gram-Schmidt orthogonalization
                real_t
                v = vecs[i]->dot(*vecs[j]);

                Assert2(fabs(v) < std::sqrt(std::numeric_limits<real_t>::epsilon()),
                        fabs(v), std::sqrt(std::numeric_limits<real_t>::epsilon()),
                        "Vector should be orthogonal");
            }
        }

        // vector of vectors to be sent to PETSc
        std::vector<Vec> v(vecs.size());
        for (uint_t i=0; i<vecs.size(); i++)
            v[i] = dynamic_cast<libMesh::PetscVector<real_t>*>(vecs[i])->vec();
        
        ierr = MatNullSpaceCreate(_sys.comm().get(),
                                  PETSC_FALSE,
                                  vecs.size(),
                                  &v[0],
                                  &_mns);
        CHKERRABORT(_sys.comm().get(), ierr);
        
        // now delete the vectors
        for (uint_t i=0; i<vecs.size(); i++)
            delete vecs[i];
    }
    
    
    
    
    libMesh::System    &_sys;
    const uint_t        _dim;
    MatNullSpace        _mns;

};
}  // namespace MAST
}  // namespace Physics
}  // namespace Elasticity
}  // namespace libMeshWrapper


#endif // __mast_libmesh_wrapper_elasticity_null_space_h__
