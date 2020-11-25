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

#ifndef __mast_petsc_linear_solver_h__
#define __mast_petsc_linear_solver_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// PETSc includes
#include <petsc.h>


namespace MAST {
namespace Solvers {
namespace PETScWrapper {


class LinearSolver {
  
public:
    
    LinearSolver(const MPI_Comm comm):
    _comm  (comm),
    _A     (nullptr) {
        
    }
    
    
    virtual ~LinearSolver() {
        
        if (_ksp) {
            
            PetscErrorCode
            ierr = KSPDestroy(&_ksp);
            CHKERRABORT(_comm, ierr);
        }
    }
    
    /*!
     * initialize the solver for operator matrix \p A. This creates the \p KSP object
     * using the command line options. If \p scope is provided then the solver will
     * pass this to the \p KSPSetOptionsPrefix method. This allows specific
     * selection of solver options for different linear solvers in a code.
     */
    inline void init(Mat A, const std::string* scope = nullptr) {

        Assert0(!_A, "solver already initialized");

        _A = &A;
        
        PC pc;
        
        // setup the KSP
        PetscErrorCode
        ierr = KSPCreate(_comm, &_ksp);
        CHKERRABORT(_comm, ierr);

        if (scope) {
            
            std::string nm = *scope + "_";
            ierr = KSPSetOptionsPrefix(_ksp, nm.c_str());
            CHKERRABORT(_comm, ierr);
        }

        ierr = KSPSetOperators(_ksp, *_A, *_A);
        CHKERRABORT(_comm, ierr);
        
        ierr = KSPSetFromOptions(_ksp);
        CHKERRABORT(_comm, ierr);
        
        // setup the PC
        ierr = KSPGetPC(_ksp, &pc);
        CHKERRABORT(_comm, ierr);
        
        ierr = PCSetFromOptions(pc);
        CHKERRABORT(_comm, ierr);
    }
    
    
    /*!
     * Solves \f$ A x = b \f$, where \f$ A \f$ is the system matrix. associated with this
     * solver
     */
    inline void solve(Vec x, Vec b) {
        
        Assert0(_A, "solver not initialized");
        
        PetscErrorCode
        ierr = 0;
        
        // now solve
        ierr = KSPSolve(_ksp, b, x);
        CHKERRABORT(_comm, ierr);
    }
    

    KSP ksp() {
        
        return _ksp;
    }
    
    
private:

    const MPI_Comm   _comm;
    
    KSP  _ksp;
    Mat *_A;
};

}
}
}

#endif // __mast_petsc_linear_solver_h__
