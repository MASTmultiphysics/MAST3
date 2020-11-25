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

// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/solvers/petsc/nonlinear_solver.hpp>

// Test includes
#include <test_helpers.h>

// libMesh includes
#include <libmesh/libmesh.h>
#include <timpi/communicator.h>

extern libMesh::LibMeshInit *p_global_init;

namespace MAST {
namespace Test {
namespace Solvers {
namespace PETSc {

/*!
 * a quadratic n-dimensional function \f$ f(x_1, \ldots, x_n) = \sum_i x_i^2 \f$
 */
class Function {
  
public:

    uint_t n;

    Function(MPI_Comm comm):
    n (10) {
    
        MatCreate(p_global_init->comm().get(), &jac);
        MatSetSizes(jac, PETSC_DECIDE, PETSC_DECIDE, n, n);
        MatSetType(jac, MATSEQDENSE);
        MatSetFromOptions(jac);
        MatSetUp(jac);
    }

    virtual ~Function() {
        MatDestroy(&jac);
    }
    
    inline Mat* matrix() { return &jac;}
    
    inline void residual(Vec x, Vec res) {
        
        real_t
        v = 0.;
        
        VecZeroEntries(res);
        
        for (int_t i=0; i<n; i++) {
            
            VecGetValues(x, 1, &i, &v);
            VecSetValue(res, i, v*v, INSERT_VALUES);
        }
        
        VecAssemblyBegin(res);
        VecAssemblyEnd(res);
    }
    
    inline void jacobian(Vec x, Mat jac) {

        real_t
        v = 0.;
        
        MatZeroEntries(jac);
        
        for (int_t i=0; i<n; i++) {
            
            VecGetValues(x, 1, &i, &v);
            MatSetValue(jac, i, i, 2*v, INSERT_VALUES);
        }
        
        MatAssemblyBegin(jac, MAT_FINAL_ASSEMBLY);
        MatAssemblyEnd(jac, MAT_FINAL_ASSEMBLY);
    }

protected:
    
    Mat    jac;
};




TEST_CASE("petsc_nonlinear_solver",
          "[Algebra][Solvers][Nonlinear][PETSc]") {

    using f_type = MAST::Test::Solvers::PETSc::Function;
    
    f_type f(p_global_init->comm().get());
    
    MAST::Solvers::PETScWrapper::NonlinearSolver<f_type>
    solver(p_global_init->comm().get());
    
    Vec x;
    MatCreateVecs(*f.matrix(), &x, PETSC_NULL);
    VecSetRandom(x, PETSC_NULL);
    
    solver.solve(f, x);

    real_t *vals;
    VecGetArray(x, &vals);

    
    CHECK_THAT(std::vector<real_t>(vals, vals+f.n),
               Catch::Approx(std::vector<real_t>(f.n, 0.)));
}

} // namespace PETSc
} // namespace Solvers
} // namespace Test
} // namespace MAST


