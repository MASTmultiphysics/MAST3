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
#include <mast/solvers/slepc/hermitian_eigen_solver.hpp>

// Test includes
#include <test_helpers.h>

// libMesh includes
#include <libmesh/libmesh.h>
#include <timpi/communicator.h>

extern libMesh::LibMeshInit *p_global_init;

namespace MAST {
namespace Test {
namespace Solvers {
namespace SLEPc {

void setup_matrices (real_t L, real_t EA, real_t rhoA, Mat *A, Mat *B) {

    // setup matrix for FEA of bar modal analysis
    
    uint_t n = 100; // number of free nodes
    real_t h = L/n;
    

    MatCreate(p_global_init->comm().get(), A);
    MatCreate(p_global_init->comm().get(), B);
    MatSetSizes(*A, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetSizes(*B, PETSC_DECIDE, PETSC_DECIDE, n, n);
    MatSetType(*A, MATSEQDENSE);
    MatSetType(*B, MATSEQDENSE);
    MatSetFromOptions(*A);
    MatSetFromOptions(*B);
    MatSetUp(*A);
    MatSetUp(*B);

    Eigen::Matrix<real_t, 2, 2>
    k_e = Eigen::Matrix<real_t, 2, 2>::Zero(),
    m_e = Eigen::Matrix<real_t, 2, 2>::Zero();
    
    k_e << 1, -1, -1, 1,
    m_e << 2./3., 1./3., 1./3., 2./3.;

    k_e *= EA/h;
    m_e *= rhoA*h/2.;
    
    // 11 elements with constrained ends leaves 10 free nodes.
    for (uint_t i=0; i<=n; i++) {
        for (uint_t j=0; j<2; j++) {
            for (uint_t k=0; k<2; k++) {
                
                // first node of first element is not included
                if (i == 0) {
                    if (j>0  && k>0) {
                        MatSetValue(*A, i+j-1, i+k-1, k_e(j,k), ADD_VALUES);
                        MatSetValue(*B, i+j-1, i+k-1, m_e(j,k), ADD_VALUES);
                    }
                }
                // last node of last element is not included
                else if (i == n) {
                    if (j==0 && k==0) {
                        MatSetValue(*A, i+j-1, i+k-1, k_e(j,k), ADD_VALUES);
                        MatSetValue(*B, i+j-1, i+k-1, m_e(j,k), ADD_VALUES);
                    }
                }
                else {
                    
                    MatSetValue(*A, i+j-1, i+k-1, k_e(j,k), ADD_VALUES);
                    MatSetValue(*B, i+j-1, i+k-1, m_e(j,k), ADD_VALUES);
                }
            }
        }
    }
    
    MatAssemblyBegin(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*A, MAT_FINAL_ASSEMBLY);
    MatAssemblyBegin(*B, MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(*B, MAT_FINAL_ASSEMBLY);
}



TEST_CASE("slepc_hermitian_interface",
          "[Algebra][Solvers][SLEPc]") {
    
    uint_t
    n_ev = 2;
    
    real_t
    L              = 3,
    EA             = 3.,
    rhoA           = 0.5,
    eig            = 0.,
    eig_analytical = 0.,
    pi             = acos(-1.);
    
    Mat A, B, Asens, Bsens;
    
    setup_matrices(L, EA, rhoA,     &A,     &B);
    setup_matrices(L, 1.,   0., &Asens, &Bsens);

    MAST::Solvers::SLEPcWrapper::HermitianEigenSolver eig_solver(EPS_GHEP);
    
    eig_solver.solve(A, &B, n_ev, EPS_SMALLEST_REAL, true);
    
    for (uint_t i=0; i<n_ev; i++) {
        
        // check the eigenvalues
        eig            = eig_solver.eig(i);
        eig_analytical = EA/rhoA* pow((i+1)*pi/L, 2);
        
        CHECK(eig == Catch::Detail::Approx(eig_analytical).margin(1.e-1*eig_analytical));

        // check sensitivity values
        eig            = eig_solver.sensitivity_solve(B, Asens, &Bsens, i);
        eig_analytical = 1./rhoA* pow((i+1)*pi/L, 2);

        CHECK(eig == Catch::Detail::Approx(eig_analytical).margin(1.e-1*eig_analytical));
    }
}

} // namespace SLEPc
} // namespace Solvers
} // namespace Test
} // namespace MAST


