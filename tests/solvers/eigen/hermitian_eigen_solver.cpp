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
#include <mast/base/mast_data_types.h>

// Test includes
#include <test_helpers.h>

// Eigen includes
#include <Eigen/Eigenvalues>


namespace MAST {
namespace Test {
namespace Solvers {
namespace EigenWrapper {

template <typename ScalarType>
void setup_matrices (real_t                                                     L,
                     ScalarType                                                 EA,
                     ScalarType                                                 rhoA,
                     Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> &A,
                     Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic> &B) {

    using matrix_t = Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>;
    
    // setup matrix for FEA of bar modal analysis
    
    uint_t n = 100; // number of free nodes
    real_t h = L/n;

    
    A = matrix_t::Zero(n, n);
    B = matrix_t::Zero(n, n);

    Eigen::Matrix<ScalarType, 2, 2>
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
                        A(i+j-1, i+k-1) += k_e(j,k);
                        B(i+j-1, i+k-1) += m_e(j,k);
                    }
                }
                // last node of last element is not included
                else if (i == n) {
                    if (j==0 && k==0) {
                        A(i+j-1, i+k-1) += k_e(j,k);
                        B(i+j-1, i+k-1) += m_e(j,k);
                    }
                }
                else {
                    
                    A(i+j-1, i+k-1) += k_e(j,k);
                    B(i+j-1, i+k-1) += m_e(j,k);
                }
            }
        }
    }
}



TEST_CASE("eigen_hermitian_eigen_solver",
          "[Algebra][Solvers][Eigen][Sensitivity][ComplexStep][AdolC]") {
    
    uint_t
    n_ev = 2;
    
    real_t
    L              = 3,
    EA             = 3.,
    rhoA           = 0.5,
    eig            = 0.,
    eig_analytical = 0.,
    pi             = acos(-1.);

    
    using vector_t = Eigen::Matrix<real_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    matrix_t
    A,
    B,
    eig_vec,
    Asens,
    Bsens;
    
    setup_matrices<real_t>(L, EA, rhoA,      A,      B);
    setup_matrices<real_t>(L, 1.,   0.,  Asens,  Bsens);

    Eigen::GeneralizedSelfAdjointEigenSolver<matrix_t>
    solver(A, B, Eigen::ComputeEigenvectors|Eigen::Ax_lBx);
    
    for (uint_t i=0; i<n_ev; i++) {
        
        // check the eigenvalues
        eig            = solver.eigenvalues()(i);
        eig_analytical = EA/rhoA* pow((i+1)*pi/L, 2);
        
        CHECK(eig == Catch::Detail::Approx(eig_analytical).margin(1.e-1*eig_analytical));

        // check sensitivity values
        vector_t
        eig_vec = solver.eigenvectors().col(i);
        eig =
        eig_vec.dot( (Asens * eig_vec) - (eig * Bsens * eig_vec) )/ // numerator
        eig_vec.dot(B*eig_vec); // denominator
         
        eig_analytical = 1./rhoA* pow((i+1)*pi/L, 2);

        CHECK(eig == Catch::Detail::Approx(eig_analytical).margin(1.e-1*eig_analytical));
    }
}

} // namespace EigenWrapper
} // namespace Solvers
} // namespace Test
} // namespace MAST


