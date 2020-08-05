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
#include <mast/physics/elasticity/von_mises_stress.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace vonMisesStress {
namespace ComplexStep {


template <uint_t   Dim>
inline void test_von_mises_stress_sensitivity()  {
    
    const uint_t
    n_basis  = 20,
    n_strain = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value;
    
    
    Eigen::Matrix<real_t, n_strain, 1>
    stress   = Eigen::Matrix<real_t, n_strain, 1>::Random(),
    dstress  = Eigen::Matrix<real_t, n_strain, 1>::Random();
    
    Eigen::Matrix<real_t, n_strain, Eigen::Dynamic>
    stress_adj_mat = Eigen::Matrix<real_t, n_strain, Eigen::Dynamic>::Random(n_strain, n_basis);

    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    vm_adj    = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(n_basis),
    vm_adj_cs = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(n_basis);
    
    real_t
    vm  = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    value(stress),
    dvm = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    derivative_sens(stress, dstress),
    dvm_cs = 0.;
    
    MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    stress_dX(stress, stress_adj_mat, vm_adj);
    
    // extremely small perturbations do not work for this routine that involves
    // square and square-root operations
    for (uint_t i=0; i<n_strain; i++) {
        
        Eigen::Matrix<complex_t, n_strain, 1>
        stress_c   = stress.template cast<complex_t>();
        
        stress_c(i) += complex_t(0., sqrt(ComplexStepDelta));
        
        // linearized contribution of ith stress component
        dvm_cs += dstress(i)/sqrt(ComplexStepDelta) *
        MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<complex_t, Dim>::value(stress_c).imag();

        // linearized contribution of ith stress component
        vm_adj_cs += stress_adj_mat.row(i)/sqrt(ComplexStepDelta) *
        MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<complex_t, Dim>::value(stress_c).imag();
    }
    
    CHECK(dvm == Catch::Detail::Approx(dvm_cs));

    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(vm_adj),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(vm_adj_cs)));
}



TEST_CASE("von_mises_stress_complex_step",
          "[Physics][Elasticity][vonMisesStress][ComplexStep]") {
    
    test_von_mises_stress_sensitivity<2>();
    
    test_von_mises_stress_sensitivity<3>();
}

} // namespace ComplexStep
} // namespace vonMisesStress
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


