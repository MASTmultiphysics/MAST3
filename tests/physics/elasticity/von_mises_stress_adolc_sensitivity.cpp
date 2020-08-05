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
namespace AdolC {


template <uint_t   Dim>
inline void test_von_mises_stress_sensitivity()  {
    
    const uint_t
    n_basis  = 20,
    n_strain = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value;
    
    
    Eigen::Matrix<real_t, n_strain, 1>
    stress   = Eigen::Matrix<real_t, n_strain, 1>::Random(),
    dstress  = Eigen::Matrix<real_t, n_strain, 1>::Random(),
    dsvm     = Eigen::Matrix<real_t, n_strain, 1>::Zero(),
    dsvm_ad  = Eigen::Matrix<real_t, n_strain, 1>::Zero();
    

    Eigen::Matrix<real_t, n_strain, n_strain>
    d2stress       = Eigen::Matrix<real_t, n_strain, n_strain>::Zero(),
    d2stress_ad    = Eigen::Matrix<real_t, n_strain, n_strain>::Zero();

    Eigen::Matrix<real_t, n_strain, Eigen::Dynamic>
    stress_adj_mat = Eigen::Matrix<real_t, n_strain, Eigen::Dynamic>::Random(n_strain, n_basis);

    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    vm_adj    = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(n_basis),
    vm_adj_ad = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(n_basis);
    
    real_t
    vm  = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    value(stress),
    dvm = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    derivative_sens(stress, dstress),
    dvm_ad = 0.;

    // stress adjoint
    MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    stress_dX(stress, stress_adj_mat, vm_adj);
    
    // dsigma_vm/dsigma
    MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    derivative(stress, dsvm);

    // d2sigma_vm/dsigma2
    MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<real_t, Dim>::
    second_derivative(stress, d2stress);
    
    {
        // the number of directions for which we compute the sensitivity is = n_strain
        Eigen::Matrix<adouble_tl_t, n_strain, 1>
        stress_ad,
        ds_ad;
        
        for (uint_t i=0; i<n_strain; i++) {
            
            stress_ad(i) = stress(i);
            stress_ad(i).setADValue(&(dstress.data()[i]));
        }
        
        adouble_tl_t
        vm_ad = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<adouble_tl_t, Dim>::
        value(stress_ad);

        dvm_ad = *vm_ad.getADValue();
    }

    
    // first and second derivative of sigma_vm wrt stress vector
    for (uint_t i=0; i<n_strain; i++) {
        
        // the number of directions for which we compute the sensitivity is = n_strain
        Eigen::Matrix<adouble_tl_t, n_strain, 1>
        stress_ad,
        ds_ad;
        
        real_t
        v = 1.;
        
        for (uint_t j=0; j<n_strain; j++) stress_ad(j) = stress(j);
        
        // derivative wrt ith sigma value
        stress_ad(i).setADValue(&v);

        adouble_tl_t
        vm_ad = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<adouble_tl_t, Dim>::
        value(stress_ad);

        MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<adouble_tl_t, Dim>::
        derivative(stress_ad, ds_ad);
        
        dsvm_ad(i) = *vm_ad.getADValue();

        for (uint_t j=0; j<n_strain; j++)
            d2stress_ad(j,i) = *ds_ad(j).getADValue();
    }
    
    
    // adjoint
    {
        adtl::setNumDir(n_basis);

        Eigen::Matrix<adouble_tl_t, n_strain, 1>
        stress_ad;

        // the adjoint can be computed in adol-c traceless vector mode
        // with ndof components.
        for (uint_t i=0; i<n_strain; i++) {

            stress_ad(i) = stress(i);

            for (uint_t j=0; j<n_basis; j++) {
                
                stress_ad(i).setADValue(j, stress_adj_mat(i,j));
            }
        }
        
        adouble_tl_t
        vm_ad = MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<adouble_tl_t, Dim>::
        value(stress_ad);
        
        for (uint_t j=0; j<n_basis; j++)
            vm_adj_ad(j) = vm_ad.getADValue(j);
    }
    
    CHECK(dvm == Catch::Detail::Approx(dvm_ad));

    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(vm_adj),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(vm_adj_ad)));

    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dsvm),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(dsvm_ad)));

    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(d2stress),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(d2stress_ad)));
}



TEST_CASE("von_mises_stress_adolc",
          "[Physics][Elasticity][vonMisesStress][AdolC]") {
    
    test_von_mises_stress_sensitivity<2>();
    
    test_von_mises_stress_sensitivity<3>();
}

} // namespace AdolC
} // namespace vonMisesStress
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


