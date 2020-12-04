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

template <typename ScalarType, uint_t Dim>
using stress_vec_t = MAST::Physics::Elasticity::LinearContinuum::stress_vec_t<ScalarType, Dim>;

template <typename ScalarType, uint_t Dim>
using stress_adjoint_mat_t = MAST::Physics::Elasticity::LinearContinuum::stress_adjoint_mat_t<ScalarType, Dim>;



template <uint_t   Dim>
inline void test_von_mises_stress_sensitivity()  {
    
    const uint_t
    n_basis  = 20,
    n_strain = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value;
    
    
    stress_vec_t<real_t, Dim>
    stress   = stress_vec_t<real_t, Dim>::Random(),
    dstress  = stress_vec_t<real_t, Dim>::Random();
    
    stress_adjoint_mat_t<real_t, Dim>
    stress_adj_mat = stress_adjoint_mat_t<real_t, Dim>::Random(n_strain, n_basis);

    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    vm_adj    = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(n_basis),
    vm_adj_cs = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(n_basis);
    
    real_t
    vm  = MAST::Physics::Elasticity::LinearContinuum::vonMises_stress<real_t, Dim>(stress),
    dvm = MAST::Physics::Elasticity::LinearContinuum::vonMises_stress_derivative<real_t, Dim>(stress, dstress),
    dvm_cs = 0.;
    
    MAST::Physics::Elasticity::LinearContinuum::vonMises_stress_dX<real_t, Dim>(stress,
                                                                                stress_adj_mat,
                                                                                vm_adj);
    
    {
        // the number of directions for which we compute the sensitivity is = n_strain
        adtl::setNumDir(1);

        stress_vec_t<adouble_tl_t, Dim>
        stress_ad;
        
        for (uint_t i=0; i<n_strain; i++) {
            
            stress_ad(i) = stress(i);
            stress_ad(i).setADValue(&(dstress.data()[i]));
        }
        
        adouble_tl_t
        vm_ad = MAST::Physics::Elasticity::LinearContinuum::vonMises_stress<adouble_tl_t, Dim>(stress_ad);

        dvm_cs = *vm_ad.getADValue();
    }
    
    // adjoint
    {
        adtl::setNumDir(n_basis);

        stress_vec_t<adouble_tl_t, Dim>
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
        vm_ad = MAST::Physics::Elasticity::LinearContinuum::vonMises_stress<adouble_tl_t, Dim>(stress_ad);
        
        for (uint_t j=0; j<n_basis; j++)
            vm_adj_cs(j) = vm_ad.getADValue(j);
    }
    
    CHECK(dvm == Catch::Detail::Approx(dvm_cs));

    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(vm_adj),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(vm_adj_cs)));
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


