
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
    
    // extremely small perturbations do not work for this routine that involves
    // square and square-root operations
    for (uint_t i=0; i<n_strain; i++) {
        
        stress_vec_t<complex_t, Dim>
        stress_c   = stress.template cast<complex_t>();
        
        stress_c(i) += complex_t(0., sqrt(ComplexStepDelta));
        
        // linearized contribution of ith stress component
        dvm_cs += dstress(i)/sqrt(ComplexStepDelta) *
        MAST::Physics::Elasticity::LinearContinuum::vonMises_stress<complex_t, Dim>(stress_c).imag();

        // linearized contribution of ith stress component
        vm_adj_cs += stress_adj_mat.row(i)/sqrt(ComplexStepDelta) *
        MAST::Physics::Elasticity::LinearContinuum::vonMises_stress<complex_t, Dim>(stress_c).imag();
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


