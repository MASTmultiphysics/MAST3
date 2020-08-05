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
#include <mast/physics/elasticity/von_mises_yield_criterion.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace vonMisesYieldCriterion {

template <typename AccessorType>
struct Context {
    
    Context():
    current_plasticity_accessor  (nullptr),
    previous_plasticity_accessor (nullptr)
    { }
    
    
    AccessorType *current_plasticity_accessor;
    AccessorType *previous_plasticity_accessor;
};


template <typename ScalarType, uint_t Dim>
struct Traits {

    using scalar_t  = ScalarType;
    using modulus_t = typename MAST::Base::ScalarConstant<ScalarType>;
    using nu_t      = typename MAST::Base::ScalarConstant<ScalarType>;
    using prop_t    = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<ScalarType, Dim, modulus_t, nu_t>;
    using yield_t   = MAST::Physics::Elasticity::ElastoPlasticity::vonMisesYieldFunction<ScalarType, prop_t>;
    using accessor_t= typename MAST::Physics::Elasticity::ElastoPlasticity::Accessor<scalar_t, yield_t>;
};

template <typename Traits>
struct Prop {
    
    Prop():
    E    (new typename Traits::modulus_t(72.e9)),
    nu   (new typename Traits::nu_t(0.33)),
    prop (new typename Traits::prop_t) {
        
        prop->set_modulus_and_nu(*E, *nu);
    }
    
    virtual ~Prop() { }
    
    std::unique_ptr<typename Traits::modulus_t>  E;
    std::unique_ptr<typename Traits::nu_t>       nu;
    std::unique_ptr<typename Traits::prop_t>     prop;
};


inline void
test_von_mises_yield_criterion_jacobian() {
    
    using traits_t         = Traits<real_t, 2>;
    using traits_complex_t = Traits<complex_t, 2>;
    
    Context<traits_t::accessor_t> c;
    Prop<traits_t> p;

    traits_t::yield_t
    yield;
    
    Eigen::Matrix<real_t, traits_t::yield_t::n_strain, 1>
    stress = Eigen::Matrix<real_t, traits_t::yield_t::n_strain, 1>::Zero(),
    strain = Eigen::Matrix<real_t, traits_t::yield_t::n_strain, 1>::Zero();

    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    v = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Zero(2*traits_t::yield_t::n_strain+1);

    MAST::Physics::Elasticity::ElastoPlasticity::Accessor<real_t, traits_t::yield_t>
    accessor;
    accessor.init(yield, v.data());
    
    yield.set_material(*p.prop);
    yield.set_limit_stress(1.e9);
    yield.compute(c, strain, accessor);
}


TEST_CASE("von_mises_yield_criterion",
          "[Physics][Elasticity][vonMisesStress][ComplexStep]") {
    
    test_von_mises_yield_criterion_jacobian();
}

} // namespace MAST
} // namespace Test
} // namespace Physics
} // namespace Elasticity
} // namespace vonMisesYieldCriterion
