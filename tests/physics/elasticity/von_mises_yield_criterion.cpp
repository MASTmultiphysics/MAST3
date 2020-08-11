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
    static const uint_t
    n_strain        = yield_t::n_strain;
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



inline void copy_value(real_t                                    factor,
                       Eigen::Matrix<real_t, Eigen::Dynamic, 1>& from,
                       Eigen::Matrix<real_t, Eigen::Dynamic, 1>& to) {
    
    to = factor * from;
}


inline void copy_value(real_t                                          factor,
                       Eigen::Matrix<adouble_tl_t, Eigen::Dynamic, 1>& from,
                       Eigen::Matrix<adouble_tl_t, Eigen::Dynamic, 1>& to) {
    
    for (uint_t i=0; i<from.size(); i++) to(i) = factor * from(i).getValue();
}



template <typename Traits>
inline void von_mises_yield_criterion_jacobian
(Eigen::Matrix<typename Traits::scalar_t, Traits::n_strain,   1> &strain,
 Eigen::Matrix<typename Traits::scalar_t, Traits::n_strain+1, 1> &x,
 Eigen::Matrix<typename Traits::scalar_t, Traits::n_strain+1, 1> &res,
 Eigen::Matrix<typename Traits::scalar_t, Traits::n_strain+1, Traits::n_strain+1> *jac) {
    
    using scalar_t         = typename Traits::scalar_t;
    
    Context<typename Traits::accessor_t> c;
    Prop<Traits> p;
    
    typename Traits::yield_t
    yield;

    const uint_t
    n_strain = Traits::n_strain,
    n_dofs   = yield.n_variables();
    

    Eigen::Matrix<scalar_t, n_strain, n_strain>
    stiff = Eigen::Matrix<scalar_t, n_strain, n_strain>::Zero();

    Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>
    v0 = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>::Zero(n_dofs),
    v1 = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>::Zero(n_dofs);

    v1.topRows(n_strain) = x.topRows(n_strain);
    v1(n_dofs-1)         = x(n_strain);
    
    copy_value(0.8, v1, v0);
    
    MAST::Physics::Elasticity::ElastoPlasticity::Accessor<scalar_t, typename Traits::yield_t>
    accessor0,
    accessor1;
    
    accessor0.init(yield, v0.data());
    accessor1.init(yield, v1.data());
        
    c.previous_plasticity_accessor = &accessor0;
    c.current_plasticity_accessor  = &accessor1;
    
    yield.set_material(*p.prop);
    yield.set_limit_stress(5.e6);
    //yield.compute(c, strain, accessor, &stiff);
    yield.return_mapping_residual_and_jacobian(c, strain, accessor1, res, jac);
}



inline void
test_von_mises_yield_criterion_jacobian() {
    
    using traits_t         = Traits<real_t, 2>;
    
    const uint_t
    n_strain = traits_t::yield_t::n_strain;

    Eigen::Matrix<real_t, n_strain, 1>
    strain = 1.e-4 * Eigen::Matrix<real_t, n_strain, 1>::Random();
    
    Eigen::Matrix<real_t, n_strain+1, 1>
    x      = Eigen::Matrix<real_t, n_strain+1, 1>::Random(),
    res    = Eigen::Matrix<real_t, n_strain+1, 1>::Zero();
    x.topRows(n_strain) *= 1.e4;
    
    Eigen::Matrix<real_t, n_strain+1, n_strain+1>
    jac    = Eigen::Matrix<real_t, n_strain+1, n_strain+1>::Zero(),
    jac_ad = Eigen::Matrix<real_t, n_strain+1, n_strain+1>::Zero();
    
    von_mises_yield_criterion_jacobian<traits_t>(strain, x, res, &jac);
    
    
    {
        adtl::setNumDir(n_strain+1);
        using traits_ad_t = Traits<adouble_tl_t, 2>;
        
        Eigen::Matrix<adouble_tl_t, n_strain, 1>
        strain_ad = Eigen::Matrix<adouble_tl_t, n_strain, 1>::Zero();
                
        Eigen::Matrix<adouble_tl_t, n_strain+1, 1>
        x_ad   = Eigen::Matrix<adouble_tl_t, n_strain+1, 1>::Zero(),
        res_   = Eigen::Matrix<adouble_tl_t, n_strain+1, 1>::Zero();
        
        Eigen::Matrix<adouble_tl_t, n_strain+1, n_strain+1>
        *jac_ = nullptr;
        
        // the adjoint can be computed in adol-c traceless vector mode
        // with ndof components.
        for (uint_t i=0; i<n_strain; i++) strain_ad(i) = strain(i);
        
        for (uint_t i=0; i<n_strain+1; i++) {
            
            x_ad(i) = x(i);
            x_ad(i).setADValue(i, 1.);
        }
        
        von_mises_yield_criterion_jacobian<traits_ad_t>(strain_ad, x_ad, res_, jac_);

        for (uint_t i=0; i<n_strain+1; i++)
            for (uint_t j=0; j<n_strain+1; j++)
                jac_ad(i,j) = res_(i).getADValue(j);
    }
    
    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(jac),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(jac_ad)));
}


TEST_CASE("von_mises_yield_criterion",
          "[Physics][Elasticity][Elastoplasticity][vonMisesPlasticity]") {
    
    vonMisesYieldCriterion::test_von_mises_yield_criterion_jacobian();
}

} // namespace MAST
} // namespace Test
} // namespace Physics
} // namespace Elasticity
} // namespace vonMisesYieldCriterion
