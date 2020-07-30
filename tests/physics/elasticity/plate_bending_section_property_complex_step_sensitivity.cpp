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
#include <mast/physics/elasticity/plate_bending_section_property.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace Material {
namespace PlateBendingSectionProperty {

struct Context {};

template <typename ScalarType>
struct Traits {

    using scalar_t          = ScalarType;
    using modulus_t         = typename MAST::Base::ScalarConstant<ScalarType>;
    using nu_t              = typename MAST::Base::ScalarConstant<ScalarType>;
    using thickness_t       = typename MAST::Base::ScalarConstant<ScalarType>;
    using material_t        = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<ScalarType, 2, modulus_t, nu_t, Context>;
    using section_t         = typename MAST::Physics::Elasticity::PlateBendingSectionProperty<ScalarType, material_t, thickness_t, Context>;

};


template <typename Traits>
struct Prop {
    
    Prop():
    E        (new typename Traits::modulus_t(72.e9)),
    nu       (new typename Traits::nu_t(0.33)),
    th       (new typename Traits::thickness_t(0.003)),
    material (new typename Traits::material_t),
    section  (new typename Traits::section_t) {
        
        material->set_modulus_and_nu(*E, *nu);
        section->set_material_and_thickness(*material, *th);
    }
    
    virtual ~Prop() { }
    
    std::unique_ptr<typename Traits::modulus_t>    E;
    std::unique_ptr<typename Traits::nu_t>         nu;
    std::unique_ptr<typename Traits::thickness_t>  th;
    std::unique_ptr<typename Traits::material_t>   material;
    std::unique_ptr<typename Traits::section_t>    section;
};


void test_sensitivity()     {
    
    using traits_t         = Traits<real_t>;
    using traits_complex_t = Traits<complex_t>;
    
    Context c;
    Prop<traits_t> p;
    
    typename traits_t::section_t::inplane_value_t
    inplane_stiff,
    inplane_stiff_cs;
    
    typename traits_complex_t::section_t::inplane_value_t
    inplane_stiff_c;

    typename traits_t::section_t::shear_value_t
    shear_stiff,
    shear_stiff_cs;
    
    typename traits_complex_t::section_t::shear_value_t
    shear_stiff_c;

    
    // complex-step sensitivity wrt E
    {
        // analytical sensitivity
        p.section->inplane_derivative(c, *p.E, inplane_stiff);
        p.section->shear_derivative(c, *p.E, shear_stiff);

        Prop<traits_complex_t> p_c;
        
        (*p_c.E)() += complex_t(0., ComplexStepDelta);
        p_c.section->inplane_value(c, inplane_stiff_c);
        p_c.section->shear_value(c, shear_stiff_c);
        inplane_stiff_cs = inplane_stiff_c.imag()/ComplexStepDelta;
        shear_stiff_cs   = shear_stiff_c.imag()/ComplexStepDelta;

        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(inplane_stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(inplane_stiff_cs)));

        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(shear_stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(shear_stiff_cs)));
    }

    // complex-step sensitivity wrt nu
    {
        // analytical sensitivity
        p.section->inplane_derivative(c, *p.nu, inplane_stiff);
        p.section->shear_derivative(c, *p.nu, shear_stiff);

        Prop<traits_complex_t> p_c;
        
        (*p_c.nu)() += complex_t(0., ComplexStepDelta);
        p_c.section->inplane_value(c, inplane_stiff_c);
        p_c.section->shear_value(c, shear_stiff_c);
        inplane_stiff_cs = inplane_stiff_c.imag()/ComplexStepDelta;
        shear_stiff_cs   = shear_stiff_c.imag()/ComplexStepDelta;

        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(inplane_stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(inplane_stiff_cs)));
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(shear_stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(shear_stiff_cs)));
    }

    // complex-step sensitivity wrt th
    {
        // analytical sensitivity
        p.section->inplane_derivative(c, *p.th, inplane_stiff);
        p.section->shear_derivative(c, *p.th, shear_stiff);

        Prop<traits_complex_t> p_c;
        
        (*p_c.th)() += complex_t(0., ComplexStepDelta);
        p_c.section->inplane_value(c, inplane_stiff_c);
        p_c.section->shear_value(c, shear_stiff_c);
        inplane_stiff_cs = inplane_stiff_c.imag()/ComplexStepDelta;
        shear_stiff_cs   = shear_stiff_c.imag()/ComplexStepDelta;

        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(inplane_stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(inplane_stiff_cs)));
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(shear_stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(shear_stiff_cs)));
    }
}



TEST_CASE("plate_bending_section_complex_step",
          "[Elasticity][Linear][SectionProperty][Bending]") {

    test_sensitivity();
}

} // namespace PlateBendingSectionProperty
} // namespace Material
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


