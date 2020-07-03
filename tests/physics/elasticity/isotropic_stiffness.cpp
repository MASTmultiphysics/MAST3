
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace Material {
namespace IsotropicStiffness {

struct Context {};

template <typename ScalarType,
          uint_t   Dim>
struct Traits {

    using scalar_t          = ScalarType;
    using modulus_t         = typename MAST::Base::ScalarConstant<ScalarType>;
    using nu_t              = typename MAST::Base::ScalarConstant<ScalarType>;
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<ScalarType, Dim, modulus_t, nu_t, Context>;
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


template <uint_t Dim>
void test_sensitivity()     {
    
    using traits_t         = Traits<real_t, Dim>;
    using traits_complex_t = Traits<complex_t, Dim>;
    
    Context c;
    Prop<traits_t> p;
    
    typename traits_t::prop_t::value_t
    stiff,
    stiff_cs;
    
    typename traits_complex_t::prop_t::value_t
    stiff_c;
    
    
    // complex-step sensitivity wrt E
    {
        // analytical sensitivity
        p.prop->derivative(c, *p.E, stiff);

        Prop<traits_complex_t> p_c;
        
        (*p_c.E)() += complex_t(0., ComplexStepDelta);
        p_c.prop->value(c, stiff_c);
        stiff_cs = stiff_c.imag()/ComplexStepDelta;
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(stiff_cs)));
    }

    // complex-step sensitivity wrt nu
    {
        // analytical sensitivity
        p.prop->derivative(c, *p.nu, stiff);

        Prop<traits_complex_t> p_c;
        
        (*p_c.nu)() += complex_t(0., ComplexStepDelta);
        p_c.prop->value(c, stiff_c);
        stiff_cs = stiff_c.imag()/ComplexStepDelta;
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(stiff),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(stiff_cs)));
    }
}



TEST_CASE("isotropic_linear_stiffness",
          "[Elasticity][Linear][MaterialProperty]") {

    // test for 2D
    test_sensitivity<2>();
    
    // test for 3D
    test_sensitivity<3>();
}

} // namespace IsotropicStiffness
} // namespace Material
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


