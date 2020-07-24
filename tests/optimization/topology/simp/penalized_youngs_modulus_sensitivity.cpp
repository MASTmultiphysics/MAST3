
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/optimization/topology/simp/penalized_density.hpp>
#include <mast/optimization/topology/simp/penalized_youngs_modulus.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Optimization {
namespace Topology {
namespace SIMP {

struct Context {};

template <typename ScalarType>
struct DensityField {
    
    DensityField(): v(0.5), dv(0.75) {}
    
    inline ScalarType
    value(const Context& c) const { return v;}
    
    template <typename ScalarFieldType>
    inline ScalarType
    derivative(const Context& c, const ScalarFieldType& f) const
    {
        return dv;
    }
    
    ScalarType v;
    real_t     dv;
};

inline void test_penalized_youngs_modulus_sensitivity()  {

    using density_r_t  = MAST::Optimization::Topology::SIMP::PenalizedDensity<real_t, DensityField<real_t>>;
    using density_cs_t = MAST::Optimization::Topology::SIMP::PenalizedDensity<complex_t, DensityField<complex_t>>;
    using density_ad_t = MAST::Optimization::Topology::SIMP::PenalizedDensity<adouble_tl_t, DensityField<adouble_tl_t>>;

    using youngs_mod_r_t  = MAST::Optimization::Topology::SIMP::PenalizedYoungsModulus<real_t, density_r_t>;
    using youngs_mod_cs_t = MAST::Optimization::Topology::SIMP::PenalizedYoungsModulus<complex_t, density_cs_t>;
    using youngs_mod_ad_t = MAST::Optimization::Topology::SIMP::PenalizedYoungsModulus<adouble_tl_t, density_ad_t>;

    
    Context c;
    DensityField<real_t> field;

    density_r_t density;
    density.set_density_field(field);
    density.set_penalty(3.);
    
    youngs_mod_r_t modulus;
    modulus.set_density(density);
    modulus.set_modulus(72.e9, 72.e2);
    
    real_t
    E       = modulus.value(c),
    dE      = modulus.derivative(c, field),
    dE_cs   = 0.,
    dE_ad   = 0.;
    
    // complex step sensitivity
    {
        DensityField<complex_t> field_cs;
        field_cs.v += complex_t(0., ComplexStepDelta);
        
        density_cs_t density_cs;
        density_cs.set_density_field(field_cs);
        density_cs.set_penalty(3.);
        
        youngs_mod_cs_t modulus_cs;
        modulus_cs.set_density(density_cs);
        modulus_cs.set_modulus(72.e9, 72.e2);

        dE_cs = field_cs.dv * modulus_cs.value(c).imag()/ComplexStepDelta;
    }


    // automatic differentiation
    {
        DensityField<adouble_tl_t> field_ad;
        field_ad.v.setADValue(&field_ad.dv);
        
        density_ad_t density_ad;
        density_ad.set_density_field(field_ad);
        density_ad.set_penalty(3.);
        
        youngs_mod_ad_t modulus_ad;
        modulus_ad.set_density(density_ad);
        modulus_ad.set_modulus(72.e9, 72.e2);

        dE_ad = *modulus_ad.value(c).getADValue();
    }
    
    CHECK(dE == Catch::Detail::Approx(dE_cs));
    CHECK(dE == Catch::Detail::Approx(dE_ad));
}



TEST_CASE("penalized_youngs_modulus_sensitivity",
          "[Optimization][Topology][SIMP][ComplexStep][AdolC]") {
    
    test_penalized_youngs_modulus_sensitivity();
}

} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace Test
} // namespace MAST


