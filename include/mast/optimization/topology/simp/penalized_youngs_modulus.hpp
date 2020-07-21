
#ifndef __mast_simp_penalized_youngs_modulus_h__
#define __mast_simp_penalized_youngs_modulus_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {

template <typename ScalarType, typename PenalizedDensityType>
class PenalizedYoungsModulus {
    
public:
    
    using scalar_t = ScalarType;
    
    PenalizedYoungsModulus():
    _E0     (0.),
    _E_min  (0.),
    _d      (nullptr)
    { }
    
    virtual ~PenalizedYoungsModulus() {}
    
    inline void set_density(const PenalizedDensityType &d) { _d = &d;}
    
    inline void set_modulus(const ScalarType E0,
                            const ScalarType Emin) {
        _E0    = E0;
        _E_min = Emin;
    }
    
    template <typename ContextType>
    inline ScalarType value(const ContextType& c) {
        
        Assert0(_E0, "Modulus value not initialized");
        Assert0(_d,  "Density field not initialized");
        
        return _E_min + _E0 * _d->value(c);
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType&     c,
                                 const ScalarFieldType& f) {
        
        Assert0(_E0, "Modulus value not initialized");
        Assert0( _d, "Density field not initialized");
        
        return _E0 * _d->derivative(c, f);
    }

    
private:
    
    ScalarType                   _E0;
    ScalarType                   _E_min;
    const PenalizedDensityType  *_d;
};
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif  // __mast_simp_penalized_youngs_modulus_h__
