
#ifndef __mast_simp_penalized_density_h__
#define __mast_simp_penalized_density_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Topology {
namespace SIMP {

template <typename ScalarType, typename DensityFieldType>
class PenalizedDensity {
    
public:
    
    PenalizedDensity():
    _p   (0),
    _d   (nullptr)
    { }
    
    virtual ~PenalizedDensity() {}
    
    inline void set_density_field(const DensityFieldType& d) { _d = &d;}
    
    inline void set_penalty(const real_t p) { _p = p;}
    
    template <typename ContextType>
    inline ScalarType value(const ContextType& c) const {
        
        Assert0(_p, "Penalty value not initialized");
        Assert0(_d, "Density field not initialized");
        
        return std::pow(_d->value(c), _p);
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType&     c,
                                 const ScalarFieldType& f) const {
        
        Assert0(_p, "Penalty value not initialized");
        Assert0(_d, "Density field not initialized");
        
        return _p * std::pow(_d->value(c), _p-1.) * _d->derivative(c, f);
    }

    
private:
    
    real_t                   _p;
    const DensityFieldType  *_d;
};
} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace MAST

#endif  // __mast_simp_penalized_density_h__
