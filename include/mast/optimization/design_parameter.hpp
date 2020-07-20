

#ifndef __mast_optimization_design_parameter_h__
#define __mast_optimization_design_parameter_h__

// MAST includes
#include <mast/base/scalar_constant.hpp>

namespace MAST {
namespace Optimization {

template <typename ScalarType>
class DesignParameter: public MAST::Base::ScalarConstant<ScalarType> {

public:
    
    using scalar_t = ScalarType;
    
    DesignParameter(ScalarType v = 0.):
    MAST::Base::ScalarConstant<ScalarType>  (v) {

        _point.setZero();
    }
    
    virtual ~DesignParameter() { }

    inline void set_point(real_t     x,
                          real_t     y = 0.,
                          real_t     z = 0.) {
        
        _point(0) = x;
        _point(1) = y;
        _point(2) = z;
    }
    
    inline const Eigen::Matrix<real_t, 3, 1>& point() const { return _point;}

private:
    
    /// point to which this parameter is attached
    Eigen::Matrix<real_t, 3, 1>  _point;
};
}
}

#endif // __mast_optimization_design_parameter_h__
