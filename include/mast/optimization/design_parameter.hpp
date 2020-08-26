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
    MAST::Base::ScalarConstant<ScalarType>  (v),
    _id    (-1) {

        _point.setZero();
    }
    
    virtual ~DesignParameter() { }

    inline void set_id(uint_t i) { _id = i;}

    inline uint_t id() const { return _id;}

    inline void set_point(real_t     x,
                          real_t     y = 0.,
                          real_t     z = 0.) {
        
        _point(0) = x;
        _point(1) = y;
        _point(2) = z;
    }
    
    inline const Eigen::Matrix<real_t, 3, 1>& point() const { return _point;}

private:
    
    // ID of the design parameter
    uint_t                        _id;
    
    /// point to which this parameter is attached
    Eigen::Matrix<real_t, 3, 1>  _point;
};
}
}

#endif // __mast_optimization_design_parameter_h__
