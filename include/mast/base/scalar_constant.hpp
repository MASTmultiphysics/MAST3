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

#ifndef __mast_scalar_constant_h__
#define __mast_scalar_constant_h__

namespace MAST {
namespace Base {

template <typename ScalarType>
class ScalarConstant {

public:
    
    using scalar_t = ScalarType;
    
    ScalarConstant(ScalarType v = 0.):
    _v  (v)
    { }
    
    virtual ~ScalarConstant() {}
    
    inline ScalarType& operator= (const ScalarType& v) {
        _v = v;
        return _v;
    }

    inline ScalarType& operator() () {
        return _v;
    }

    inline ScalarType operator() () const {
        return _v;
    }

    template <typename ContextType>
    inline ScalarType value(ContextType& c) const {
        return _v;
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(ContextType& c,
                                 const ScalarFieldType& f) const {
        return &f==this?1.:0.;
    }

private:
    
    ScalarType _v;
};
}
}

#endif // __mast_scalar_constant_h__
