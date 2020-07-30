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

#ifndef __mast_scalar_field_wrapper_h__
#define __mast_scalar_field_wrapper_h__


namespace MAST {
namespace FEBasis {

template <typename ScalarType, typename FEVarType>
class ScalarFieldWrapper {

public:
    
    ScalarFieldWrapper():
    _fe              (nullptr),
    _fe_derivative   (nullptr),
    _comp            (0),
    _comp_derivative (0)
    {}
    
    virtual ~ScalarFieldWrapper() { }
    
    inline void
    set_fe_object_and_component(const FEVarType& fe, uint_t comp) {
        
        _fe   = &fe;
        _comp = comp;
    }

    inline void
    set_derivative_fe_object_and_component(const FEVarType& fe,
                                           uint_t comp) {
        
        _fe_derivative   = &fe;
        _comp_derivative = comp;
    }

    template <typename ContextType>
    inline ScalarType value(const ContextType& c) const {
        
        Assert0(_fe, "Object not initialized");
        return _fe->u(c.qp, _comp);
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(const ContextType     &c,
                                 const ScalarFieldType &f) const {
        
        Assert0(_fe_derivative, "Object not initialized");
        return _fe_derivative->u(c.qp, _comp_derivative);
    }

    
private:
    
    const FEVarType *   _fe;
    const FEVarType *   _fe_derivative;

    uint_t              _comp;
    uint_t              _comp_derivative;

};

} // namespace FEBasis
} // namespace MAST

#endif  // __mast_scalar_field_wrapper_h__
