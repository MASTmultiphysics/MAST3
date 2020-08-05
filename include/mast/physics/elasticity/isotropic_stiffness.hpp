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

#ifndef __mast_isotropic_material_stiffness_h__
#define __mast_isotropic_material_stiffness_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {

template <typename ScalarType>
inline ScalarType
shear_modulus(ScalarType E, ScalarType nu) { return E/2./(1.+nu);}

template <typename ScalarType>
inline ScalarType
shear_modulus_derivative(ScalarType  E, ScalarType nu,
                         ScalarType dE, ScalarType dnu)
{ return dE/2./(1.+nu) - E/2./pow(1.+nu,2) * dnu;}


template <typename ScalarType, uint_t Dim, typename ModulusType, typename PoissonType>
class IsotropicMaterialStiffness;

template <typename ScalarType, typename ModulusType, typename PoissonType>
class IsotropicMaterialStiffness<ScalarType, 2, ModulusType, PoissonType> {
public:
    
    static const
    uint_t dim        = 2;
    using E_scalar_t  = typename ModulusType::scalar_t;
    using nu_scalar_t = typename PoissonType::scalar_t;
    using scalar_t    = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<E_scalar_t, nu_scalar_t>::type, ScalarType>::type;
    using value_t     = typename Eigen::Matrix<scalar_t, 3, 3>;
    
    IsotropicMaterialStiffness():
    _E     (nullptr),
    _nu    (nullptr) { }
    
    virtual ~IsotropicMaterialStiffness() {}
    
    inline void set_modulus_and_nu(const ModulusType& E, const PoissonType& nu) {
        
        _E  = &E;
        _nu = &nu;
    }
    
    
    inline const ModulusType& get_E() const {return *_E;}
    
    inline const PoissonType& get_nu() const {return *_nu;}
    
    template <typename ContextType>
    inline scalar_t G(ContextType& c) const {
        
        return shear_modulus<scalar_t>(_E->value(c), _nu->value(c));
    }
    
    template <typename ContextType, typename ScalarFieldType>
    inline scalar_t G_derivative(ContextType& c, const ScalarFieldType& f) const {
        
        return shear_modulus_derivative<scalar_t>(_E->value(c),
                                                  _nu->value(c),
                                                  _E->derivative(c, f),
                                                  _nu->derivative(c, f));
    }
    
    template <typename ContextType>
    inline void value(ContextType& c, value_t& m) const {
                
        Assert0(_E && _nu, "Material values not provided");
        
        const E_scalar_t
        E  = _E->value(c);
        const nu_scalar_t
        nu = _nu->value(c);
        
        m.setZero();
        
        m(0, 0) = m(1, 1) = E/(1.-nu*nu);
        m(0, 1) = m(1, 0) = E*nu/(1.-nu*nu);
        m(2, 2) = shear_modulus(E, nu);
    }
    
    
    template <typename ContextType, typename ScalarFieldType>
    inline void derivative(ContextType&           c,
                           const ScalarFieldType& f,
                           value_t&               m) const {
        
        Assert0(_E && _nu, "Material values not provided");

        const E_scalar_t
        E      = _E->value(c),
        dEdp   = _E->derivative(c, f);

        const nu_scalar_t
        nu     = _nu->value(c),
        dnudp  = _nu->derivative(c, f);

        m.setZero();

        m(0, 0) = m(1, 1) =
        1./(1.-nu*nu) * dEdp + 2. * E/pow(1.-nu*nu,2) * nu * dnudp;
        
        m(0, 1) = m(1, 0) =
        nu/(1.-nu*nu) * dEdp + (E/(1.-nu*nu) + 2. * E*nu/pow(1.-nu*nu,2) * nu) * dnudp;
        
        m(2, 2) = shear_modulus_derivative(E, nu, dEdp, dnudp);
    }

private:
    
    const ModulusType*  _E;
    const PoissonType* _nu;
};



template <typename ScalarType, typename ModulusType, typename PoissonType>
class IsotropicMaterialStiffness<ScalarType, 3, ModulusType, PoissonType> {
public:
    
    static const
    uint_t dim        = 3;
    using E_scalar_t  = typename ModulusType::scalar_t;
    using nu_scalar_t = typename PoissonType::scalar_t;
    using scalar_t    = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<E_scalar_t, nu_scalar_t>::type, ScalarType>::type;
    using value_t     = typename Eigen::Matrix<scalar_t, 6, 6>;
    
    IsotropicMaterialStiffness():
    _E     (nullptr),
    _nu    (nullptr) { }
    
    virtual ~IsotropicMaterialStiffness() {}
    
    inline void set_modulus_and_nu(const ModulusType& E, const PoissonType& nu) {
        
        _E  = &E;
        _nu = &nu;
    }

    template <typename ContextType>
    inline void value(const ContextType& c, value_t& m) const {
                
        Assert0(_E && _nu, "Material values not provided");

        const E_scalar_t
        E  = _E->value(c);
        const nu_scalar_t
        nu = _nu->value(c);
        
        m.setZero();
        
        m(0, 0) = m(1, 1) = m(2, 2) = E*(1.-nu)/(1.-nu-2.*nu*nu);
        m(0, 1) = m(0, 2) = m(1, 0) = m(1, 2) = m(2, 0) = m(2, 1) = E*nu/(1.-nu-2.*nu*nu);
        m(3, 3) = m(4, 4) = m(5, 5) = shear_modulus(E, nu);
    }
    
    
    template <typename ContextType, typename ScalarFieldType>
    inline void derivative(const ContextType&     c,
                           const ScalarFieldType& f,
                           value_t&               m) const {
        
        Assert0(_E && _nu, "Material values not provided");

        const E_scalar_t
        E      = _E->value(c),
        dEdp   = _E->derivative(c, f);

        const nu_scalar_t
        nu     = _nu->value(c),
        dnudp  = _nu->derivative(c, f);

        m.setZero();

        m(0, 0) = m(1, 1) = m(2, 2) =
        (1.-nu)/(1.-nu-2.*nu*nu) * dEdp +
        (-E/(1.-nu-2.*nu*nu) + E*(1.-nu)/pow(1.-nu-2.*nu*nu,2)*(1.+4.*nu)) * dnudp ;
        
        m(0, 1) = m(0, 2) = m(1, 0) = m(1, 2) = m(2, 0) = m(2, 1) =
        nu/(1.-nu-2.*nu*nu) * dEdp +
        (E/(1.-nu-2.*nu*nu) + E*nu/pow(1.-nu-2.*nu*nu,2)*(1.+4.*nu)) * dnudp;
        
        m(3, 3) = m(4, 4) = m(5, 5) = shear_modulus_derivative(E, nu, dEdp, dnudp);
    }

private:
    
    const ModulusType*  _E;
    const PoissonType* _nu;
};


} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_isotropic_material_stiffness_h__
