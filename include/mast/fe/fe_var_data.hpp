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

#ifndef __mast_fe_var_data_h__
#define __mast_fe_var_data_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>


namespace MAST {
namespace FEBasis {

template <typename BasisScalarType,
          typename NodalScalarType,
          typename SolScalarType,
          uint_t   NComponents,
          uint_t   Dim,
          typename ContextType,
          typename FEBasisDerivativeType>
class FEVarData {
  
public:
    
    using fe_shape_deriv_t = FEBasisDerivativeType;
    using scalar_t         = typename MAST::DeducedScalarType<NodalScalarType, SolScalarType>::type;
    using sol_vec_view_t   = Eigen::Map<const typename Eigen::Matrix<scalar_t, NComponents, 1>>;
    
    FEVarData():
    _compute_du_dx   (false),
    _fe              (nullptr)
    {}
    virtual ~FEVarData() {}

    template <typename AccessorType>
    inline void init(const ContextType& c,
                     const AccessorType& coeffs) {

        _init_coefficients(c, coeffs);
        _init_variables(c);
    }
    
    inline void clear_coeffs_and_vars() {
        
        _coeff_vec.setZero();
        
        _u.setZero();
        _du_dx.setZero();
    }
    
    inline void set_compute_du_dx(bool f) { _compute_du_dx = f;}
    
    inline void set_fe_shape_data(const FEBasisDerivativeType& fe) {
        
        Assert0(!_fe, "FE pointer already initialized.");
        _fe = &fe;
    }
    
    inline const FEBasisDerivativeType& get_fe_shape_data() const {
        
        Assert0(_fe, "FE pointer not initialized.");
        return *_fe;
    }
    
    inline uint_t n_components() const { return NComponents;}
    
    inline uint_t n_q_points() const {
        
        Assert0(_fe, "FE pointer not initialized");
        return _fe->n_q_points();
    }
    
    inline scalar_t u(uint_t qp, uint_t comp) const  {
        
        Assert0(_coeff_vec.size(), "Object not initialized");
        Assert2(comp <= NComponents,
                comp, NComponents,
                "Invalid component index");

        return _u(comp, qp);
    }
    
    inline scalar_t du_dx(uint_t qp, uint_t comp, uint_t x_i) const
    {
        Assert0(_compute_du_dx, "Object not initialized with du/dx");
        Assert0(_coeff_vec.size(), "Object not initialized");
        Assert2(comp <= NComponents,
                comp, NComponents,
                "Invalid component index");

        return _du_dx(x_i*NComponents+comp, qp);
    }

protected:
    
    template <typename AccessorType>
    inline void _init_coefficients(const ContextType& c,
                                   const AccessorType& coeffs) {
        
        uint_t
        n_coeffs = coeffs.size();

        Assert0(_fe, "FE pointer not initialized");
        Assert2(n_coeffs == _fe->n_basis() * NComponents,
                n_coeffs, _fe->n_basis() * NComponents,
                "Incompatible dimensions of coefficient vector");
        
        _coeff_vec = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>::Zero(n_coeffs);
        
        for (uint_t i=0; i<n_coeffs; i++)
            _coeff_vec(i) = coeffs(i);
    }
    
    
    inline void _init_variables(const ContextType& c) {
        
        Assert0(_fe, "FE pointer not initialized");

        uint_t
        n_qp     = _fe->n_q_points(),
        n_basis  = _fe->n_basis();

        Assert2(_coeff_vec.size() == n_basis*NComponents,
                _coeff_vec.size(), n_basis*NComponents,
                "Coefficients not initialized");
        
        _u = Eigen::Matrix<scalar_t, NComponents, Eigen::Dynamic>::Zero(NComponents, n_qp);
        
        // now, initialize the solution value and derivatives.
        for (uint_t i=0; i<n_qp; i++)
            for (uint_t j=0; j<NComponents; j++)
                for (uint_t k=0; k<n_basis; k++)
                    _u(j, i) += _fe->phi(i, k) * _coeff_vec(j*n_basis+k);
        
        if (_compute_du_dx) {
            
            _du_dx = Eigen::Matrix<scalar_t, NComponents*Dim, Eigen::Dynamic>::Zero(NComponents*Dim, n_qp);
            
            for (uint_t i=0; i<n_qp; i++)
                for (uint_t j=0; j<NComponents; j++)
                    for (uint_t l=0; l<Dim; l++)
                        for (uint_t k=0; k<n_basis; k++)
                            _du_dx(NComponents*l+j, i) += _fe->dphi_dx(i, k, l) * _coeff_vec(j*n_basis+k);
        }
        else
            _du_dx.setZero();
    }
    
    
    template <typename VecType, typename V=SolScalarType>
    inline
    typename std::enable_if<std::is_same<V, complex_t>::value, void>::type
    _add_complex_perturbation(VecType& v, uint_t i) {
        
        Assert2(i <= v.size(), i, v.size(), "Invalid vector index");
        
        v(i) += complex_t(0., ComplexStepDelta);
    }

    bool                               _compute_du_dx;
    const FEBasisDerivativeType       *_fe;
    Eigen::Matrix<scalar_t, NComponents, Eigen::Dynamic>     _u;
    Eigen::Matrix<scalar_t, NComponents*Dim, Eigen::Dynamic> _du_dx;
    Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>               _coeff_vec;
};


} // namespace FEBasis
} // namespace MAST

#endif // __mast_fe_var_data_h__
