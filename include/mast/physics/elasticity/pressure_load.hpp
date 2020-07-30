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

#ifndef __mast_pressure_load_h__
#define __mast_pressure_load_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {

template <typename FEVarType,
          typename PressureFieldType,
          typename SectionAreaType,
          uint_t Dim,
          typename ContextType>
class SurfacePressureLoad {

public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    SurfacePressureLoad():
    _section       (nullptr),
    _pressure      (nullptr),
    _fe_var_data   (nullptr)
    { }
    
    virtual ~SurfacePressureLoad() { }
    
    inline void set_section_area(const SectionAreaType& s) { _section = &s;}
    
    inline void set_pressure(const PressureFieldType& p) { _pressure = &p;}
    
    inline void set_fe_var_data(const FEVarType& fe) { _fe_var_data = &fe;}

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return Dim*_fe_var_data->get_fe_shape_data().n_basis();
    }

    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_section, "Section property not initialized");
        Assert0(_pressure, "Pressure not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            scalar_t p = _pressure->value(c) * _section->value(c);
            
            for (uint_t j=0; j<Dim; j++) {
                
                // j-th component of normal vector at ith quadrature point
                scalar_t nj = fe.normal(i, j);
                
                if (nj != 0.) {
                    for (uint_t k=0; k<fe.n_basis(); k++)
                        res(j*fe.n_basis() + k) -= fe.detJxW(i) * fe.phi(i, k) * p * nj;
                }
            }
        }
    }
    
    
    template <typename ScalarFieldType>
    inline void derivative(ContextType& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_section, "Section property not initialized");
        Assert0(_pressure, "Pressure not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            scalar_t p = (_pressure->value(c) * _section->derivative(c, f) +
                          _pressure->derivative(c, f) * _section->value(c)) ;
            
            for (uint_t j=0; j<Dim; j++) {
                
                // j-th component of normal vector at ith quadrature point
                scalar_t nj = fe.normal(i, j);
                
                if (nj != 0.) {
                    for (uint_t k=0; k<fe.n_basis(); k++)
                        res(j*fe.n_basis() + k) -= fe.detJxW(i) * fe.phi(i, k) * p * nj;
                }
            }
        }
    }
    
private:

    const SectionAreaType      *_section;
    const PressureFieldType    *_pressure;
    const FEVarType            *_fe_var_data;
};


} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_pressure_load_h__
