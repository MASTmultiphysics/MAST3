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

#ifndef __mast_linear_continuum_stress_h__
#define __mast_linear_continuum_stress_h__

// MAST includes
#include <mast/physics/elasticity/linear_elastic_strain_operator.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {
namespace LinearContinuum {


template <typename FEVarType,
          typename MaterialPropertyType,
          uint_t Dim,
          typename ContextType>
class Stress {

public:
    
    using scalar_t          = typename FEVarType::scalar_t;
    using nodal_scalar_t    = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using var_scalar_t      = typename FEVarType::scalar_t;
    static const uint_t
    n_strain                = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value;

    Stress():
    _property      (nullptr),
    _fe_var_data   (nullptr)
    { }


    inline void
    set_section_property(const MaterialPropertyType& p) {
        
        Assert0(!_property, "Property already initialized.");
        
        _property = &p;
    }

    
    inline void set_fe_var_data(const FEVarType &fe_data) {

        Assert0(!_fe_var_data, "FE data already initialized.");
        _fe_var_data = &fe_data;
    }
    
    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return Dim*_fe_var_data->get_fe_shape_data().n_basis();
    }
    

    template <typename AccessorType>
    inline void
    compute(ContextType      &c,
            AccessorType     &stress) const {

        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");

        Assert2(stress.size() == n_strain,
                stress.size(), n_strain,
                "Incorrect stress vector dimension");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        Assert2(c.qp < fe.n_q_points(),
                c.qp, fe.n_q_points(),
                "Invalid quadrature point index");

        typename Eigen::Matrix<scalar_t, n_strain, 1>
        epsilon;

        typename MaterialPropertyType::value_t
        m;

        MAST::Numerics::FEMOperatorMatrix<nodal_scalar_t>
        Bmat;
        Bmat.reinit(n_strain, Dim, _fe_var_data->get_fe_shape_data().n_basis());

        _property->value(c, m);
        
        MAST::Physics::Elasticity::LinearContinuum::strain
        <nodal_scalar_t, var_scalar_t, FEVarType, Dim>
        (*_fe_var_data, c.qp, epsilon, Bmat);
        
        // set value for ith quadrature point
        stress = m * epsilon;
    }
    
    
    template <typename AccessorType, typename ScalarFieldType>
    inline void derivative(ContextType            &c,
                           const ScalarFieldType  &f,
                           AccessorType           &dstress) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");

        Assert2(dstress.size() == n_strain,
                dstress.size(), n_strain,
                "Incorrect stress vector dimension");
                
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        Assert2(c.qp < fe.n_q_points(),
                c.qp, fe.n_q_points(),
                "Invalid quadrature point index");

        typename Eigen::Matrix<scalar_t, n_strain, 1>
        epsilon;

        typename MaterialPropertyType::value_t
        dm;

        MAST::Numerics::FEMOperatorMatrix<nodal_scalar_t>
        Bmat;
        Bmat.reinit(n_strain, Dim, _fe_var_data->get_fe_shape_data().n_basis());

        _property->derivative(c, f, dm);
        
        MAST::Physics::Elasticity::LinearContinuum::strain
        <nodal_scalar_t, var_scalar_t, FEVarType, Dim>
        (*_fe_var_data, c.qp, epsilon, Bmat);
        
        // set value for ith quadrature point
        dstress = dm * epsilon;
    }

    
    template <typename AccessorType>
    inline void adjoint_derivative(ContextType      &c,
                                   AccessorType     &stress_adj) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");

        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();

        Assert2(c.qp < fe.n_q_points(),
                c.qp, fe.n_q_points(),
                "Invalid quadrature point index");

        Assert2(stress_adj.rows() == n_strain,
                stress_adj.rows(), n_strain,
                "Incorrect stress adjoint accessor rows");
        
        Assert2(stress_adj.cols() == 2*fe.n_basis(),
                stress_adj.cols(), 2*fe.n_basis(),
                "Incorrect stress adjoint accessor rows");

        
        typename Eigen::Matrix<scalar_t, n_strain, 1>
        epsilon;

        typename MaterialPropertyType::value_t
        m;

        MAST::Numerics::FEMOperatorMatrix<nodal_scalar_t>
        Bmat;
        Bmat.reinit(n_strain, Dim, _fe_var_data->get_fe_shape_data().n_basis());

        _property->value(c, m);
        
        MAST::Physics::Elasticity::LinearContinuum::strain
        <nodal_scalar_t, var_scalar_t, FEVarType, Dim>
        (*_fe_var_data, c.qp, epsilon, Bmat);
        
        Bmat.left_multiply(stress_adj, m);
    }

    
private:
    
    const MaterialPropertyType                           *_property;
    const FEVarType                                      *_fe_var_data;
    
};
}  // namespace LinearContinuum
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST

#endif // __mast_linear_continuum_stress_h__
