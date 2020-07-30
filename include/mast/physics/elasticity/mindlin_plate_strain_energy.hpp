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

#ifndef __mast_linear_mindlin_plate_strain_energy_h__
#define __mast_linear_mindlin_plate_strain_energy_h__

// MAST includes
#include <mast/physics/elasticity/mindlin_strain_operator.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {
namespace MindlinPlate {


template <typename FEVarType,
          typename SectionPropertyType,
          typename ContextType>
class StrainEnergy {
    
public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using fe_shape_deriv_t = typename FEVarType::fe_shape_deriv_t;

    StrainEnergy():
    _property            (nullptr),
    _bending_fe_var_data (nullptr),
    _shear_fe_var_data   (nullptr)
    { }
    
    virtual ~StrainEnergy() { }

    inline void
    set_section_property(const SectionPropertyType& p) {
        
        Assert0(!_property, "Property already initialized.");
        
        _property = &p;
    }

    inline void set_fe_var_data(const FEVarType& bending_fe_data,
                                const FEVarType& shear_fe_data) {
        
        Assert0(!_bending_fe_var_data && !_shear_fe_var_data,
                "FE data already initialized.");
        Assert2(bending_fe_data.n_components() == shear_fe_data.n_components(),
                bending_fe_data.n_components(), shear_fe_data.n_components(),
                "Bending and shear FE data must have same number of components");
        
        _bending_fe_var_data = &bending_fe_data;
        _shear_fe_var_data   = &shear_fe_data;
    }

    inline uint_t n_dofs() const {

        Assert0(_bending_fe_var_data, "FE data not initialized.");

        return 3*_bending_fe_var_data->get_fe_shape_data().n_basis();
    }
    
    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_bending_fe_var_data && _shear_fe_var_data,
                "FE data not initialized.");
        Assert0(_property, "Section property not initialized");
                
        // process the inplane strain components
        {
            const typename FEVarType::fe_shape_deriv_t
            &fe = _bending_fe_var_data->get_fe_shape_data();

            typename Eigen::Matrix<scalar_t, 3, 1>
            epsilon,
            stress;
            vector_t
            vec     = vector_t::Zero(3*fe.n_basis());
            
            typename SectionPropertyType::inplane_value_t
            mat;
            
            matrix_t
            mat1 = matrix_t::Zero(3, 3*fe.n_basis()),
            mat2 = matrix_t::Zero(3*fe.n_basis(), 3*fe.n_basis());
            
            MAST::Numerics::FEMOperatorMatrix<scalar_t>
            Bxmat;
            Bxmat.reinit(3, 3, fe.n_basis());
            
            
            for (uint_t i=0; i<fe.n_q_points(); i++) {
                
                c.qp = i;
                
                _property->inplane_value(c, mat);
                MAST::Physics::Elasticity::MindlinPlate::inplane_strain
                <scalar_t, scalar_t, FEVarType>
                (*_bending_fe_var_data, i, 1., epsilon, Bxmat);
                stress = mat * epsilon;
                Bxmat.vector_mult_transpose(vec, stress);
                res += fe.detJxW(i) * vec;
                
                if (jac) {
                    
                    Bxmat.left_multiply(mat1, mat);
                    Bxmat.right_multiply_transpose(mat2, mat1);
                    (*jac) += fe.detJxW(i) * mat2;
                }
            }
        }
        
        // process the transverse shear strain components
        {
            const typename FEVarType::fe_shape_deriv_t
            &fe = _shear_fe_var_data->get_fe_shape_data();

            typename Eigen::Matrix<scalar_t, 2, 1>
            epsilon,
            stress;
            vector_t
            vec     = vector_t::Zero(3*fe.n_basis());
            
            typename SectionPropertyType::shear_value_t
            mat;
            
            matrix_t
            mat1 = matrix_t::Zero(2, 3*fe.n_basis()),
            mat2 = matrix_t::Zero(3*fe.n_basis(), 3*fe.n_basis());
            
            MAST::Numerics::FEMOperatorMatrix<scalar_t>
            Bxmat;
            Bxmat.reinit(2, 3, fe.n_basis());
            
            
            for (uint_t i=0; i<fe.n_q_points(); i++) {
                
                c.qp = i;
                
                _property->shear_value(c, mat);
                MAST::Physics::Elasticity::MindlinPlate::transverse_shear_strain
                <scalar_t, scalar_t, FEVarType>
                (*_shear_fe_var_data, i, epsilon, Bxmat);
                stress = mat * epsilon;
                Bxmat.vector_mult_transpose(vec, stress);
                res += fe.detJxW(i) * vec;
                
                if (jac) {
                    
                    Bxmat.left_multiply(mat1, mat);
                    Bxmat.right_multiply_transpose(mat2, mat1);
                    (*jac) += fe.detJxW(i) * mat2;
                }
            }
        }
    }

    template <typename ScalarFieldType>
    inline void derivative(ContextType& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac = nullptr) const {
        
        Assert0(_bending_fe_var_data && _shear_fe_var_data,
                "FE data not initialized.");
        Assert0(_property, "Section property not initialized");
                
        // process the inplane strain components
        {
            const typename FEVarType::fe_shape_deriv_t
            &fe = _bending_fe_var_data->get_fe_shape_data();

            typename Eigen::Matrix<scalar_t, 3, 1>
            epsilon,
            stress;
            vector_t
            vec     = vector_t::Zero(3*fe.n_basis());
            
            typename SectionPropertyType::inplane_value_t
            mat;
            
            matrix_t
            mat1 = matrix_t::Zero(3, 3*fe.n_basis()),
            mat2 = matrix_t::Zero(3*fe.n_basis(), 3*fe.n_basis());
            
            MAST::Numerics::FEMOperatorMatrix<scalar_t>
            Bxmat;
            Bxmat.reinit(3, 3, fe.n_basis());
            
            
            for (uint_t i=0; i<fe.n_q_points(); i++) {
                
                c.qp = i;
                
                _property->inplane_derivative(c, f, mat);
                MAST::Physics::Elasticity::MindlinPlate::inplane_strain
                <scalar_t, scalar_t, FEVarType>
                (*_bending_fe_var_data, i, 1., epsilon, Bxmat);
                stress = mat * epsilon;
                Bxmat.vector_mult_transpose(vec, stress);
                res += fe.detJxW(i) * vec;
                
                if (jac) {
                    
                    Bxmat.left_multiply(mat1, mat);
                    Bxmat.right_multiply_transpose(mat2, mat1);
                    (*jac) += fe.detJxW(i) * mat2;
                }
            }
        }
        
        // process the transverse shear strain components
        {
            const typename FEVarType::fe_shape_deriv_t
            &fe = _shear_fe_var_data->get_fe_shape_data();

            typename Eigen::Matrix<scalar_t, 2, 1>
            epsilon,
            stress;
            vector_t
            vec     = vector_t::Zero(3*fe.n_basis());
            
            typename SectionPropertyType::shear_value_t
            mat;
            
            matrix_t
            mat1 = matrix_t::Zero(2, 3*fe.n_basis()),
            mat2 = matrix_t::Zero(3*fe.n_basis(), 3*fe.n_basis());
            
            MAST::Numerics::FEMOperatorMatrix<scalar_t>
            Bxmat;
            Bxmat.reinit(2, 3, fe.n_basis());
            
            
            for (uint_t i=0; i<fe.n_q_points(); i++) {
                
                c.qp = i;
                
                _property->shear_derivative(c, f, mat);
                MAST::Physics::Elasticity::MindlinPlate::transverse_shear_strain
                <scalar_t, scalar_t, FEVarType>
                (*_shear_fe_var_data, i, epsilon, Bxmat);
                stress = mat * epsilon;
                Bxmat.vector_mult_transpose(vec, stress);
                res += fe.detJxW(i) * vec;
                
                if (jac) {
                    
                    Bxmat.left_multiply(mat1, mat);
                    Bxmat.right_multiply_transpose(mat2, mat1);
                    (*jac) += fe.detJxW(i) * mat2;
                }
            }
        }
    }

    
private:
    
    
    
    const SectionPropertyType       *_property;
    const FEVarType                 *_bending_fe_var_data;
    const FEVarType                 *_shear_fe_var_data;
};

}  // namespace MindlinPlate
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST

#endif // __mast_linear_mindlin_plate_strain_energy_h__
