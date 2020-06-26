
#ifndef __mast_linear_continuum_strain_energy_h__
#define __mast_linear_continuum_strain_energy_h__

// MAST includes
#include <mast/physics/elasticity/linear_elastic_strain_operator.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {



template <typename FEVarType,
          typename SectionPropertyType,
          uint_t Dim,
          typename ContextType>
class LinearContinuumStrainEnergy {
    
public:
    
    template <uint_t D> struct NStrainComponents { };
    template <> struct NStrainComponents<1> { static const uint_t value = 1; };
    template <> struct NStrainComponents<2> { static const uint_t value = 3; };
    template <> struct NStrainComponents<3> { static const uint_t value = 6; };

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_basis_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using section_scalar_t = typename SectionPropertyType::scalar_t;
    using fe_basis_t       = typename FEVarType::fe_basis_t;
    static const uint_t n_strain  = NStrainComponents<Dim>::value;
    
    LinearContinuumStrainEnergy():
    _property    (nullptr),
    _fe_var_data (nullptr)
    { }
    
    virtual ~LinearContinuumStrainEnergy() { }

    inline void
    set_section_property(const SectionPropertyType& p) {
        
        Assert0(!_property, "Property already initialized.");
        
        _property = &p;
    }

    inline void set_fe_var_data(const FEVarType& fe_data)
    {
        Assert0(!_fe_var_data, "FE data already initialized.");
        _fe_var_data = &fe_data;
    }

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return Dim*_fe_var_data->get_fe_shape_data().n_basis();
    }
    
    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");
        
        const typename FEVarType::fe_basis_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        typename Eigen::Matrix<scalar_t, n_strain, 1>
        epsilon,
        stress;
        typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>
        vec     = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>::Zero(2*fe.n_basis());
        
        typename SectionPropertyType::value_t
        mat;
        
        typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>
        mat1 = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_strain, 2*fe.n_basis()),
        mat2 = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(2*fe.n_basis(), 2*fe.n_basis());

        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bxmat;
        Bxmat.reinit(n_strain, 2, fe.n_basis());

        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp = i;
            
            _property->value(c, mat);
            MAST::Physics::Elasticity::linear_continuum_strain
            <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, epsilon, Bxmat);
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

    template <typename ScalarFieldType>
    inline void derivative(ContextType& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_property, "Section property not initialized");
        
        const typename FEVarType::fe_basis_t
        &fe = _fe_var_data->get_fe_shape_data();

        typename Eigen::Matrix<scalar_t, n_strain, 1>
        epsilon,
        stress;
        typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>
        vec     = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>::Zero(2*fe.n_basis());

        typename SectionPropertyType::value_t
        mat;
        typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>
        mat1 = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_strain, 2*fe.n_basis()),
        mat2 = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(2*fe.n_basis(), 2*fe.n_basis());

        MAST::Numerics::FEMOperatorMatrix<scalar_t>
        Bxmat;
        Bxmat.reinit(n_strain, Dim, fe.n_basis());

        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp = i;
            
            _property->derivative(c, f, mat);
            MAST::Physics::Elasticity::linear_continuum_strain
            <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, epsilon, Bxmat);
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

    
private:
    
    
    const SectionPropertyType       *_property;
    const FEVarType                 *_fe_var_data;
};

}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST

#endif // __mast_linear_continuum_strain_energy_h__
