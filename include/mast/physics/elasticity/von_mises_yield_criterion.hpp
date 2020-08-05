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

#ifndef __mast_elastoplasticity_von_mises_yield_criterion_h__
#define __mast_elastoplasticity_von_mises_yield_criterion_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/physics/elasticity/linear_elastic_strain_operator.hpp>
//#include <mast/physics/elasticity/return_mapping_solver.hpp>
#include <mast/physics/elasticity/von_mises_stress.hpp>


namespace MAST {
namespace Physics {
namespace Elasticity {
namespace ElastoPlasticity {

template <typename ScalarType,
          typename YieldCriterionType>
class Accessor {
  
public:
    
    using stress_t       =
    Eigen::Map<Eigen::Matrix<ScalarType, YieldCriterionType::n_strain, 1>>;
    using const_stress_t =
    Eigen::Map<const Eigen::Matrix<ScalarType, YieldCriterionType::n_strain, 1>>;

    Accessor():
    _n_dofs  (-1),
    _data    (nullptr),
    _yield   (nullptr) { }
    
    
    virtual ~Accessor() { }

    
    inline void init(YieldCriterionType   &yield,
                     ScalarType           *data) {
        
        _yield  = &yield;
        _data   = data;
        _n_dofs = yield.n_variables();
    }
    
    
    inline stress_t stress() {
        
        return stress_t(_data, YieldCriterionType::n_strain, 1);
    }

    
    inline const_stress_t stress() const {
        
        return const_stress_t(_data, YieldCriterionType::n_strain, 1);
    }


    inline stress_t plastic_strain() {
        
        return stress_t(&(_data[YieldCriterionType::n_strain-1]), YieldCriterionType::n_strain, 1);
    }

    
    inline const_stress_t plastic_strain() const {
        
        return const_stress_t(&(_data[YieldCriterionType::n_strain-1]), YieldCriterionType::n_strain, 1);
    }

    
    inline stress_t back_stress() {
        
        Assert0(_yield->if_kinematic_hardening,
                "Kinematic hardening not enabled for yield criterion");
        
        return stress_t(&(_data[2*YieldCriterionType::n_strain-1]), YieldCriterionType::n_strain, 1);
    }

    
    inline const_stress_t back_stress() const {
        
        Assert0(_yield->if_kinematic_hardening,
                "Kinematic hardening not enabled for yield criterion");
        
        return const_stress_t(&(_data[2*YieldCriterionType::n_strain-1]), YieldCriterionType::n_strain, 1);
    }

    
    inline ScalarType consistency_parameter() {
        
        return _data[_n_dofs-1];
    }

    
private:
   
    uint_t               _n_dofs;
    ScalarType          *_data;
    YieldCriterionType  *_yield;
};



template <typename ScalarType,
          typename MaterialType>
class vonMisesYieldFunction {
    
public:
    
    using scalar_t   = typename MaterialType::scalar_t;
    static_assert(std::is_same<scalar_t, typename MaterialType::scalar_t>::value,
                  "Scalar type should be same for yield function and material stiffness");
    using material_t = MaterialType;
    static const
    uint_t dim       = MaterialType::dim;
    static const
    uint_t n_strain  = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<dim>::value;
    using stiff_t    = typename MaterialType::value_t;
    
    
    vonMisesYieldFunction():
    _if_kinematic_hardening (false),
    _vm_lim                 (0.),
    _material               (nullptr) {
        
    }
    
    
    virtual ~vonMisesYieldFunction() { }
    
    
    
    inline void set_limit_stress(const real_t v) {
        
        _vm_lim = v;
    }


    inline void set_material(material_t  &m) {
        
        _material = &m;
    }

    
    inline bool if_kinematic_hardening() const {
        
        return _if_kinematic_hardening;
    }

    
    inline uint_t n_variables() const {

        uint_t
        n  = 2*n_strain+1; // stress, plastic strain and consistency parameter
        
        // if kinematic hardening is enabled then a backstress is included
        n += _if_kinematic_hardening?n_strain:0;
            
        return n;
    }


    
    template <typename ContextType,
              typename VecType,
              typename AccessorType>
    inline void compute(ContextType     &c,
                        const VecType   &strain,
                        AccessorType    &internal,
                        stiff_t         *stiff = nullptr) {
        
        Assert2(strain.size() == n_strain,
                strain.size(), n_strain,
                "Incompatible strain vector dimension");
        Assert2(strain.size() == n_strain,
                strain.size(), n_strain,
                "Incompatible strain vector dimension");

        
        Eigen::Matrix<scalar_t, n_strain, 1>
        strain_e  = Eigen::Matrix<scalar_t, n_strain, 1>::Zero(n_strain);
        
        // elastic part of the strain
        strain_e = strain - internal.plastic_strain();
        
        stiff_t
        m_stiff = stiff_t::Zero();
        
        // material stiffness matrix
        _material->value(c, m_stiff);
        
        // stress due to elastic strain
        internal.stress() = m_stiff * strain_e;
        
        // evaluate the value of yield criterion
        scalar_t
        yield = this->yield_function(c, internal);
        
        // if the yield function is violated then an update needs to be computed
        if (yield > 0.) {
            
            Eigen::Matrix<scalar_t, n_strain, 1>
            n    = Eigen::Matrix<scalar_t, n_strain, 1>::Zero();

            Eigen::Matrix<scalar_t, n_strain+1, 1>
            x    = Eigen::Matrix<scalar_t, n_strain+1, 1>::Zero(),
            res  = Eigen::Matrix<scalar_t, n_strain+1, 1>::Zero();
            
            Eigen::Matrix<scalar_t, n_strain, n_strain>
            dnds = Eigen::Matrix<scalar_t, n_strain, n_strain>::Zero();

            Eigen::Matrix<scalar_t, n_strain+1, n_strain+1>
            jac  = Eigen::Matrix<scalar_t, n_strain+1, n_strain+1>::Zero();

            bool
            terminate = false;

            // initialize the solution
            x.topRows(n_strain) = internal.stress();
            x(n_strain) = (c.current_plasticity_accessor->consistency_parameter() -
                           c.previous_plasticity_accessor->consistency_parameter());
            
            real_t
            res_norm = 0.,
            tol      = 1.e-10;
            
            uint_t
            iter     = 0,
            max_it   = 10;
            
            while (!terminate) {
                
                MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<scalar_t, dim>::
                derivative(internal.stress(), n);
                
                // derivative of normal, needed for Jacobian
                MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<scalar_t, dim>::
                second_derivative(internal.stress(), dnds);

                ////////////////////////////
                // residual: stress update
                ////////////////////////////
                res.topRows(n_strain) =
                x.topRows(n_strain) - c.previous_plasticity_accessor->stress() +
                m_stiff * (strain - c.previous_plasticity_accessor->plastic_strain() -
                           x(n_strain) * n);
                
                // yield criterion
                res(n_strain) = yield_function(x, internal);

                // check the norm and if we need to continue iteration
                res_norm = res.norm();
                std::cout
                << "Iter: " << std::setw(5) << iter
                << " || res ||_2 = "
                << std::setw(20) << res_norm << std::endl;
                
                if (res_norm >= tol && iter < max_it) {
                    
                    ////////////////////////////
                    // Jacobian
                    ////////////////////////////
                    jac.topLeftCorner(n_strain, n_strain) = x(n_variables()) * m_stiff * dnds;
                    for (uint_t i=0; i<n_strain; i++) jac(i,i) += 1.;
                    
                    jac.topRightCorner(n_strain, 1) = m_stiff * n;
                    
                    jac.bottomLeftCorner(1, n_strain) = n.transpose();
                    
                    ////////////////////////////
                    // update to the solution
                    ////////////////////////////
                    x -= Eigen::FullPivLU<Eigen::Matrix<scalar_t, n_strain+1, n_strain+1>>(jac).solve(res);
                    
                    // increment the iteration
                    iter++;
                }
                else if (iter == max_it) {
                    
                    terminate = true;
                    std::cout << "Terminating Return-Mapping due to maximum iteration" << std::endl;
                }
                else if (res_norm < tol) {
                    
                    terminate = true;
                    std::cout << "Terminating Return-Mapping due to converged residual" << std::endl;
                }
            }
        }
        
        // compute the stiffness tangent if the matrix was provided
        if (stiff) {
            
            if (yield <= 0.)
                // for elastic material response the standard material stiffness is acceptable
                *stiff = m_stiff;
            else
                // for plastic material the tangent stiffness is to be computed using the current
                // solution data
                this->compute_tangent_stiffness(c, internal, *stiff);
        }
    }
    
    
    /*!
     * computes \f[ f(\sigma) = \sqrt(\sigma_{vm}) - \bar{\sigma}  \f], where \f$ \sigma_{vm} \f$ is the
     * von-Mises stress and \f$ \bar{sigma} \f$ is the limit value.
     */
    template <typename ContextType,
              typename AccessorType>
    inline scalar_t yield_function(ContextType   &c,
                                   AccessorType  &internal) {

        /*Eigen::Matrix<scalar_t, n_strain, 1>
        s;
        
        for (uint_t i=0; i<n_strain; i++)
            s(i) = stress(i);// - internal(i);
         */
        
        scalar_t
        v =  MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<scalar_t, dim>::value(internal.stress());
        
        v -= _vm_lim;
        
        return v;
    }
    
    

    /*!
     * computes \f$  C^{tan} = C - \frac{ C \partial_\sigma f C n }{ \partial_\sigma f C n } \f$
     */
    template <typename ContextType,
              typename AccessorType>
    inline void compute_tangent_stiffness(ContextType         &c,
                                          const AccessorType  &internal,
                                          stiff_t             &stiff) {

        Assert2(internal.stress().size() == n_strain,
                internal.stress().size(), n_strain,
                "Incompatible vector size");
        Assert2(stiff.rows() == n_strain,
                stiff.rows(), n_strain,
                "Incompatible matrix row dimension");
        Assert2(stiff.cols() == n_strain,
                stiff.cols(), n_strain,
                "Incompatible matrix column dimension");


        _material->value(c, stiff);

        Eigen::Matrix<scalar_t, n_strain, 1>
        n   = Eigen::Matrix<scalar_t, n_strain, 1>::Zero(n_strain),
        Cn  = Eigen::Matrix<scalar_t, n_strain, 1>::Zero(n_strain);
        
        // computation of \partial f/partial \sigma
        MAST::Physics::Elasticity::LinearContinuum::vonMisesStress<scalar_t, dim>::
        derivative(internal.stress(), n);
        
        Cn  = stiff * n;
        
        stiff -= (Cn*Cn.transpose())/(n.dot(Cn));
    }
    
private:
    
    bool             _if_kinematic_hardening;
    real_t           _vm_lim;
    MaterialType    *_material;
};

} // namespace MAST
} // namespace Physics
} // namespace Elasticity
} // namespace ElastoPlasticity

#endif // __mast_elastoplasticity_von_mises_yield_criterion_h__
