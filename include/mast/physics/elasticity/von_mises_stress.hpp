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

#ifndef __mast_linear_continuum_von_mises_stress_h__
#define __mast_linear_continuum_von_mises_stress_h__

// MAST includes
#include <mast/physics/elasticity/linear_elastic_strain_operator.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {
namespace LinearContinuum {



template <typename ScalarType, uint_t Dim>
struct
vonMisesStress {};


template <typename ScalarType>
struct
vonMisesStress<ScalarType, 2> {
    
    template <typename VecType>
    static inline ScalarType
    value(const VecType& stress) {

        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<VecType>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        
        return
        pow(0.5 * (pow(stress(0)-stress(1),2) +     //(((sigma_xx - sigma_yy)^2    +
                   pow(stress(1),2) +               //  (sigma_yy)^2    +
                   pow(stress(0),2)) +              //  (sigma_xx)^2)/2 +
            3.0 * (pow(stress(2), 2)), 0.5);        //   3.0 * tau_xy^2)^.5
    }

    
    
    template <typename Vec1Type, typename Vec2Type>
    static inline void
    derivative(const Vec1Type  &stress,
               Vec2Type        &dstress) {
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec1Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec2Type>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                dstress.size(),
                "Incorrect stress vector dimension");
        
        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1),2) +               // (sigma_yy)^2    +
               pow(stress(0),2)) +              // (sigma_xx)^2)/2 +
        3.0 * (pow(stress(2), 2));              //  3.0 * tau_xy^2)
        
        dstress.setZero();
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.) {
            
            dstress(0) =   (stress(0) - stress(1)) + stress(0);
            dstress(1) = - (stress(0) - stress(1)) + stress(1);
            dstress(2) = 6. * stress(2);
            
            dstress *= 0.5 * pow(p, -0.5);
        }
    }

    
    template <typename VecType, typename MatType>
    static inline void
    second_derivative(const VecType   &stress,
                      MatType         &dstress) {
        
        using scalar_t = typename Eigen::internal::traits<VecType>::Scalar;
        const uint_t
        ns = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value;
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<VecType>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<MatType>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress.rows() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                dstress.rows(),
                "Incorrect derivative matrix row dimension");
        Assert1(dstress.cols() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                dstress.cols(),
                "Incorrect derivative matrix column dimension");

        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1),2) +               // (sigma_yy)^2    +
               pow(stress(0),2)) +              // (sigma_xx)^2)/2 +
        3.0 * (pow(stress(2), 2)),              //  3.0 * tau_xy^2)
        sqrtp_2 = .5 * pow(p, -0.5);
        
        dstress.setZero();
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.) {
            
            Eigen::Matrix<scalar_t, ns, 1>
            ds;
            ds(0) =   (stress(0) - stress(1)) + stress(0);
            ds(1) = - (stress(0) - stress(1)) + stress(1);
            ds(2) = 6. * stress(2);
            
            dstress  = -.25 / pow(p, 1.5) * ds * ds.transpose();
            
            dstress(0, 0) +=  2. * sqrtp_2;
            dstress(0, 1) += -1. * sqrtp_2;
            
            dstress(1, 0) += -1. * sqrtp_2;
            dstress(1, 1) +=  2. * sqrtp_2;

            dstress(2, 2) +=  6. * sqrtp_2;
        }
    }

    
    
    template <typename Vec1Type, typename Vec2Type>
    static inline ScalarType
    derivative_sens(const Vec1Type  &stress,
                    const Vec2Type  &dstress_dp) {
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec1Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec2Type>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress_dp.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                dstress_dp.size(),
                "Incorrect stress vector dimension");
        
        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1),2) +               // (sigma_yy)^2    +
               pow(stress(0),2)) +              // (sigma_xx)^2)/2 +
        3.0 * (pow(stress(2), 2)),              //  3.0 * tau_xy^2)
        dp = 0.;
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.)
            dp =
            (((dstress_dp(0) - dstress_dp(1)) * (stress(0) - stress(1)) +
              (dstress_dp(1)) * (stress(1)) +
              (dstress_dp(0)) * (stress(0))) +
             6.0 * (dstress_dp(2) * stress(2))) * 0.5 * pow(p, -0.5);
        
        return dp;
    }
    
    
    
    template <typename Vec1Type, typename MatType, typename Vec2Type>
    static inline void
    stress_dX(const Vec1Type   &stress,
              const MatType    &dstress_dX,
              Vec2Type         &vm_adjoint) {
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec1Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec2Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<MatType>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress_dX.rows() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value,
                dstress_dX.rows(),
                "Incorrect stress adjoint matrix rows");
        Assert2(vm_adjoint.size() == dstress_dX.cols(),
                vm_adjoint.size(),   dstress_dX.cols(),
                "Incorrect von Mises adjoint vector dimension");
        
        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1),2) +               // (sigma_yy)^2    +
               pow(stress(0),2)) +              // (sigma_xx)^2)/2 +
        3.0 * (pow(stress(2), 2));              // 3* (tau_xy^2)
        
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.)
            vm_adjoint =
            (((dstress_dX.row(0) - dstress_dX.row(1)) * (stress(0) - stress(1)) +
              dstress_dX.row(1) * stress(1) +
              dstress_dX.row(0) * stress(0)) +
             6.0 * dstress_dX.row(2) * stress(2)) * 0.5 * pow(p, -0.5);
    }
    
    
};


template <typename ScalarType>
struct
vonMisesStress<ScalarType, 3> {
    
    template <typename VecType>
    static inline ScalarType
    value(const VecType& stress) {

        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<VecType>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        

        return
        pow(0.5 * (pow(stress(0)-stress(1),2) +    //(((sigma_xx - sigma_yy)^2    +
                   pow(stress(1)-stress(2),2) +    //  (sigma_yy - sigma_zz)^2    +
                   pow(stress(2)-stress(0),2)) +   //  (sigma_zz - sigma_xx)^2)/2 +
            3.0 * (pow(stress(3), 2) +              // 3* (tau_xy^2 +
                   pow(stress(4), 2) +              //     tau_yz^2 +
                   pow(stress(5), 2)), 0.5);        //     tau_zx^2))^.5
    }
    
    
    /*!
     * computes the derivative of
     */
    template <typename Vec1Type, typename Vec2Type>
    static inline void
    derivative(const Vec1Type  &stress,
               Vec2Type        &dstress) {
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec1Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec2Type>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                dstress.size(),
                "Incorrect stress vector dimension");

        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1)-stress(2),2) +    // (sigma_yy - sigma_zz)^2    +
               pow(stress(2)-stress(0),2)) +   // (sigma_zz - sigma_xx)^2)/2 +
        3.0 * (pow(stress(3), 2) +              // 3* (tau_xy^2 +
               pow(stress(4), 2) +              //     tau_yz^2 +
               pow(stress(5), 2));              //     tau_zx^2)
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.) {
            
            dstress(0) =   (stress(0) - stress(1)) - (stress(2) - stress(0));
            dstress(1) = - (stress(0) - stress(1)) + (stress(1) - stress(2));
            dstress(2) = - (stress(1) - stress(2)) + (stress(2) - stress(0));
            dstress(3) = 6. * stress(3);
            dstress(4) = 6. * stress(4);
            dstress(5) = 6. * stress(5);
            
            dstress *= 0.5 * pow(p, -0.5);
        }
    }

    
    
    template <typename VecType, typename MatType>
    static inline void
    second_derivative(const VecType   &stress,
                      MatType         &dstress) {
        
        using scalar_t = typename Eigen::internal::traits<VecType>::Scalar;
        const uint_t
        ns = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value;

        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<VecType>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<MatType>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress.rows() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                dstress.rows(),
                "Incorrect derivative matrix row dimension");
        Assert1(dstress.cols() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                dstress.cols(),
                "Incorrect derivative matrix column dimension");

        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1)-stress(2),2) +    // (sigma_yy - sigma_zz)^2    +
               pow(stress(2)-stress(0),2)) +   // (sigma_zz - sigma_xx)^2)/2 +
        3.0 * (pow(stress(3), 2) +              // 3* (tau_xy^2 +
               pow(stress(4), 2) +              //     tau_yz^2 +
               pow(stress(5), 2)),              //     tau_zx^2)
        sqrtp_2 = .5 * pow(p, -0.5);

        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.) {

            Eigen::Matrix<scalar_t, ns, 1>
            ds;
            ds(0) =   (stress(0) - stress(1)) - (stress(2) - stress(0));
            ds(1) = - (stress(0) - stress(1)) + (stress(1) - stress(2));
            ds(2) = - (stress(1) - stress(2)) + (stress(2) - stress(0));
            ds(3) = 6. * stress(3);
            ds(4) = 6. * stress(4);
            ds(5) = 6. * stress(5);

            dstress  = -.25 / pow(p, 1.5) * ds * ds.transpose();
            
            dstress(0, 0) +=   2. * sqrtp_2;
            dstress(0, 1) +=  -1. * sqrtp_2;
            dstress(0, 2) +=  -1. * sqrtp_2;

            dstress(1, 0) +=  -1. * sqrtp_2;
            dstress(1, 1) +=   2. * sqrtp_2;
            dstress(1, 2) +=  -1. * sqrtp_2;

            dstress(2, 0) +=  -1. * sqrtp_2;
            dstress(2, 1) +=  -1. * sqrtp_2;
            dstress(2, 2) +=   2. * sqrtp_2;

            dstress(3, 3) += 6. * sqrtp_2;
            dstress(4, 4) += 6. * sqrtp_2;
            dstress(5, 5) += 6. * sqrtp_2;
        }
    }

    
    template <typename Vec1Type, typename Vec2Type>
    static inline ScalarType
    derivative_sens(const Vec1Type  &stress,
                    const Vec2Type  &dstress_dp) {
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec1Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec2Type>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress_dp.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                dstress_dp.size(),
                "Incorrect stress vector dimension");

        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1)-stress(2),2) +    // (sigma_yy - sigma_zz)^2    +
               pow(stress(2)-stress(0),2)) +   // (sigma_zz - sigma_xx)^2)/2 +
        3.0 * (pow(stress(3), 2) +              // 3* (tau_xy^2 +
               pow(stress(4), 2) +              //     tau_yz^2 +
               pow(stress(5), 2)),              //     tau_zx^2)
        dp = 0.;
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.)
            dp =
            (((dstress_dp(0) - dstress_dp(1)) * (stress(0) - stress(1)) +
              (dstress_dp(1) - dstress_dp(2)) * (stress(1) - stress(2)) +
              (dstress_dp(2) - dstress_dp(0)) * (stress(2) - stress(0))) +
             6.0 * (dstress_dp(3) * stress(3)+
                    dstress_dp(4) * stress(4)+
                    dstress_dp(5) * stress(5))) * 0.5 * pow(p, -0.5);
        
        return dp;
    }

    
    template <typename Vec1Type, typename MatType, typename Vec2Type>
    static inline void
    stress_dX(const Vec1Type   &stress,
              const MatType    &dstress_dX,
              Vec2Type         &vm_adjoint) {
        
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec1Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<Vec2Type>::Scalar>::value,
                      "Scalar type must be same");
        static_assert(std::is_same<ScalarType, typename Eigen::internal::traits<MatType>::Scalar>::value,
                      "Scalar type must be same");
        Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                stress.size(),
                "Incorrect stress vector dimension");
        Assert1(dstress_dX.rows() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<3>::value,
                dstress_dX.rows(),
                "Incorrect stress adjoint matrix rows");
        Assert2(vm_adjoint.size() == dstress_dX.cols(),
                vm_adjoint.size(),   dstress_dX.cols(),
                "Incorrect von Mises adjoint vector dimension");

        ScalarType
        p =
        0.5 * (pow(stress(0)-stress(1),2) +    //((sigma_xx - sigma_yy)^2    +
               pow(stress(1)-stress(2),2) +    // (sigma_yy - sigma_zz)^2    +
               pow(stress(2)-stress(0),2)) +   // (sigma_zz - sigma_xx)^2)/2 +
        3.0 * (pow(stress(3), 2) +              // 3* (tau_xx^2 +
               pow(stress(4), 2) +              //     tau_yy^2 +
               pow(stress(5), 2));              //     tau_zz^2)
        
        
        // if p == 0, then the sensitivity returns nan
        // Hence, we are avoiding this by setting it to zero whenever p = 0.
        if (fabs(p) > 0.)
            vm_adjoint =
            (((dstress_dX.row(0) - dstress_dX.row(1)) * (stress(0) - stress(1)) +
              (dstress_dX.row(1) - dstress_dX.row(2)) * (stress(1) - stress(2)) +
              (dstress_dX.row(2) - dstress_dX.row(0)) * (stress(2) - stress(0))) +
             6.0 * (dstress_dX.row(3) * stress(3)+
                    dstress_dX.row(4) * stress(4)+
                    dstress_dX.row(5) * stress(5))) * 0.5 * pow(p, -0.5);
    }
};

}  // namespace LinearContinuum
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST

#endif // __mast_linear_continuum_von_mises_stress_h__
