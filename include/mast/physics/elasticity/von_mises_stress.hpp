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
using stress_vec_t = Eigen::Matrix<ScalarType,
                                   MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
                                   1>;

template <typename ScalarType, uint_t Dim>
using stress_adjoint_mat_t = Eigen::Matrix<ScalarType,
                                   MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
                                   Eigen::Dynamic>;



template <typename ScalarType, uint_t Dim>
inline
typename std::enable_if<Dim==2, ScalarType>::type
vonMises_stress(stress_vec_t<ScalarType, Dim>& stress) {
   
    Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
            stress.size(),
            "Incorrect stress vector dimension");
    
    return
    pow(0.5 * (pow(stress(0)-stress(1),2) +     //(((sigma_xx - sigma_yy)^2    +
               pow(stress(1),2) +               //  (sigma_yy)^2    +
               pow(stress(0),2)) +              //  (sigma_xx)^2)/2 +
        3.0 * (pow(stress(2), 2)), 0.5);        //   3.0 * tau_xy^2)^.5
}



template <typename ScalarType, uint_t Dim>
inline
typename std::enable_if<Dim==3, ScalarType>::type
vonMises_stress(stress_vec_t<ScalarType, Dim>& stress) {
   
    Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
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



template <typename ScalarType, uint_t Dim>
inline
typename std::enable_if<Dim==2, ScalarType>::type
vonMises_stress_derivative(stress_vec_t<ScalarType, Dim>& stress,
                           stress_vec_t<ScalarType, Dim>& dstress_dp) {
    
    Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
            stress.size(),
            "Incorrect stress vector dimension");
    Assert1(dstress_dp.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
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




template <typename ScalarType, uint_t Dim>
inline
typename std::enable_if<Dim==3, ScalarType>::type
vonMises_stress_derivative(stress_vec_t<ScalarType, Dim>& stress,
                           stress_vec_t<ScalarType, Dim>& dstress_dp) {
    
    Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
            stress.size(),
            "Incorrect stress vector dimension");
    Assert1(dstress_dp.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
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



template <typename ScalarType, uint_t Dim>
inline
typename std::enable_if<Dim==2, void>::type
vonMises_stress_dX(stress_vec_t<ScalarType, Dim>                &stress,
                   stress_adjoint_mat_t<ScalarType, Dim>        &dstress_dX,
                   Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &vm_adjoint) {
    
    Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
            stress.size(),
            "Incorrect stress vector dimension");
    Assert1(dstress_dX.rows() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
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



template <typename ScalarType, uint_t Dim>
inline
typename std::enable_if<Dim==3, void>::type
vonMises_stress_dX(stress_vec_t<ScalarType, Dim>                &stress,
                   stress_adjoint_mat_t<ScalarType, Dim>        &dstress_dX,
                   Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &vm_adjoint) {
    
    Assert1(stress.size() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
            stress.size(),
            "Incorrect stress vector dimension");
    Assert1(dstress_dX.rows() == MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<Dim>::value,
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


}  // namespace LinearContinuum
}  // namespace Elasticity
}  // namespace Physics
}  // namespace MAST

#endif // __mast_linear_continuum_von_mises_stress_h__
