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

// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/solvers/eigen/nonlinear_solver.hpp>

// Test includes
#include <test_helpers.h>

// Eigen includes
#include <Eigen/LU>


namespace MAST {
namespace Test {
namespace Solvers {
namespace EigenWrapper {

inline void perturb(real_t    &dp) { dp = 2.5;}
inline void perturb(complex_t &dp) { dp = complex_t(2.5, 1.e-12);}

/*!
 * a quadratic n-dimensional function \f$ f(x_1, \ldots, x_n) = \sum_i x_i^2 \f$
 */
template <typename ScalarType>
class Function {
  
public:

    using scalar_t = ScalarType;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    uint_t   n;           // total system size
    uint_t   perturb_idx; // index where the origin is perturbed
    scalar_t dp;

    Function(const ScalarType &dp_val): n (10), perturb_idx(n/2), dp(dp_val) { }

    virtual ~Function() { }
    
    inline void init_vector(vector_t &v) { v.setZero(n);}
    
    inline void init_matrix(matrix_t &m) { m.setZero(n,n);}
    
    inline void ref_solution(vector_t &x) {x.setZero(n); x(perturb_idx) = dp;}
    
    inline void ref_solution_sens(vector_t &x) {x.setZero(n); x(perturb_idx) = 1.;}

    inline void residual(const vector_t &x, vector_t &res) {
        
        res.setZero(n);
        
        for (int_t i=0; i<n; i++) {
            
            res(i) = pow(x(i), 2);
        }
        
        // coordinate that is petturbed will need a different residual
        // an arbitrary perturbation of the coordinate is included
        res(perturb_idx) = pow(x(perturb_idx)-dp, 2.);
    }

    
    inline void residual_sensitivity(const vector_t &x, vector_t &dresdp) {

        dresdp.setZero(n);
        dresdp(perturb_idx) = -2.*(x(perturb_idx)-dp);
    }

    inline void jacobian(const vector_t &x, matrix_t &jac) {

        jac.setZero(n, n);
        
        for (int_t i=0; i<n; i++) {
            
            jac(i, i) = 2.*x(i);
        }

        // coordinate that is petturbed will need a different residual/jacobian
        jac(perturb_idx, perturb_idx) = 2.*(x(perturb_idx)-dp);
    }

protected:

};



template <typename FuncType>
void xinit(FuncType                                 &f,
           Eigen::Matrix<real_t, Eigen::Dynamic, 1> &x) {
    
    x = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Random(f.n);
}



template <typename FuncType>
void xinit(FuncType                                 &f,
           Eigen::Matrix<complex_t, Eigen::Dynamic, 1> &x) {
    
    x.setZero(f.n);
    x.real() = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Random(f.n);
}


template <typename FuncType>
void xinit(FuncType                                       &f,
           Eigen::Matrix<adouble_tl_t, Eigen::Dynamic, 1> &x) {
    
    x.setZero(f.n);
    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    x_rand = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Random(f.n);

    for (uint_t i=0; i<x.size(); i++) x(i) = x_rand(i);
}



template <typename ScalarType>
void
solution(const ScalarType                             &dp,
         Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &x,
         Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &x_ref) {
    
    // data types for real-valued computation
    using func_t   = MAST::Test::Solvers::EigenWrapper::Function<ScalarType>;
    using vector_t = typename func_t::vector_t;
    using matrix_t = typename func_t::matrix_t;
    using linear_solver_t = typename Eigen::FullPivLU<matrix_t>;

    func_t f(dp);
    
    vector_t
    dres  = vector_t::Zero(f.n),
    dx    = vector_t::Zero(f.n),
    dx_cs = vector_t::Zero(f.n);
    
    MAST::Test::Solvers::EigenWrapper::xinit(f, x);
    
    MAST::Solvers::EigenWrapper::NonlinearSolver<ScalarType, linear_solver_t, func_t>
    solver;
    solver.rtol = 1.e-10;
    solver.tol  = 1.e-10;
    solver.solve(f, x);

    f.ref_solution(x_ref);
}




void
sensitivity(const real_t                                   &dp,
            const Eigen::Matrix<real_t, Eigen::Dynamic, 1> &x,
            Eigen::Matrix<real_t, Eigen::Dynamic, 1>       &dx,
            Eigen::Matrix<real_t, Eigen::Dynamic, 1>       &dx_ref) {
    
    // data types for real-valued computation
    using func_t   = MAST::Test::Solvers::EigenWrapper::Function<real_t>;
    using vector_t = typename func_t::vector_t;
    using matrix_t = typename func_t::matrix_t;
    using linear_solver_t = typename Eigen::FullPivLU<matrix_t>;

    func_t f(dp);
    
    vector_t
    dres  = vector_t::Zero(f.n);

    // analytical sensitivity
    f.residual_sensitivity(x, dres);
    matrix_t jac;
    f.jacobian(x, jac);
    
    dx = -linear_solver_t(jac).solve(dres);
    f.ref_solution_sens(dx_ref);
}



TEST_CASE("eigen_nonlinear_solver",
          "[Algebra][Solvers][Nonlinear][Eigen][ComplexStep][AdolC]") {

    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    x,
    dx,
    x_ref,
    dx_ref;

    real_t
    dp;
    MAST::Test::Solvers::EigenWrapper::perturb(dp);

    // check the accuracy of solution
    MAST::Test::Solvers::EigenWrapper::solution(dp, x, x_ref);
    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(x),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(x_ref)).margin(1.e-3));


    // check the accuracy of solution sensitivity
    MAST::Test::Solvers::EigenWrapper::sensitivity(dp, x, dx, dx_ref);
    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dx),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(dx_ref)).margin(1.e-3));


    ////////////////////////////////////////////////////////////
    // complex-step sensitivity
    ////////////////////////////////////////////////////////////
    {
        Eigen::Matrix<complex_t, Eigen::Dynamic, 1>
        x_cs,
        x_ref_cs;
        complex_t
        dp_cs;
        MAST::Test::Solvers::EigenWrapper::perturb(dp_cs);

        MAST::Test::Solvers::EigenWrapper::solution(dp_cs, x_cs, x_ref_cs);
        dx = x_cs.imag()/imag(dp_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dx),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(dx_ref)).margin(1.e-6));
    }
    
    
    ////////////////////////////////////////////////////////////
    // Adol-C sensitivity of Newton solver
    ////////////////////////////////////////////////////////////
#if MAST_ENABLE_ADOLC == 1
    {
        Eigen::Matrix<adouble_tl_t, Eigen::Dynamic, 1>
        x_ad,
        x_ref_ad;
        adouble_tl_t
        dp_ad;
        real_t
        unit_sens = 1.;
        // tell the perturbation about its value and its sensitivity wrt the parameter,
        // which is itself. Therefore, the latter value = 1.
        dp_ad = dp;
        dp_ad.setADValue(&unit_sens);
        
        MAST::Test::Solvers::EigenWrapper::solution(dp_ad, x_ad, x_ref_ad);
        
        // extract the sensitivity values.
        for (uint_t i=0; i<x.size(); i++)
        dx(i) = *x_ad(i).getADValue();

        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dx),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(dx_ref)).margin(1.e-3));
    }
#endif
}

} // namespace EigenWrapper
} // namespace Solvers
} // namespace Test
} // namespace MAST


