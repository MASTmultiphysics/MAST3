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

/*!
 * a quadratic n-dimensional function \f$ f(x_1, \ldots, x_n) = \sum_i x_i^2 \f$
 */
template <typename ScalarType>
class Function {
  
public:

    using scalar_t = ScalarType;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    uint_t n;           // total system size
    uint_t perturb_idx; // index where the origin is perturbed
    real_t dp;

    Function(): n (10), perturb_idx(n/2), dp(2.5) { }

    virtual ~Function() { }
    
    inline void init_vector(vector_t &v) { v.setZero(n);}
    
    inline void init_matrix(matrix_t &m) { m.setZero(n,n);}
    
    inline void ref_solution(vector_t &x) {x.setZero(n); x(perturb_idx) = dp;}
    
    inline void ref_solution_sens(vector_t &x) {x.setZero(n); x(perturb_idx) = 1.;}

    inline void residual(const vector_t &x, vector_t &res) {
        
        res.setZero(n);
        
        for (int_t i=0; i<n; i++) {
            
            res(i) = std::pow(x(i), 2);
        }
        
        // coordinate that is petturbed will need a different residual
        // an arbitrary perturbation of the coordinate is included
        res(perturb_idx) = std::pow(x(perturb_idx)-dp, 2.);
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




template <typename ScalarType>
void
solution(Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &x,
         Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &x_ref) {
    
    // data types for real-valued computation
    using func_t   = MAST::Test::Solvers::EigenWrapper::Function<ScalarType>;
    using vector_t = typename func_t::vector_t;
    using matrix_t = typename func_t::matrix_t;
    using linear_solver_t = typename Eigen::FullPivLU<matrix_t>;

    func_t f;
    
    vector_t
    dres  = vector_t::Zero(f.n),
    dx    = vector_t::Zero(f.n),
    dx_cs = vector_t::Zero(f.n);
    x     = vector_t::Random(f.n);
    
    MAST::Solvers::EigenWrapper::NonlinearSolver<real_t, linear_solver_t, func_t>
    solver;
    solver.rtol = 1.e-10;
    solver.tol  = 1.e-10;
    solver.solve(f, x);

    f.ref_solution(x_ref);
}




template <typename ScalarType>
void
sensitivity(const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1> &x,
            Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>       &dx,
            Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>       &dx_ref) {
    
    // data types for real-valued computation
    using func_t   = MAST::Test::Solvers::EigenWrapper::Function<ScalarType>;
    using vector_t = typename func_t::vector_t;
    using matrix_t = typename func_t::matrix_t;
    using linear_solver_t = typename Eigen::FullPivLU<matrix_t>;

    func_t f;
    
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
          "[Algebra][Solvers][Nonlinear][Eigen][Complex-Step]") {

    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    x,
    dx,
    x_ref,
    dx_ref;

    // check the accuracy of solution
    MAST::Test::Solvers::EigenWrapper::solution(x, x_ref);
    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(x),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(x_ref)).margin(1.e-3));


    // check the accuracy of solution sensitivity
    MAST::Test::Solvers::EigenWrapper::sensitivity(x, dx, dx_ref);
    CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dx),
               Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(dx_ref)).margin(1.e-3));


    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    x_cs,
    x_ref_cs;
    
    // complex-step sensitivity
}

} // namespace EigenWrapper
} // namespace Solvers
} // namespace Test
} // namespace MAST


