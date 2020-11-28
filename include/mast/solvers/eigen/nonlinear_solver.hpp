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

#ifndef __mast_eigen_nonlinear_solver_h__
#define __mast_eigen_nonlinear_solver_h__

// C++ includes
#include <iomanip>

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// PETSc includes
#include <petscmat.h>


namespace MAST {
namespace Solvers {
namespace EigenWrapper {

template <typename ScalarType,
          typename LinearSolverType,
          typename FuncType>
class NonlinearSolver {
  
public:
    
    using scalar_t = ScalarType;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = typename FuncType::matrix_t;

    static_assert(std::is_same<ScalarType, typename FuncType::scalar_t>::value,
                  "Scalar type of function and nonlinear solver must be same");
    static_assert(std::is_same<vector_t, typename FuncType::vector_t>::value,
                  "Vector type of function and nonlinear solver must be same");


    real_t tol;
    real_t rtol;
    uint_t max_iter;
    
    NonlinearSolver():
    tol      (1.e-6),
    rtol     (1.e-6),
    max_iter (20),
    _func    (nullptr) {
        
    }
    
    
    virtual ~NonlinearSolver() {
        
    }
    
    /*!
     * initialize the solver for function object \p func that provides the residual and jacobian evaluation.
     * If \p scope is provided then the solver will pass this to the \p KSPSetOptionsPrefix method.
     * This allows specific selection of solver options for different linear solvers in a code.
     * Solves \f$ R(x) = 0 \f$ for solution \f$ x \f$. with \par x0 as the initial guess.
     */
    inline void solve(FuncType          &func,
                      vector_t          &x) {
     
        
        vector_t
        res,
        x0,
        dx;
        
        matrix_t
        jac;
        
        func.init_vector(res);
        func.init_vector(x0);
        func.init_vector(dx);
        func.init_matrix(jac);
        
        func.residual(x, res);

        bool
        if_cont = true;
        
        real_t
        res_l2  = 0.,
        res0_l2 = 0.,
        dx_l2   = 0.;
        
        uint_t
        iter = 0;
        
        res0_l2 = res_l2 = res.norm();
        
        std::cout
        << " Iter: " << std::setw(5) << iter
        << " : || res ||_2 = "
        << std::setw(15) << res_l2;
        
        while (if_cont) {
            
            func.jacobian(x, jac);

            dx    = LinearSolverType(jac).solve(res);
            
            dx_l2 = dx.norm();

            // output
            std::cout
            << " : || dx ||_2 = "
            << std::setw(15) << dx_l2 << std::endl;

            // x = x + dx
            x -= dx;

            // copy solution to another vector
            x0 = x;
            iter++;

            // new residual
            func.residual(x, res);
            
            // check for convergence
            res_l2 = res.norm();

            std::cout
            << " Iter: " << std::setw(5) << iter
            << " : || res ||_2 = "
            << std::setw(15) << res_l2;

            if (res_l2/res0_l2 < rtol) {
                
                if_cont = false;
                std::cout
                << " Terminating due to residual norm relative convergence"
                << std::endl;
            }
            if (res_l2 < tol) {
                
                if_cont = false;
                std::cout
                << " Terminating due to residual norm convergence"
                << std::endl;
            }
            if (dx_l2 < tol) {
                
                if_cont = false;
                std::cout
                << " Terminating due to step norm convergence"
                << std::endl;
            }
            if (iter >= max_iter) {
                
                if_cont = false;
                std::cout
                << " Terminating due to maximum iterations"
                << std::endl;
            }
        }
    }
    
    
private:

    FuncType        *_func;
    std::string      _nm;
};

} // EigenWrapper
} // Solvers
} // MAST

#endif // __mast_eigen_nonlinear_solver_h__
