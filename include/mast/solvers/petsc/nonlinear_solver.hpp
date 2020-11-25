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

#ifndef __mast_petsc_nonlinear_solver_h__
#define __mast_petsc_nonlinear_solver_h__

// C++ includes
#include <iomanip>

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/solvers/petsc/linear_solver.hpp>

// PETSc includes
#include <petscmat.h>


namespace MAST {
namespace Solvers {
namespace PETScWrapper {

template <typename FuncType>
class NonlinearSolver {
  
public:
    
    real_t tol;
    real_t rtol;
    uint_t max_iter;
    
    NonlinearSolver(const MPI_Comm comm):
    tol      (1.e-6),
    rtol     (1.e-6),
    max_iter (20),
    _comm    (comm),
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
                      Vec                x,
                      const std::string *scope = nullptr) {
     
        
        Vec res, x0, dx;
        Mat *jac;
        
        VecDuplicate(x, &res);
        VecDuplicate(x,  &x0);
        VecDuplicate(x,  &dx);
        
        VecZeroEntries(res);

        func.residual(x0, res);
        jac = func.matrix();

        bool
        if_cont = true;
        
        real_t
        res_l2  = 0.,
        res0_l2 = 0.,
        dx_l2   = 0.;
        
        uint_t
        iter = 0;
        
        VecNorm(res, NORM_2, &res_l2);
        res0_l2 = res_l2;
        
        std::cout
        << " Iter: " << std::setw(5) << iter
        << " : || res ||_2 = "
        << std::setw(15) << res_l2;
        
        while (if_cont) {
            
            func.jacobian(x, *jac);

            MAST::Solvers::PETScWrapper::LinearSolver solver(_comm);
            init(jac, _nm.size()?&_nm:nullptr);
            solve(dx, res);
            
            VecNorm(dx, NORM_2, &dx_l2);

            // output
            std::cout
            << " : || dx ||_2 = "
            << std::setw(15) << dx_l2 << std::endl;

            // x = x + dx
            VecAXPY(x, -1., dx);

            // copy solution to another vector
            VecCopy(x, x0);
            iter++;

            // new residual
            func.residual(x, res);
            
            // check for convergence
            VecNorm(res, NORM_2, &res_l2);

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

    const MPI_Comm   _comm;
    FuncType        *_func;
    std::string      _nm;
};

} // PETScWrapper
} // Solvers
} // MAST

#endif // __mast_petsc_nonlinear_solver_h__
