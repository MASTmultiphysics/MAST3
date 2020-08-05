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

#ifndef __mast_numerics_nonlinear_solver_h__
#define __mast_numerics_nonlinear_solver_h__


// MAST includes
#include <mast/base/parameter_data.hpp>


namespace MAST {
namespace Numerics {
namespace EigenWrapper {

template <typename VecType,
          typename MatType,
          typename ComputeKernelType,
          typename LinearSolverType = Eigen::FullPivLU<MatType>,
          typename LineSearchType = void>
class NonlinearSolver {
    
public:
    
    static_assert(std::is_same<VecType::Scalar, MatType::Scalar>::value,
                  "Vector and Matrix must use same scalar types");
    using scalar_t        = typename VecType::Scalar;
    using linear_solver_t = LinearSolverType;
    
    uint_t       output_indent_level;
    uint_t       verbose_level;
    uint_t       max_iterations;
    uint_t       update_jacobian_frequency;
    real_t       residual_atol;
    real_t       residual_rtol;
    real_t       dx_atol;
    
    
    NonlinearSolver():
    _linear_solver   (nullptr),
    _line_search     (nullptr),
    _compute         (nullptr),
    _output_indent_level       (0),
    _verbose_level             (1),
    _max_iterations            (20),
    _update_jacobian_frequency (1),
    residual_atol              (1.e-10),
    residual_rtol              (1.e-8),
    dx_atol                    (1.e-10)
    { }
    
    virtual ~NonlinearSolver() {
        
        delete _parameter_data;
    }
    
    
    inline void set_linear_solver(LinearSolverType &solver) {
        
        _linear_solver = &solver;
    }

    inline void set_line_search(LineSearchType &ls) {
        
        _line_search = &ls;
    }

    inline void set_compute_kernel(ComputeKernelType &compute) {
        
        _compute = &compute;
    }

    
    inline MAST::Base::ParameterData&
    get_parameter_data() {
        return _parameter_data;
    }

    /*!
     *  solve starting with the initial guess
     */
    template <typename ContextType>
    inline void solve(ContextType c, VecType& x) {
        
        Assert0(_linear_solver, "Linear solver not set");
        Assert0(_compute, "Residual and Jacobian compute object not set");
        
        bool
        terminate = false;
        
        uint_t
        n_dofs          = c->n_dofs(),
        it              = 0;
        
        real_t
        dx_norm  = 0.,
        res0     = 0.,  // norm at first iteration
        res_norm = 0.;  // norm at current iteration

        VecType
        res = VecType::Zero(n_dofs),
        dx  = VecType::Zero(n_dofs),
        x1  = VecType::Zero(n_dofs);
        

        while (!terminate) {
            
            if (it%jac_update_freq == 0) {
                
                _compute->compute(c, x, res, &jac);
                _linear_solver->compute(jac);
            }
            else
                _compute->compute(c, x, res, nullptr);

            // first check for residual convergence
            res_norm = res.norm();
            
            if (verbose > 0)
                std::cout
                << std::setw(output_indent_level) << " "
                << "Iter: " << std::setw(5) << it << std::endl
                << std::setw(output_indent_level+2) << " "
                << "|| res ||_2 = "
                << std::setw(20) << res_norm << std::endl;
            
            // check for convergence of residual, otherwise solve for next step
            if (_check_residual_convergence(it, res0, res_norm)) {
                
                terminate = true;
                
                if (verbose > 0)
                    std::cout
                    << std::setw(output_indent_level+2) << " "
                    << "Terminating due to residual norm convergence." << std::endl;
            }
            else {
                
                // solve for the solution update, which is also the
                // search direction for the line search
                dx = _linear_solver->solve(res);
                
                if (_line_search) {
                    
                    dx *= -1.;
                    _line_search->compute(c, x0, dx, x1);
                    dx = x1-x0;
                }
                else
                    // use the standard NR udpate if no linear search is specified
                    x1 = x0 - dx;

                dx_norm = dx.norm();

                if (verbose > 0)
                    std::cout
                    << std::setw(output_indent_level+2) << " "
                    << "|| dx ||_2 = "
                    << std::setw(20) << dx_norm << std::endl;

                // check for the convergence of solution update
                if (dx_norm <= dx_atol) {
                    
                    terminate = true;
                    
                    if (verbose > 0)
                        std::cout
                        << std::setw(output_indent_level+2) << " "
                        << "Terminating due to solution update convergence." << std::endl;
                }
            }
            
            it++;
        }
    }
    
    
protected:
    
    LinearSolverType          *_linear_solver;

    LineSearchType            *_line_search;
    
    ComputeKernelType         *_compute;
};

} // namespace EigenWrapper
} // namespace Numerics
} // namespace MAST


#endif // __mast_numerics_nonlinear_solver_h__
