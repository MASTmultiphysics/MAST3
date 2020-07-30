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

#ifndef __mast_gcmma_optimization_interface_h__
#define __mast_gcmma_optimization_interface_h__

// C++ includes
#include <iomanip>

// MAST includes
#include <mast/base/mast_config.h>
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// TIMPI includes
#include <timpi/communicator.h>
#include <timpi/parallel_implementation.h>


extern "C" {
extern void raasta_(int *M, int *N,
                    double *RAA0, double *RAA,
                    double *XMIN, double *XMAX,
                    double *DF0DX, double *DFDX);
extern void asympg_(int *ITER, int *M, int *N,
                    double *ALBEFA, double *GHINIT,
                    double *GHDECR, double *GHINCR,
                    double *XVAL, double *XMIN, double *XMAX,
                    double *XOLD1, double *XOLD2,
                    double *XLOW, double *XUPP,
                    double *ALFA, double *BETA);
extern void mmasug_(int *ITER, int *M, int *N, double *GEPS, int *IYFREE,
                    double *XVAL, double *XMMA,
                    double *XMIN, double *XMAX,
                    double *XLOW, double *XUPP,
                    double *ALFA, double *BETA,
                    double *A, double *B, double *C, double *Y, double *Z,
                    double *RAA0, double *RAA, double *ULAM,
                    double *F0VAL, double *FVAL,
                    double *F0APP, double *FAPP,
                    double *FMAX, double *DF0DX, double *DFDX,
                    double *P, double *Q, double *P0, double *Q0,
                    double *UU, double *GRADF, double *DSRCH, double *HESSF);
extern void conser_(int *M, int *ICONSE,
                    double *GEPS, double *F0NEW, double *F0APP,
                    double *FNEW, double *FAPP);
extern void raaupd_(int *M, int *N, double *GEPS,
                    double *XMMA, double *XVAL,
                    double *XMIN, double *XMAX,
                    double *XLOW, double *XUPP,
                    double *F0NEW, double *FNEW,
                    double *F0APP, double *FAPP,
                    double *RAA0, double *RAA);
extern void xupdat_(int *N, int *ITER,
                    double *XMMA, double *XVAL,
                    double *XOLD1, double *XOLD2);
extern void fupdat_(int *M, double *F0NEW, double *FNEW,
                    double *F0VAL, double *FVAL);
}


namespace MAST {
namespace Optimization {
namespace Solvers {



template <typename FunctionEvaluationType>
class GCMMAInterface {
    
public:
    
    GCMMAInterface(TIMPI::Communicator &comm):
    _comm                         (comm),
    constr_penalty                (5.e1),
    initial_rel_step              (1.e-2),
    asymptote_reduction           (0.7),
    asymptote_expansion           (1.2),
    rel_change_tol                (1.e-8),
    max_inner_iters               (15),
    max_iters                     (1000),
    n_rel_change_iters            (5),
    write_internal_iteration_data (false),
    _feval                        (nullptr)
    { }

    
    virtual ~GCMMAInterface()
    { }

    real_t           constr_penalty;
    real_t           initial_rel_step;
    real_t           asymptote_reduction;
    real_t           asymptote_expansion;
    real_t           rel_change_tol;
    uint_t           max_inner_iters;
    uint_t           max_iters;
    uint_t           n_rel_change_iters;
    bool             write_internal_iteration_data;
    
    inline void set_function_evaluation(FunctionEvaluationType& feval) {
        
        Assert0(!_feval, "Function evaluation object already set");
        
        _feval = &feval;
    }
    
    
    inline void optimize() {
        
#if MAST_ENABLE_GCMMA == 1
        
        Assert0(_feval, "Function evaluation object not set");
                
        int
        N                  = _feval->n_vars(),
        M                  = _feval->n_eq() + _feval->n_ineq();
        
        Assert1(N > 0, N, "Design variables must be greater than 0");
        
        std::vector<real_t>  XVAL(N, 0.), XOLD1(N, 0.), XOLD2(N, 0.),
        XMMA(N, 0.), XMIN(N, 0.), XMAX(N, 0.), XLOW(N, 0.), XUPP(N, 0.),
        ALFA(N, 0.), BETA(N, 0.), DF0DX(N, 0.),
        A(M, 0.), B(M, 0.), C(M, 0.), Y(M, 0.), RAA(M, 0.), ULAM(M, 0.),
        FVAL(M, 0.), FAPP(M, 0.), FNEW(M, 0.), FMAX(M, 0.),
        DFDX(M*N, 0.), P(M*N, 0.), Q(M*N, 0.), P0(N, 0.), Q0(N, 0.),
        UU(M, 0.), GRADF(M, 0.), DSRCH(M, 0.), HESSF(M*(M+1)/2, 0.),
        f0_iters(n_rel_change_iters);
        
        std::vector<int> IYFREE(M, 0);
        std::vector<bool> eval_grads(M, false);
        
        real_t
        ALBEFA  = 0.1,
        GHINIT  = initial_rel_step,
        GHDECR  = asymptote_reduction,
        GHINCR  = asymptote_expansion,
        F0VAL   = 0.,
        F0NEW   = 0.,
        F0APP   = 0.,
        RAA0    = 0.,
        Z       = 0.,
        GEPS    = rel_change_tol;
        
        
        /*C********+*********+*********+*********+*********+*********+*********+
         C
         C  The meaning of some of the scalars and vectors in the program:
         C
         C     N  = Complex of variables x_j in the problem.
         C     M  = Complex of constraints in the problem (not including
         C          the simple upper and lower bounds on the variables).
         C ALBEFA = Relative spacing between asymptote and mode limit. Lower value
         C          will cause the move limit (alpha,beta) to move closer to asymptote
         C          values (l, u).
         C GHINIT = Initial asymptote setting. For the first two iterations the
         C          asymptotes (l, u) are defined based on offsets from the design
         C          point as this fraction of the design variable bounds, ie.
         C              l_j   =   x_j^k  - GHINIT * (x_j^max - x_j^min)
         C              u_j   =   x_j^k  + GHINIT * (x_j^max - x_j^min)
         C GHDECR = Fraction by which the asymptote is reduced for oscillating
         C          changes in design variables based on three consecutive iterations
         C GHINCR = Fraction by which the asymptote is increased for non-oscillating
         C          changes in design variables based on three consecutive iterations
         C INNMAX = Maximal number of inner iterations within each outer iter.
         C          A reasonable choice is INNMAX=10.
         C  ITER  = Current outer iteration number ( =1 the first iteration).
         C  GEPS  = Tolerance parameter for the constraints.
         C          (Used in the termination criteria for the subproblem.)
         C
         C   XVAL(j) = Current value of the variable x_j.
         C  XOLD1(j) = Value of the variable x_j one iteration ago.
         C  XOLD2(j) = Value of the variable x_j two iterations ago.
         C   XMMA(j) = Optimal value of x_j in the MMA subproblem.
         C   XMIN(j) = Original lower bound for the variable x_j.
         C   XMAX(j) = Original upper bound for the variable x_j.
         C   XLOW(j) = Value of the lower asymptot l_j.
         C   XUPP(j) = Value of the upper asymptot u_j.
         C   ALFA(j) = Lower bound for x_j in the MMA subproblem.
         C   BETA(j) = Upper bound for x_j in the MMA subproblem.
         C    F0VAL  = Value of the objective function f_0(x)
         C   FVAL(i) = Value of the i:th constraint function f_i(x).
         C  DF0DX(j) = Derivative of f_0(x) with respect to x_j.
         C   FMAX(i) = Right hand side of the i:th constraint.
         C   DFDX(k) = Derivative of f_i(x) with respect to x_j,
         C             where k = (j-1)*M + i.
         C      P(k) = Coefficient p_ij in the MMA subproblem, where
         C             k = (j-1)*M + i.
         C      Q(k) = Coefficient q_ij in the MMA subproblem, where
         C             k = (j-1)*M + i.
         C     P0(j) = Coefficient p_0j in the MMA subproblem.
         C     Q0(j) = Coefficient q_0j in the MMA subproblem.
         C      B(i) = Right hand side b_i in the MMA subproblem.
         C    F0APP  = Value of the approximating objective function
         C             at the optimal soultion of the MMA subproblem.
         C   FAPP(i) = Value of the approximating i:th constraint function
         C             at the optimal soultion of the MMA subproblem.
         C    RAA0   = Parameter raa_0 in the MMA subproblem.
         C    RAA(i) = Parameter raa_i in the MMA subproblem.
         C      Y(i) = Value of the "artificial" variable y_i.
         C      Z    = Value of the "minimax" variable z.
         C      A(i) = Coefficient a_i for the variable z.
         C      C(i) = Coefficient c_i for the variable y_i.
         C   ULAM(i) = Value of the dual variable lambda_i.
         C  GRADF(i) = Gradient component of the dual objective function.
         C  DSRCH(i) = Search direction component in the dual subproblem.
         C  HESSF(k) = Hessian matrix component of the dual function.
         C IYFREE(i) = 0 for dual variables which are fixed to zero in
         C               the current subspace of the dual subproblem,
         C           = 1 for dual variables which are "free" in
         C               the current subspace of the dual subproblem.
         C
         C********+*********+*********+*********+*********+*********+*********+*/
        
        
        /*
         *  The USER should now give values to the parameters
         *  M, N, GEPS, XVAL (starting point),
         *  XMIN, XMAX, FMAX, A and C.
         */
        // _initi(M,N,GEPS,XVAL,XMIN,XMAX,FMAX,A,C);
        // Assumed:  FMAX == A
        _feval->init_dvar(XVAL, XMIN, XMAX);
        // set the value of C[i] to be very large numbers
        real_t max_x = 0.;
        for (uint_t i=0; i<N; i++)
            if (max_x < fabs(XVAL[i]))
                max_x = fabs(XVAL[i]);
        
        int INNMAX=max_inner_iters, ITER=0, ITE=0, INNER=0, ICONSE=0;
        /*
         *  The outer iterative process starts.
         */
        bool terminate = false, inner_terminate=false;
        while (!terminate) {
            
            std::fill(C.begin(), C.end(), std::max(1.e0*max_x, constr_penalty));
            GHINIT  = initial_rel_step,
            GHDECR  = asymptote_reduction,
            GHINCR  = asymptote_expansion,
            
            ITER=ITER+1;
            ITE=ITE+1;
            /*
             *  The USER should now calculate function values and gradients
             *  at XVAL. The result should be put in F0VAL,DF0DX,FVAL,DFDX.
             */
            std::fill(eval_grads.begin(), eval_grads.end(), true);
            _feval->evaluate(XVAL,
                             F0VAL, true, DF0DX,
                             FVAL, eval_grads, DFDX);
            if (ITER == 1)
                // output the very first iteration
                _feval->output(0, XVAL, F0VAL, FVAL);
            
            /*
             *  RAA0,RAA,XLOW,XUPP,ALFA and BETA are calculated.
             */
            raasta_(&M, &N, &RAA0, &RAA[0], &XMIN[0], &XMAX[0], &DF0DX[0], &DFDX[0]);
            asympg_(&ITER, &M, &N, &ALBEFA, &GHINIT, &GHDECR, &GHINCR,
                    &XVAL[0], &XMIN[0], &XMAX[0], &XOLD1[0], &XOLD2[0],
                    &XLOW[0], &XUPP[0], &ALFA[0], &BETA[0]);
            /*
             *  The inner iterative process starts.
             */
            
            // write the asymptote data for the inneriterations
            if (write_internal_iteration_data)
                _output_iteration_data(ITER, XVAL, XMIN, XMAX, XLOW, XUPP, ALFA, BETA);
            
            INNER=0;
            inner_terminate = false;
            while (!inner_terminate) {
                
                /*
                 *  The subproblem is generated and solved.
                 */
                mmasug_(&ITER, &M, &N, &GEPS, &IYFREE[0], &XVAL[0], &XMMA[0],
                        &XMIN[0], &XMAX[0], &XLOW[0], &XUPP[0], &ALFA[0], &BETA[0],
                        &A[0], &B[0], &C[0], &Y[0], &Z, &RAA0, &RAA[0], &ULAM[0],
                        &F0VAL, &FVAL[0], &F0APP, &FAPP[0], &FMAX[0], &DF0DX[0], &DFDX[0],
                        &P[0], &Q[0], &P0[0], &Q0[0], &UU[0], &GRADF[0], &DSRCH[0], &HESSF[0]);
                /*
                 *  The USER should now calculate function values at XMMA.
                 *  The result should be put in F0NEW and FNEW.
                 */
                std::fill(eval_grads.begin(), eval_grads.end(), false);
                _feval->evaluate(XMMA,
                                 F0NEW, false, DF0DX,
                                 FNEW, eval_grads, DFDX);
                
                ///////////////////////////////////////////////////////////////
                // if the solution is poor, backtrack
                std::vector<real_t> XMMA_new(XMMA);
                real_t frac = 0.5;
                while (M && FNEW[0] > 1.e2) {
                    std::cout << "*** Backtracking: frac = "
                    << frac
                    << "  constr: " << FNEW[0]
                    << std::endl;
                    for (uint_t i=0; i<XMMA.size(); i++)
                        XMMA_new[i] = XOLD1[i] + frac*(XMMA[i]-XOLD1[i]);
                    
                    _feval->evaluate(XMMA_new,
                                     F0NEW, false, DF0DX,
                                     FNEW, eval_grads, DFDX);
                    frac *= frac;
                }
                for (uint_t i=0; i<XMMA.size(); i++)
                    XMMA[i] = XMMA_new[i];
                ///////////////////////////////////////////////////////////////
                
                if (INNER >= INNMAX) {
                    std::cout
                    << "** Max Inner Iter Reached: Terminating! Inner Iter = "
                    << INNER << std::endl;
                    inner_terminate = true;
                }
                else {
                    /*
                     *  It is checked if the approximations were conservative.
                     */
                    conser_( &M, &ICONSE, &GEPS, &F0NEW, &F0APP, &FNEW[0], &FAPP[0]);
                    if (ICONSE == 1) {
                        std::cout
                        << "** Conservative Solution: Terminating! Inner Iter = "
                        << INNER << std::endl;
                        inner_terminate = true;
                    }
                    else {
                        /*
                         *  The approximations were not conservative, so RAA0 and RAA
                         *  are updated and one more inner iteration is started.
                         */
                        INNER=INNER+1;
                        raaupd_( &M, &N, &GEPS, &XMMA[0], &XVAL[0],
                                &XMIN[0], &XMAX[0], &XLOW[0], &XUPP[0],
                                &F0NEW, &FNEW[0], &F0APP, &FAPP[0], &RAA0, &RAA[0]);
                    }
                }
            }
            
            /*
             *  The inner iterative process has terminated, which means
             *  that an outer iteration has been completed.
             *  The variables are updated so that XVAL stands for the new
             *  outer iteration point. The fuction values are also updated.
             */
            xupdat_( &N, &ITER, &XMMA[0], &XVAL[0], &XOLD1[0], &XOLD2[0]);
            fupdat_( &M, &F0NEW, &FNEW[0], &F0VAL, &FVAL[0]);
            /*
             *  The USER may now write the current solution.
             */
            _feval->output(ITER, XVAL, F0VAL, FVAL);
            f0_iters[(ITE-1)%n_rel_change_iters] = F0VAL;
            
            /*
             *  One more outer iteration is started as long as
             *  ITE is less than MAXITE:
             */
            if (ITE == max_iters) {
                std::cout
                << "GCMMA: Reached maximum iterations, terminating! "
                << std::endl;
                terminate = true;
            }
            
            // relative change in objective
            bool rel_change_conv = true;
            real_t f0_curr = f0_iters[n_rel_change_iters-1];
            
            for (uint_t i=0; i<n_rel_change_iters-1; i++) {
                if (f0_curr > sqrt(GEPS))
                    rel_change_conv = (rel_change_conv &&
                                       fabs(f0_iters[i]-f0_curr)/fabs(f0_curr) < GEPS);
                else
                    rel_change_conv = (rel_change_conv &&
                                       fabs(f0_iters[i]-f0_curr) < GEPS);
            }
            if (rel_change_conv) {
                std::cout
                << "GCMMA: Converged relative change tolerance, terminating! "
                << std::endl;
                terminate = true;
            }
            // tell all processors about the decision here
            _comm.broadcast(terminate, 0);
        }
        
#endif //MAST_ENABLE_GCMMA == 1
    }
    
    
private:
    
    TIMPI::Communicator  &_comm;
    
    inline void _output_iteration_data(uint_t i,
                                       const std::vector<real_t>& XVAL,
                                       const std::vector<real_t>& XMIN,
                                       const std::vector<real_t>& XMAX,
                                       const std::vector<real_t>& XLOW,
                                       const std::vector<real_t>& XUPP,
                                       const std::vector<real_t>& ALFA,
                                       const std::vector<real_t>& BETA) {

        Assert0(_feval, "Evaluation function not initialized");
        Assert2(XVAL.size() == _feval->n_vars(),
                XVAL.size(), _feval->n_vars(),
                "Incorrect vector size");
        Assert2(XMIN.size() == _feval->n_vars(),
                XMIN.size(), _feval->n_vars(),
                "Incorrect vector size");
        Assert2(XMAX.size() == _feval->n_vars(),
                XMAX.size(), _feval->n_vars(),
                "Incorrect vector size");
        Assert2(XLOW.size() == _feval->n_vars(),
                XLOW.size(), _feval->n_vars(),
                "Incorrect vector size");
        Assert2(XUPP.size() == _feval->n_vars(),
                XUPP.size(), _feval->n_vars(),
                "Incorrect vector size");
        Assert2(ALFA.size() == _feval->n_vars(),
                ALFA.size(), _feval->n_vars(),
                "Incorrect vector size");
        Assert2(BETA.size() == _feval->n_vars(),
                BETA.size(), _feval->n_vars(),
                "Incorrect vector size");

        std::cout
        << "****************************************************\n"
        << "             GCMMA: ASYMPTOTE DATA                  \n"
        << "****************************************************\n"
        << std::setw(5) << "Iter: " << i << std::endl
        << std::setw(5) << "DV"
        << std::setw(20) << "XMIN"
        << std::setw(20) << "XLOW"
        << std::setw(20) << "ALFA"
        << std::setw(20) << "X"
        << std::setw(20) << "BETA"
        << std::setw(20) << "XUP"
        << std::setw(20) << "XMAX" << std::endl;

        for (uint_t j=0; j<_feval->n_vars(); j++)
            std::cout
            << std::setw(5) << j
            << std::setw(20) << XMIN[j]
            << std::setw(20) << XLOW[j]
            << std::setw(20) << ALFA[j]
            << std::setw(20) << XVAL[j]
            << std::setw(20) << BETA[j]
            << std::setw(20) << XUPP[j]
            << std::setw(20) << XMAX[j] << std::endl;
    }
    
    inline void
    _evaluate_wrapper(const std::vector<real_t> &x,
                      real_t                    &obj,
                      bool                      eval_obj_grad,
                      std::vector<real_t>       &obj_grad,
                      std::vector<real_t>       &fvals,
                      std::vector<bool>         &eval_grads,
                      std::vector<real_t>       &grads) {
        
        // rank 0 will broadcase the DV values to all ranks
        _comm.broadcast(x, 0, true);
        _comm.broadcast(eval_obj_grad, 0);
        _comm.broadcast(eval_grads, 0, true);
        
        _feval->evaluate(x,
                         obj,
                         eval_obj_grad,
                         obj_grad,
                         fvals,
                         eval_grads,
                         grads);
        
        // make sure that all data returned is consistent across
        // processors
        Assert0(_comm.verify(obj),
                "Objective function has different values on ranks");
        Assert0(_comm.verify(obj_grad),
                "Objective function gradient has different values on ranks");
        Assert0(_comm.verify(fvals),
                "Constraint functions has different values on ranks");
        Assert0(_comm.verify(grads),
                "Constraint function gradient has different values on ranks");
    }
    
    FunctionEvaluationType *_feval;
};

} // namespace Solvers
} // namespace Optimization
} // namespace MAST

#endif // __mast_gcmma_optimization_interface_h__
