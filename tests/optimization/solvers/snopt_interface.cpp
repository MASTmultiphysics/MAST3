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
#include <mast/optimization/solvers/snopt_interface.hpp>

// Test includes
#include <test_helpers.h>

// libMesh includes
#include <libmesh/libmesh.h>

extern libMesh::LibMeshInit *p_global_init;

namespace MAST {
namespace Test {
namespace Optimization {
namespace Solvers {
namespace SNOPT {

void
optim_obj(int*    mode,
          int*    n,
          double* x,
          double* f,
          double* g,
          int*    nstate);
void
optim_con(int*    mode,
          int*    ncnln,
          int*    n,
          int*    ldJ,
          int*    needc,
          double* x,
          double* c,
          double* cJac,
          int*    nstate);

struct RosenbrockFunction {

    RosenbrockFunction(): a(1.), b(100.), obj(0.) {}
    
    real_t a, b, obj;
    std::vector<real_t> x;
    
    inline uint_t n_vars() const {return 2;}
    inline uint_t   n_eq() const {return 0;}
    inline uint_t n_ineq() const {return 0;}
    virtual void init_dvar(std::vector<real_t>& x,
                           std::vector<real_t>& xmin,
                           std::vector<real_t>& xmax) {
        
        x    = {  5.,  -5.};
        xmin = {-10., -10.};
        xmax = { 10.,  10.};
    }

    virtual void evaluate(const std::vector<real_t>& x,
                          real_t& obj,
                          bool eval_obj_grad,
                          std::vector<real_t>& obj_grad,
                          std::vector<real_t>& fvals,
                          std::vector<bool>& eval_grads,
                          std::vector<real_t>& grads) {
       
        obj = b*pow((x[1]-pow(x[0],2)),2) + pow(a-x[0], 2);

        if (eval_obj_grad) {
            
            obj_grad =
            {-2*(a-x[0])-4*b*(x[1]-pow(x[0],2))*x[0],
                2*b*(x[1]-pow(x[0],2))};
        }
    }

    inline void output(const uint_t                iter,
                       const std::vector<real_t>  &dvars,
                       real_t                     &o,
                       std::vector<real_t>        &fvals) {
        
        obj = o;
        x   = dvars;
    }
    
    MAST::Optimization::Solvers::funobj
    get_objective_evaluation_function() {
    
        return optim_obj;
    }

    MAST::Optimization::Solvers::funcon
    get_constraint_evaluation_function() {
    
        return optim_con;
    }
};


MAST::Test::Optimization::Solvers::SNOPT::RosenbrockFunction* _my_func_eval = nullptr;
unsigned int it_num = 0;

void
optim_obj(int*    mode,
          int*    n,
          double* x,
          double* f,
          double* g,
          int*    nstate) {

    //
    // make sure that the global variable has been setup
    //
    libmesh_assert(_my_func_eval);

    //
    // initialize the local variables
    //
    real_t
    obj = 0.;

    unsigned int
    n_vars  =  _my_func_eval->n_vars(),
    n_con   =  _my_func_eval->n_eq()+_my_func_eval->n_ineq();

    libmesh_assert_equal_to(*n, n_vars);

    std::vector<real_t>
    dvars   (*n,    0.),
    obj_grad(*n,    0.),
    fvals   (n_con, 0.),
    grads   (0);

    std::vector<bool>
    eval_grads(n_con);
    std::fill(eval_grads.begin(), eval_grads.end(), false);
    
    //
    // copy the dvars
    //
    for (unsigned int i=0; i<n_vars; i++)
        dvars[i] = x[i];


    _my_func_eval->evaluate(dvars,
                            obj,
                            *mode>0,       // request the derivatives of obj
                            obj_grad,
                            fvals,
                            eval_grads,
                            grads);

    //
    // now copy them back as necessary
    //
    *f  = obj;
    if (*mode > 0) {
        
        // output data to the file
        _my_func_eval->output(it_num, dvars, obj, fvals);
        it_num++;

        for (unsigned int i=0; i<n_vars; i++)
            g[i] = obj_grad[i];
    }

    if (obj > 1.e5) *mode = -1;
}


void
optim_con(int*    mode,
          int*    ncnln,
          int*    n,
          int*    ldJ,
          int*    needc,
          double* x,
          double* c,
          double* cJac,
          int*    nstate) {

    //
    // make sure that the global variable has been setup
    //
    libmesh_assert(_my_func_eval);

    //
    // initialize the local variables
    //
    real_t
    obj = 0.;

    unsigned int
    n_vars  =  _my_func_eval->n_vars(),
    n_con   =  _my_func_eval->n_eq()+_my_func_eval->n_ineq();

    libmesh_assert_equal_to(    *n, n_vars);
    libmesh_assert_equal_to(*ncnln, n_con);

    std::vector<real_t>
    dvars   (*n,    0.),
    obj_grad(*n,    0.),
    fvals   (n_con, 0.),
    grads   (n_vars*n_con, 0.);

    std::vector<bool>
    eval_grads(n_con);
    std::fill(eval_grads.begin(), eval_grads.end(), *mode>0);

    //
    // copy the dvars
    //
    for (unsigned int i=0; i<n_vars; i++)
        dvars[i] = x[i];


    _my_func_eval->evaluate(dvars,
                            obj,
                            false,       // request the derivatives of obj
                            obj_grad,
                            fvals,
                            eval_grads,
                            grads);

    //
    // now copy them back as necessary
    //
    // first the constraint functions
    //
    for (unsigned int i=0; i<n_con; i++)
        c[i] = fvals[i];

    if (*mode > 0) {
        //
        // next, the constraint gradients
        //
        for (unsigned int i=0; i<n_con*n_vars; i++)
            cJac[i] = grads[i];
    }
    
    if (obj > 1.e5) *mode = -1;
}


TEST_CASE("snopt_interface",
          "[Optimization][Solvers][SNOPT]") {
    
    MAST::Test::Optimization::Solvers::SNOPT::RosenbrockFunction f;
    MAST::Test::Optimization::Solvers::SNOPT::_my_func_eval = &f;
#if MAST_ENABLE_SNOPT == 1

    MAST::Optimization::Solvers::SNOPTInterface<RosenbrockFunction>
    opt(p_global_init->comm());
    opt.set_function_evaluation(f);
    opt.init();
    
    opt.optimize();
    
    CHECK(f.obj == Catch::Detail::Approx(0.).margin(1.e-5));
    CHECK_THAT(f.x, Catch::Approx(std::vector<real_t>({1., 1.})).epsilon(1.e-2));
#endif
}

} // namespace SNOPT
} // namespace Solvers
} // namespace Optimization
} // namespace Test
} // namespace MAST


