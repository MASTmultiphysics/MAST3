
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/optimization/solvers/gcmma_interface.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Optimization {
namespace Solvers {
namespace GCMMA {

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
};



TEST_CASE("gcmma_interface",
          "[Optimization][Solvers][GCMMA]") {
    
    MAST::Test::Optimization::Solvers::GCMMA::RosenbrockFunction f;
#if MAST_ENABLE_GCMMA == 1

    MAST::Optimization::Solvers::GCMMAInterface<RosenbrockFunction>
    opt;
    opt.set_function_evaluation(f);
    
    opt.optimize();
    
    CHECK(f.obj == Catch::Detail::Approx(0.).margin(1.e-5));
    CHECK_THAT(f.x, Catch::Approx(std::vector<real_t>({1., 1.})).epsilon(1.e-2));
#endif
}

} // namespace GCMMA
} // namespace Solvers
} // namespace Optimization
} // namespace Test
} // namespace MAST


