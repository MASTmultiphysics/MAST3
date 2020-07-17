
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/shell_face_pressure_load.hpp>

// Test includes
#include <test_helpers.h>

// libMesh includes
#include <libmesh/elem.h>

extern libMesh::LibMeshInit* p_global_init;

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace FlatShellFacePressureLoad {

struct Context {
    Context(): elem(nullptr), qp(-1) {}
    uint_t elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    inline bool elem_is_quad() const {return (elem->type() == libMesh::QUAD4 ||
                                              elem->type() == libMesh::QUAD8 ||
                                              elem->type() == libMesh::QUAD9);}
    const libMesh::Elem* elem;
    uint_t qp;
};


template <typename BasisScalarType,
          typename NodalScalarType,
          typename SolScalarType,
          uint_t   Dim>
struct Traits {

    using scalar_t     = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using vector_t     = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t     = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using quadrature_t = MAST::Quadrature::libMeshWrapper::Quadrature<BasisScalarType, 2>;
    using fe_basis_t   = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, Dim>;
    using fe_shape_t   = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, Dim, Dim, fe_basis_t>;
    using fe_data_t    = typename MAST::FEBasis::libMeshWrapper::FEData<Dim, fe_basis_t, fe_shape_t>;
    using fe_var_t     = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 1, Dim, Context, fe_shape_t>;
    using pressure_t   = typename MAST::Base::ScalarConstant<SolScalarType>;
    using press_load_t = typename MAST::Physics::Elasticity::ShellFacePressureLoad<fe_var_t, pressure_t, Context>;
};


template <typename Traits>
struct ElemOps {
  
    ElemOps():
    fe_data  (new typename Traits::fe_data_t),
    fe_var   (new typename Traits::fe_var_t),
    p        (new typename Traits::pressure_t(2.e3)),
    press_e  (new typename Traits::press_load_t) {
        
        // initialize the shape function derivatives wrt reference coordinates
        fe_data->init(libMesh::FOURTH, libMesh::QGAUSS,
                      libMesh::FIRST, libMesh::LAGRANGE);
        
        // initialize the shape function derivatives wrt spatial coordinates
        fe_data->fe_basis().set_compute_dphi_dxi(true);

        fe_data->fe_derivative().set_compute_detJ(true);
        fe_data->fe_derivative().set_compute_detJxW(true);

        // initialize the variable data
        fe_var->set_fe_shape_data(fe_data->fe_derivative());
        
        press_e->set_pressure(*p);
        press_e->set_fe_var_data(*fe_var, 1);
    }
    
    virtual ~ElemOps() {}

    inline uint_t n_dofs() const { return press_e->n_dofs();}
    
    inline void init(const libMesh::Elem* e) {
        
        c.elem = e;
        fe_data->reinit(c);
    }
    
    inline void compute(const typename Traits::vector_t& sol,
                        typename Traits::vector_t& res,
                        typename Traits::matrix_t* jac=nullptr) {

        fe_var->init(c, sol);
        press_e->compute(c, res, jac);
    }

    template <typename ScalarFieldType>
    inline void derivative(const ScalarFieldType& f,
                           typename Traits::vector_t& res,
                           typename Traits::matrix_t* jac = nullptr) {
        
        press_e->derivative(c, f, res, jac);
    }

    std::unique_ptr<typename Traits::fe_data_t>      fe_data;
    std::unique_ptr<typename Traits::fe_var_t>       fe_var;
    std::unique_ptr<typename Traits::pressure_t>     p;
    std::unique_ptr<typename Traits::press_load_t>   press_e;
    Context                                          c;
};


template <typename ScalarConstantType,
          typename Traits,
          typename TraitsComplex>
inline void complex_step_derivative(ElemOps<TraitsComplex>          &e_ops_c,
                                    const libMesh::Elem             *e,
                                    ScalarConstantType              &f,
                                    const typename Traits::vector_t &sol,
                                    typename Traits::vector_t       &res,
                                    typename Traits::matrix_t       &jac)  {
    

    typename TraitsComplex::vector_t
    sol_c,
    res_c;

    typename TraitsComplex::matrix_t
    jac_c;

    e_ops_c.init(e);
    
    sol_c = sol.template cast<complex_t>();
    res_c = TraitsComplex::vector_t::Zero(e_ops_c.n_dofs());
    jac_c = TraitsComplex::matrix_t::Zero(e_ops_c.n_dofs(), e_ops_c.n_dofs());

    // add perturbation to parameter
    f() += complex_t(0., ComplexStepDelta);
    
    e_ops_c.compute(sol_c, res_c, &jac_c);
    
    res = res_c.imag()/ComplexStepDelta;
    jac = jac_c.imag()/ComplexStepDelta;
}



TEST_CASE("flat_shell_face_linear_pressure_load",
          "[2D][QUAD4][Elasticity][Linear][ShellFace][PressureLoad]") {
    
    const uint_t
    n_basis = 4;
    
    Eigen::Matrix<real_t, 4, 1>
    x_vec,
    y_vec;
    
    x_vec << -1., 1., 1., -1.;
    y_vec << -1., -1., 1., 1.;
    
    // randomly perturb the coordinates
    x_vec += 0.1 * Eigen::Matrix<real_t, 4, 1>::Random();
    y_vec += 0.1 * Eigen::Matrix<real_t, 4, 1>::Random();
    
    std::unique_ptr<libMesh::Elem>
    e(libMesh::Elem::build(libMesh::QUAD4).release());
    
    std::vector<libMesh::Node*> nodes(4, nullptr);
    for (uint_t i=0; i<e->n_nodes(); i++) {
        nodes[i] = libMesh::Node::build(libMesh::Point(x_vec(i), y_vec(i)), i).release();
        e->set_node(i) = nodes[i];
    }
    
    using traits_t         = Traits<real_t, real_t, real_t, 2>;
    
    typename traits_t::vector_t
    sol,
    res,
    res_cs;

    typename traits_t::matrix_t
    jac,
    jac_cs;

    
    ElemOps<traits_t> e_ops;
    e_ops.init(e.get());
    
    sol    = 0.1 * traits_t::vector_t::Random(e_ops.n_dofs());
    res    = traits_t::vector_t::Zero(e_ops.n_dofs());
    jac    = traits_t::matrix_t::Zero(e_ops.n_dofs(), e_ops.n_dofs());
    jac_cs = traits_t::matrix_t::Zero(e_ops.n_dofs(), e_ops.n_dofs());
    
    e_ops.compute(sol, res, &jac);
    
    using traits_complex_t = Traits<real_t, real_t, complex_t, 2>;
    
    // compute the complex-step Jacobian
    {
        for (uint_t i=0; i<sol.size(); i++) {
            
            typename traits_complex_t::vector_t
            sol_c,
            res_c;
            
            ElemOps<traits_complex_t> e_ops_c;
            e_ops_c.init(e.get());
            
            sol_c = sol.cast<complex_t>();
            res_c = traits_complex_t::vector_t::Zero(e_ops.n_dofs());
            
            // complex perturbation to dof
            sol_c(i) += complex_t(0., ComplexStepDelta);
            
            e_ops_c.compute(sol_c, res_c);
            
            jac_cs.col(i) = res_c.imag()/ComplexStepDelta;
        }
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(jac),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(jac_cs)));
    }
    
    // residual sensitivity wrt p
    {
        res.setZero();
        jac.setZero();
        e_ops.derivative(*e_ops.p, res, &jac);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::pressure_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.p, sol, res_cs, jac_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(res),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(res_cs)));
    }
    
    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}

} // namespace FlatShellFacePressureLoad
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


