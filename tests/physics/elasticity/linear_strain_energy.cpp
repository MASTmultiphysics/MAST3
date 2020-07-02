
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/fe/libmesh/fe.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/linear_strain_energy.hpp>

// Test includes
#include <fe/quad_derivatives.hpp>
#include <test_helpers.h>

// libMesh includes
#include <libmesh/elem.h>

extern libMesh::LibMeshInit* p_global_init;

struct Context {
    Context(): elem(nullptr), qp(-1), s(-1) {}
    uint_t elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    const libMesh::Elem* elem;
    uint_t qp;
    uint_t s;
};


template <typename BasisScalarType,
          typename NodalScalarType,
          typename SolScalarType,
          uint_t   Dim>
struct Traits {

    using scalar_t          = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using vector_t          = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t          = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using quadrature_t      = MAST::Quadrature::libMeshWrapper::Quadrature<BasisScalarType, 2>;
    using fe_basis_t        = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, Dim>;
    using fe_shape_t        = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, Dim, Dim, fe_basis_t>;
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, Dim, Dim, Context, fe_shape_t>;
    using modulus_t         = typename MAST::Base::ScalarConstant<SolScalarType>;
    using nu_t              = typename MAST::Base::ScalarConstant<SolScalarType>;
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, Dim, modulus_t, nu_t, Context>;
    using energy_t          = typename MAST::Physics::Elasticity::LinearContinuum::StrainEnergy<fe_var_t, prop_t, Dim, Context>;
};


template <typename Traits>
struct ElemOps {
  
    ElemOps():
    q        (new typename Traits::quadrature_t(libMesh::QGAUSS, libMesh::FOURTH)),
    fe       (new typename Traits::fe_basis_t(libMesh::FEType(libMesh::FIRST, libMesh::LAGRANGE))),
    fe_deriv (new typename Traits::fe_shape_t),
    fe_var   (new typename Traits::fe_var_t),
    E        (new typename Traits::modulus_t(72.e9)),
    nu       (new typename Traits::nu_t(0.33)),
    prop     (new typename Traits::prop_t),
    strain_e (new typename Traits::energy_t) {
        
        // initialize the shape function derivatives wrt reference coordinates
        fe->set_compute_dphi_dxi(true);
        
        // initialize the shape function derivatives wrt spatial coordinates
        fe_deriv->set_compute_dphi_dx(true);
        fe_deriv->set_compute_detJ(true);
        fe_deriv->set_compute_detJxW(true);
        fe_deriv->set_compute_Jac_inverse(true);
        fe_deriv->set_fe_basis(*fe);

        // initialize the variable data
        fe_var->set_compute_du_dx(true);
        fe_var->set_fe_shape_data(*fe_deriv);
        
        prop->set_modulus_and_nu(*E, *nu);
        
        strain_e->set_section_property(*prop);
        strain_e->set_fe_var_data(*fe_var);
    }
    
    virtual ~ElemOps() {}

    inline uint_t n_dofs() const { return strain_e->n_dofs();}
    
    inline void init(const libMesh::Elem* e) {
        
        c.elem = e;
        fe->reinit(*e, *q);
        fe_deriv->reinit(c);
    }
    
    inline void compute(const typename Traits::vector_t& sol,
                        typename Traits::vector_t& res,
                        typename Traits::matrix_t* jac=nullptr) {

        fe_var->init(c, sol);
        strain_e->compute(c, res, jac);
    }

    template <typename ScalarFieldType>
    inline void derivative(const ScalarFieldType& f,
                           typename Traits::vector_t& res,
                           typename Traits::matrix_t* jac = nullptr) {
        
        strain_e->derivative(c, f, res, jac);
    }
    
    std::unique_ptr<typename Traits::quadrature_t>   q;
    std::unique_ptr<typename Traits::fe_basis_t>     fe;
    std::unique_ptr<typename Traits::fe_shape_t>     fe_deriv;
    std::unique_ptr<typename Traits::fe_var_t>       fe_var;
    std::unique_ptr<typename Traits::modulus_t>      E;
    std::unique_ptr<typename Traits::nu_t>           nu;
    std::unique_ptr<typename Traits::prop_t>         prop;
    std::unique_ptr<typename Traits::energy_t>       strain_e;
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



TEST_CASE("linear_strain_energy",
          "[2D][QUAD4][Elasticity][Linear][StrainEnergy]") {
    
    const uint_t
    n_basis = 4;
    
    Eigen::Matrix<real_t, 4, 1>
    x_vec,
    y_vec,
    Nvec,
    dNvec_dxi,
    dNvec_deta,
    dNvec_dx,
    dNvec_dy;
    
    std::vector<real_t>
    vec(4, 0.);
    
    real_t
    xi,
    eta,
    u,
    du_dx,
    du_dy,
    J_det;
    
    Eigen::Matrix<real_t, 2, 2>
    Jac,
    Jac_inv;
    
    Eigen::Matrix<real_t, 3, 1>
    nvec;
    
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
    
    // residual sensitivity wrt E
    {
        res.setZero();
        jac.setZero();
        e_ops.derivative(*e_ops.E, res, &jac);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::modulus_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.E, sol, res_cs, jac_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(res),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(res_cs)));
    }

    // residual sensitivity wrt nu
    {
        res.setZero();
        jac.setZero();
        e_ops.derivative(*e_ops.nu, res, &jac);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::modulus_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.nu, sol, res_cs, jac_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(res),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(res_cs)));
    }

    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}

