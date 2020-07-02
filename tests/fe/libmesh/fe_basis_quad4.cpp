
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/fe/libmesh/fe.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/fe_var_data.hpp>

// Test includes
#include <fe/quad_derivatives.hpp>
#include <test_helpers.h>

// libMesh includes
#include <libmesh/elem.h>

extern libMesh::LibMeshInit* p_global_init;

struct Context {
    Context(): elem(nullptr), qp(-1), s(-1) {}
    inline uint_t elem_dim() const {return elem->dim();}
    inline uint_t  n_nodes() const {return elem->n_nodes();}
    inline real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    inline bool elem_is_quad() const {return (elem->type() == libMesh::QUAD4 ||
                                              elem->type() == libMesh::QUAD8 ||
                                              elem->type() == libMesh::QUAD9);}
    libMesh::Elem* elem;
    uint_t qp;
    uint_t s;
};


void test_quad4_fe_data(const uint_t mode) {
    
    const uint_t
    n_basis = 4;
    
    Eigen::Matrix<real_t, 4, 1>
    x_vec,
    y_vec,
    u_vec,
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
    nvec,
    tvec;
    
    x_vec << -1., 1., 1., -1.;
    y_vec << -1., -1., 1., 1.;
    u_vec << 1., 2., 3., 4.;
    
    // randomly perturb the coordinates
    x_vec += 0.1 * Eigen::Matrix<real_t, 4, 1>::Random();
    y_vec += 0.1 * Eigen::Matrix<real_t, 4, 1>::Random();
    u_vec += 0.1 * Eigen::Matrix<real_t, 4, 1>::Random();
    
    std::unique_ptr<libMesh::Elem>
    e(libMesh::Elem::build(libMesh::QUAD4).release());
    
    std::vector<libMesh::Node*> nodes(4, nullptr);
    for (uint_t i=0; i<e->n_nodes(); i++) {
        nodes[i] = libMesh::Node::build(libMesh::Point(x_vec(i), y_vec(i)), i).release();
        e->set_node(i) = nodes[i];
    }
    
    using quadrature_t = MAST::Quadrature::libMeshWrapper::Quadrature<real_t, 2>;
    using fe_t         = MAST::FEBasis::libMeshWrapper::FEBasis<real_t, 2>;
    using fe_deriv_t   = MAST::FEBasis::Evaluation::FEShapeDerivative<real_t, real_t, 2, 2, fe_t>;
    using fe_var_t     = MAST::FEBasis::FEVarData<real_t, real_t, real_t, 1, 2, Context, fe_deriv_t>;
    
    std::unique_ptr<quadrature_t>
    q(new quadrature_t(libMesh::QGAUSS, libMesh::FOURTH));
    
    std::unique_ptr<fe_t>
    fe(new fe_t(libMesh::FEType(libMesh::FIRST, libMesh::LAGRANGE)));
    
    std::unique_ptr<fe_deriv_t>
    fe_deriv(new fe_deriv_t);
    
    std::unique_ptr<fe_var_t>
    fe_var(new fe_var_t);
    
    Context c;
    c.elem = e.get();
    
    // initialize the shape function derivatives wrt reference coordinates
    fe->set_compute_dphi_dxi(true);
    fe->reinit(*e, *q);
    
    // initialize the shape function derivatives wrt spatial coordinates
    fe_deriv->set_compute_dphi_dx(true);
    fe_deriv->set_compute_detJ(true);
    fe_deriv->set_compute_Jac_inverse(true);
    fe_deriv->set_fe_basis(*fe);
    if (mode == 0)
        fe_deriv->reinit(c);
    else {
        fe_deriv->set_compute_normal(true);
        fe_deriv->reinit_for_side(c, mode-1);
    }
    
    // initialize the variable data
    fe_var->set_compute_du_dx(true);
    fe_var->set_fe_shape_data(*fe_deriv);
    fe_var->init(c, u_vec);
    
    // iterate over points and check the data for accuracy
    for (uint_t i=0; i<fe->n_q_points(); i++) {
        
        xi  = q->quadrature_object().get_points()[i](0);
        eta = q->quadrature_object().get_points()[i](1);
        CHECK(   fe->n_basis() == n_basis);
        
        MAST::Test::FEBasis::Quad4::compute_fe_quad_derivatives(xi, eta,
                                                                x_vec,
                                                                y_vec,
                                                                u_vec,
                                                                mode, // mode
                                                                u,
                                                                du_dx,
                                                                du_dy,
                                                                Nvec,
                                                                dNvec_dxi,
                                                                dNvec_deta,
                                                                dNvec_dx,
                                                                dNvec_dy,
                                                                Jac,
                                                                Jac_inv,
                                                                J_det,
                                                                nvec,
                                                                tvec);
        
        /*SECTION("Finite Element: Shape Function: Nvec")*/ {
            
            // compare shape functions
            for (uint_t j=0; j<n_basis; j++) vec[j] = fe->phi(i, j);
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(Nvec), Catch::Approx(vec));
        }
        
        /*SECTION("Finite Element: Shape Function Derivative: dNvec/dxi")*/ {
            
            // compare shape functions derivatives wrt xi
            for (uint_t j=0; j<n_basis; j++) vec[j] = fe->dphi_dxi(i, j, 0);
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dxi), Catch::Approx(vec));
        }
        
        /*SECTION("Finite Element: Shape Function Derivative: dNvec/deta")*/ {
            
            // compare shape functions derivatives wrt eta
            for (uint_t j=0; j<n_basis; j++) vec[j] = fe->dphi_dxi(i, j, 1);
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_deta), Catch::Approx(vec));
        }
        
        /*SECTION("Finite Element: Jacobian determinants")*/ {
            
            // Jacobian
            CHECK(fe_deriv->detJ(i) == Catch::Detail::Approx(J_det));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(Jac),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dx_dxi(i))));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(Jac_inv),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dxi_dx(i))));
        }
        
        /*SECTION("Finite Element: Shape Function Derivative: dNvec/dx")*/ {
            
            // compare shape functions derivatives wrt x
            for (uint_t j=0; j<n_basis; j++)
                CHECK(dNvec_dx(j) == Catch::Detail::Approx(fe_deriv->dphi_dx(i, j, 0)));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dx),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i, 0))));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dx),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i).col(0))));
        }
        
        /*SECTION("Finite Element: Shape Function Derivative: dNvec/dy")*/ {
            
            // compare shape functions derivatives wrt y
            for (uint_t j=0; j<n_basis; j++)
                CHECK(dNvec_dy(j) == Catch::Detail::Approx(fe_deriv->dphi_dx(i, j, 1)));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dy),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i, 1))));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dy),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i).col(1))));
        }
        
        /*SECTION("Finite Element: Solution and Derivatives: u, du/dx, du/dy")*/ {
            
            // compare solution at quadrature points
            CHECK(    u == Catch::Detail::Approx(fe_var->u(i, 0)));
            CHECK(du_dx == Catch::Detail::Approx(fe_var->du_dx(i, 0, 0)));
            CHECK(du_dy == Catch::Detail::Approx(fe_var->du_dx(i, 0, 1)));
        }
        
        // check the surface normal
        if (mode > 0) {
            
            for (uint_t j=0; j<2; j++)
                CHECK(fe_deriv->normal(i, j) == Catch::Detail::Approx(nvec(j)));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->normal(i)),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(nvec.topRows(2))));
        }

        // check the tangent normal
        if (mode > 0) {
            
            for (uint_t j=0; j<2; j++)
                CHECK(fe_deriv->tangent(i, j) == Catch::Detail::Approx(tvec(j)));
            CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->tangent(i)),
                       Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(tvec.topRows(2))));
        }
    }
    
    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}


TEST_CASE("quad4_basis_derivatives",
          "[2D],[QUAD4],[FEBasis]") {
    
    // mode = 0  domain
    // mode = 1, 2, 3, 4 correspond to sides 0, 1, 2, 3, respectively
    for (uint_t i=0; i<=4; i++)
        test_quad4_fe_data(i);
}

