
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
    uint_t elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    libMesh::Elem* elem;
    uint_t qp;
    uint_t s;
};

TEST_CASE("quad4_basis_derivatives",
          "[2D],[QUAD4],[FEBasis]") {

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
    nvec;

    x_vec << -1., 1., 1., -1.;
    y_vec << -1., -1., 1., 1.;
    u_vec << 1., 2., 3., 4.;
    
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
    fe_deriv->reinit(c);
    
    // initialize the variable data
    fe_var->set_compute_du_dx(true);
    fe_var->set_fe_shape_data(*fe_deriv);
    fe_var->init(c, u_vec);
    
    // iterate over points and check the data for accuracy
    for (uint_t i=0; i<fe->n_q_points(); i++) {
        
        xi  = q->quadrature_object().get_points()[i](0);
        eta = q->quadrature_object().get_points()[i](1);
        REQUIRE(   fe->n_basis() == n_basis);
        
        MAST::Test::FEBasis::Quad4::compute_fe_quad_derivatives(xi, eta,
                                                                x_vec,
                                                                y_vec,
                                                                u_vec,
                                                                0, // mode
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
                                                                nvec);
        
        /*SECTION("Finite Element: Shape Function: Nvec")*/ {

            // compare shape functions
            for (uint_t j=0; j<n_basis; j++) vec[j] = fe->phi(i, j);
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(Nvec), Catch::Approx(vec));
        }

        /*SECTION("Finite Element: Shape Function Derivative: dNvec/dxi")*/ {
            
            // compare shape functions derivatives wrt xi
            for (uint_t j=0; j<n_basis; j++) vec[j] = fe->dphi_dxi(i, j, 0);
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dxi), Catch::Approx(vec));
        }

        /*SECTION("Finite Element: Shape Function Derivative: dNvec/deta")*/ {
            
            // compare shape functions derivatives wrt eta
            for (uint_t j=0; j<n_basis; j++) vec[j] = fe->dphi_dxi(i, j, 1);
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_deta), Catch::Approx(vec));
        }

        /*SECTION("Finite Element: Jacobian determinants")*/ {
            
            // Jacobian
            REQUIRE(fe_deriv->detJ(i) == Catch::Detail::Approx(J_det));
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(Jac),
                         Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dx_dxi(i))));
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(Jac_inv),
                         Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dxi_dx(i))));
        }

        /*SECTION("Finite Element: Shape Function Derivative: dNvec/dx")*/ {
            
            // compare shape functions derivatives wrt x
            for (uint_t j=0; j<n_basis; j++)
                REQUIRE(dNvec_dx(j) == Catch::Detail::Approx(fe_deriv->dphi_dx(i, j, 0)));
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dx),
                         Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i, 0))));
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dx),
                         Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i).col(0))));
        }

        /*SECTION("Finite Element: Shape Function Derivative: dNvec/dy")*/ {
            
            // compare shape functions derivatives wrt y
            for (uint_t j=0; j<n_basis; j++)
                REQUIRE(dNvec_dy(j) == Catch::Detail::Approx(fe_deriv->dphi_dx(i, j, 1)));
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dy),
                         Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i, 1))));
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dy),
                         Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(fe_deriv->dphi_dx(i).col(1))));
        }

        /*SECTION("Finite Element: Solution and Derivatives: u, du/dx, du/dy")*/ {
            
            // compare solution at quadrature points
            REQUIRE(    u == Catch::Detail::Approx(fe_var->u(i, 0)));
            REQUIRE(du_dx == Catch::Detail::Approx(fe_var->du_dx(i, 0, 0)));
            REQUIRE(du_dy == Catch::Detail::Approx(fe_var->du_dx(i, 0, 1)));
        }

    }

    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}

