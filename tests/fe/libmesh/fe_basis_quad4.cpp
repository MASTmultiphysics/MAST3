
// Catch includes
#include "catch.hpp"

// MAST includes
#include <mast/fe/libmesh/fe.hpp>

// Test includes
#include <fe/quad_derivatives.hpp>
#include <test_helpers.h>

// libMesh includes
#include <libmesh/elem.h>

extern libMesh::LibMeshInit* p_global_init;


TEST_CASE("quad4_basis_derivatives",
          "[2D],[QUAD4],[FEBasis]") {

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

    std::unique_ptr<MAST::Quadrature::libMeshWrapper::Quadrature<real_t, 2>>
    q(new MAST::Quadrature::libMeshWrapper::Quadrature<real_t, 2>(libMesh::QGAUSS, libMesh::FOURTH));
    
    std::unique_ptr<MAST::FEBasis::libMeshWrapper::FEBasis<real_t, 2>>
    fe(new MAST::FEBasis::libMeshWrapper::FEBasis<real_t, 2>(libMesh::FEType(libMesh::FIRST, libMesh::LAGRANGE)));

    fe->set_compute_dphi_dxi(true);
    fe->reinit(*e, *q);
    
    for (uint_t i=0; i<fe->n_q_points(); i++) {
        
        xi  = q->quadrature_object().get_points()[i](0);
        eta = q->quadrature_object().get_points()[i](1);
        REQUIRE(   fe->n_basis() == 4);
        
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
        
        SECTION("Finite Element: Shape Function: Nvec") {
            
            // compare shape functions
            for (uint_t j=0; j<4; j++) vec[j] = fe->phi(i, j);
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(Nvec), Catch::Approx(vec));
        }

        SECTION("Finite Element: Shape Function Derivative: dNvec/dxi") {
            
            // compare shape functions
            for (uint_t j=0; j<4; j++) vec[j] = fe->dphi_dxi(i, j, 0);
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_dxi), Catch::Approx(vec));
        }

        SECTION("Finite Element: Shape Function Derivative: dNvec/deta") {
            
            // compare shape functions
            for (uint_t j=0; j<4; j++) vec[j] = fe->dphi_dxi(i, j, 1);
            REQUIRE_THAT(MAST::Test::eigen_matrix_to_std_vector(dNvec_deta), Catch::Approx(vec));
        }
    }

    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}

