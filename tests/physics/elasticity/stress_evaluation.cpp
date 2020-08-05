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
#include <mast/fe/libmesh/fe.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/continuum_stress.hpp>

// Test includes
#include <test_helpers.h>

// libMesh includes
#include <libmesh/elem.h>

extern libMesh::LibMeshInit* p_global_init;

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace StressEvaluation {

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
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, Dim, modulus_t, nu_t>;
    using stress_t          = typename MAST::Physics::Elasticity::LinearContinuum::Stress<fe_var_t, prop_t, Dim>;
    using stress_vec_t      = typename Eigen::Matrix<scalar_t, stress_t::n_strain, 1>;
    using stress_adj_mat_t  = typename Eigen::Matrix<scalar_t, stress_t::n_strain, Eigen::Dynamic>;
    using stress_storage_t  = typename Eigen::Matrix<scalar_t, stress_t::n_strain, Eigen::Dynamic>;
    using stress_adj_storage_t = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
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
    stress_e (new typename Traits::stress_t) {
        
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
        
        stress_e->set_section_property(*prop);
        stress_e->set_fe_var_data(*fe_var);
    }
    
    virtual ~ElemOps() {}

    inline uint_t n_dofs() const { return stress_e->n_dofs();}
    
    inline void init(const libMesh::Elem* e) {
        
        c.elem = e;
        fe->reinit(*e, *q);
        fe_deriv->reinit(c);
    }
    
    inline void compute(const typename Traits::vector_t& sol,
                        typename Traits::stress_storage_t& stress_storage) {

        fe_var->init(c, sol);

        typename Traits::stress_vec_t
        stress;
        
        for (uint_t i=0; i<q->n_points(); i++) {
            
            c.qp = i;
            stress.setZero();
            stress_e->compute(c, stress);
            stress_storage.col(i) = stress;
        }
    }

    
    inline void compute_adjoint(const typename Traits::vector_t& sol,
                                typename Traits::stress_adj_storage_t& stress_adj_storage) {

        fe_var->init(c, sol);

        typename Traits::stress_adj_mat_t
        stress_adj;
        
        const uint_t
        n_strain = Traits::stress_t::n_strain;
        
        stress_adj = Traits::stress_adj_mat_t::Zero(n_strain, this->n_dofs());
        
        for (uint_t i=0; i<q->n_points(); i++) {
            
            c.qp = i;
            stress_adj.setZero(n_strain, this->n_dofs());
            
            stress_e->adjoint_derivative(c, stress_adj);
            
            Eigen::Map<typename Traits::stress_adj_mat_t>
            m(stress_adj_storage.col(i).data(), n_strain, this->n_dofs());
            
            m = stress_adj;
        }
    }

    template <typename ScalarFieldType>
    inline void derivative(const ScalarFieldType& f,
                           typename Traits::stress_storage_t& dstress_storage) {

        typename Traits::stress_vec_t
        dstress;
        
        for (uint_t i=0; i<q->n_points(); i++) {
            
            c.qp = i;
            dstress.setZero();
            stress_e->derivative(c, f, dstress);
            dstress_storage.col(i) = dstress;
        }
    }
    
    std::unique_ptr<typename Traits::quadrature_t>   q;
    std::unique_ptr<typename Traits::fe_basis_t>     fe;
    std::unique_ptr<typename Traits::fe_shape_t>     fe_deriv;
    std::unique_ptr<typename Traits::fe_var_t>       fe_var;
    std::unique_ptr<typename Traits::modulus_t>      E;
    std::unique_ptr<typename Traits::nu_t>           nu;
    std::unique_ptr<typename Traits::prop_t>         prop;
    std::unique_ptr<typename Traits::stress_t>       stress_e;
    Context                                          c;
};



template <typename ScalarConstantType,
          typename Traits,
          typename TraitsComplex>
inline void complex_step_derivative(ElemOps<TraitsComplex>             &e_ops_c,
                                    const libMesh::Elem                *e,
                                    ScalarConstantType                 &f,
                                    const typename Traits::vector_t    &sol,
                                    typename Traits::stress_storage_t  &dstress)  {
    

    typename TraitsComplex::vector_t
    sol_c;
    
    typename TraitsComplex::stress_storage_t
    stress_c;

    e_ops_c.init(e);

    stress_c  = TraitsComplex::stress_storage_t::Zero(TraitsComplex::stress_t::n_strain,
                                                      e_ops_c.q->n_points());

    sol_c = sol.template cast<complex_t>();

    // add perturbation to parameter
    f() += complex_t(0., ComplexStepDelta);
    
    e_ops_c.compute(sol_c, stress_c);
    
    dstress = stress_c.imag()/ComplexStepDelta;
}




TEST_CASE("stress_evaluation",
          "[2D][QUAD4][Elasticity][Linear][Stress]") {
    
    const uint_t
    n_basis  = 4,
    n_strain = MAST::Physics::Elasticity::LinearContinuum::NStrainComponents<2>::value;

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
    sol;
    
    // stress at all quadrature points
    typename traits_t::stress_storage_t
    stress,
    stress_cs;

    // stress adjoint matrix at all quadrature points
    typename traits_t::stress_adj_storage_t
    dstress_dX,
    dstress_dX_cs;

    ElemOps<traits_t> e_ops;
    e_ops.init(e.get());
    
    sol           = 0.1 * traits_t::vector_t::Random(e_ops.n_dofs());
    stress        = traits_t::stress_storage_t::Zero(n_strain, e_ops.q->n_points());
    stress_cs     = traits_t::stress_storage_t::Zero(n_strain, e_ops.q->n_points());
    dstress_dX    = traits_t::stress_adj_storage_t::Zero(n_strain*e_ops.n_dofs(),
                                                         e_ops.q->n_points());
    dstress_dX_cs = traits_t::stress_adj_storage_t::Zero(n_strain*e_ops.n_dofs(),
                                                         e_ops.q->n_points());
    
    e_ops.compute(sol, stress);
    e_ops.compute_adjoint(sol, dstress_dX);
    
    using traits_complex_t = Traits<real_t, real_t, complex_t, 2>;

    // compute the complex-step adjoint derivative
    {
        for (uint_t i=0; i<sol.size(); i++) {
                        
            typename traits_complex_t::vector_t
            sol_c;

            typename traits_complex_t::stress_storage_t
            stress_c;

            ElemOps<traits_complex_t> e_ops_c;
            e_ops_c.init(e.get());
            
            sol_c    = sol.cast<complex_t>();
            stress_c = traits_complex_t::stress_storage_t::Zero(n_strain, e_ops.q->n_points());

            // complex perturbation to dof
            sol_c(i) += complex_t(0., ComplexStepDelta);
            
            e_ops_c.compute(sol_c, stress_c);
            
            // copy data for the ith dof of jth quadrature point
            for (uint_t j=0; j<e_ops.q->n_points(); j++) {
                // matrix for the jth quadrature point
                Eigen::Map<traits_t::stress_adj_mat_t>
                m(dstress_dX_cs.col(j).data(), n_strain, e_ops.n_dofs());
                
                // column corresponding to the ith dof
                m.col(i) = stress_c.col(j).imag()/ComplexStepDelta;
            }
        }

        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(dstress_dX),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(dstress_dX_cs)));
    }
    
    // residual sensitivity wrt E
    {
        stress.setZero();
        e_ops.derivative(*e_ops.E, stress);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::modulus_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.E, sol, stress_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(stress),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(stress_cs)));
    }

    // residual sensitivity wrt nu
    {
        stress.setZero();
        e_ops.derivative(*e_ops.nu, stress);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::modulus_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.nu, sol, stress_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(stress),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(stress_cs)));
    }

    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}

} // namespace StressEvaluation
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


