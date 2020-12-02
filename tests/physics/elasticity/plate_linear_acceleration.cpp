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
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/plate_bending_section_property.hpp>
#include <mast/physics/elasticity/plate_linear_acceleration.hpp>

// Test includes
#include <test_helpers.h>

// libMesh includes
#include <libmesh/elem.h>

extern libMesh::LibMeshInit* p_global_init;

namespace MAST {
namespace Test {
namespace Physics {
namespace Elasticity {
namespace LinearContinuum {
namespace PlateLinearAccelerationKernel {

struct Context {
    Context(): elem(nullptr), qp(-1), s(-1) {}
    uint_t elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    const libMesh::Elem* elem;
    uint_t qp;
    uint_t s;
};


template <typename ScalarConstType>
class ScalarField {
public:
    
    using scalar_t = typename ScalarConstType::scalar_t;
    
    ScalarField(ScalarConstType &s):_const(s) {}
    virtual ~ScalarField() {}
    
    template <typename ContextType> inline scalar_t
    value(ContextType& c) const { return _const.value(c);}

    template <typename ContextType> inline void
    value(ContextType& c, scalar_t &s) const { s = _const.value(c); }
    
    template <typename ContextType, typename ScalarFieldType>
    inline scalar_t
    derivative(ContextType& c, const ScalarFieldType &f) const
    { return _const.derivative(c, f); }

    template <typename ContextType, typename ScalarFieldType>
    inline void
    derivative(ContextType& c, const ScalarFieldType &f,  scalar_t &s) const
    { s = _const.derivative(c, f); }

private:
    ScalarConstType &_const;
};

template <typename BasisScalarType,
          typename NodalScalarType,
          typename SolScalarType>
struct Traits {

    using scalar_t          = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using vector_t          = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t          = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using quadrature_t      = MAST::Quadrature::libMeshWrapper::Quadrature<BasisScalarType, 2>;
    using fe_basis_t        = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, 2>;
    using fe_shape_t        = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, 2, 2, fe_basis_t>;
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 3, 2, Context, fe_shape_t>;
    using density_t         = typename MAST::Base::ScalarConstant<SolScalarType>;
    using th_t              = typename MAST::Base::ScalarConstant<SolScalarType>;
    using density_f_t       = typename MAST::Test::Physics::Elasticity::LinearContinuum::PlateLinearAccelerationKernel::ScalarField<density_t>;
    using th_f_t            = typename  MAST::Test::Physics::Elasticity::LinearContinuum::PlateLinearAccelerationKernel::ScalarField<th_t>;
    using section_prop_t    = typename MAST::Physics::Elasticity::PlateInertiaSectionProperty<scalar_t, density_f_t, th_f_t, Context>;
    using acc_t             = typename MAST::Physics::Elasticity::Plate::LinearAcceleration<fe_var_t, section_prop_t, Context>;
};


template <typename Traits>
struct ElemOps {
  
    ElemOps():
    q        (new typename Traits::quadrature_t(libMesh::QGAUSS, libMesh::FOURTH)),
    fe       (new typename Traits::fe_basis_t(libMesh::FEType(libMesh::FIRST, libMesh::LAGRANGE))),
    fe_deriv (new typename Traits::fe_shape_t),
    fe_var   (new typename Traits::fe_var_t),
    rho      (new typename Traits::density_t(2.7e3)),
    th       (new typename Traits::th_t(0.0025)),
    rho_f    (new typename Traits::density_f_t(*rho)),
    th_f     (new typename Traits::th_f_t(*th)),
    section  (new typename Traits::section_prop_t),
    acc_e    (new typename Traits::acc_t) {
        
        // initialize the shape function derivatives wrt reference coordinates
        fe->set_compute_dphi_dxi(true);
        
        // initialize the shape function derivatives wrt spatial coordinates
        fe_deriv->set_compute_dphi_dx(true);
        fe_deriv->set_compute_detJ(true);
        fe_deriv->set_compute_detJxW(true);
        fe_deriv->set_compute_Jac_inverse(true);
        fe_deriv->set_fe_basis(*fe);

        // initialize the variable data
        fe_var->set_compute_du_dx(false);
        fe_var->set_fe_shape_data(*fe_deriv);
        
        section->set_density_and_thickness(*rho_f, *th_f);
        
        acc_e->set_section_property(*section);
        acc_e->set_fe_var_data(*fe_var);
    }
    
    virtual ~ElemOps() {}

    inline uint_t n_dofs() const { return acc_e->n_dofs();}
    
    inline void init(const libMesh::Elem* e) {
        
        c.elem = e;
        fe->reinit(*e, *q);
        fe_deriv->reinit(c);
    }
    
    inline void compute(const typename Traits::vector_t& sol,
                        typename Traits::vector_t& res,
                        typename Traits::matrix_t* jac=nullptr) {

        fe_var->init(c, sol);
        acc_e->compute(c, res, jac);
    }

    template <typename ScalarFieldType>
    inline void derivative(const ScalarFieldType& f,
                           typename Traits::vector_t& res,
                           typename Traits::matrix_t* jac = nullptr) {
        
        acc_e->derivative(c, f, res, jac);
    }
    
    std::unique_ptr<typename Traits::quadrature_t>   q;
    std::unique_ptr<typename Traits::fe_basis_t>     fe;
    std::unique_ptr<typename Traits::fe_shape_t>     fe_deriv;
    std::unique_ptr<typename Traits::fe_var_t>       fe_var;
    std::unique_ptr<typename Traits::density_t>      rho;
    std::unique_ptr<typename Traits::th_t>           th;
    std::unique_ptr<typename Traits::density_f_t>    rho_f;
    std::unique_ptr<typename Traits::th_f_t>         th_f;
    std::unique_ptr<typename Traits::section_prop_t> section;
    std::unique_ptr<typename Traits::acc_t>          acc_e;
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



TEST_CASE("plate_linear_acceleration",
          "[2D][QUAD4][Elasticity][Linear][Acceleration][Plate]") {
    
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
    
    using traits_t         = Traits<real_t, real_t, real_t>;
    
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

    using traits_complex_t = Traits<real_t, real_t, complex_t>;

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
    
    // residual sensitivity wrt rho
    {
        res.setZero();
        jac.setZero();
        e_ops.derivative(*e_ops.rho, res, &jac);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::density_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.rho, sol, res_cs, jac_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(res),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(res_cs)));
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(jac),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(jac_cs)));
    }

    // residual sensitivity wrt th
    {
        res.setZero();
        jac.setZero();
        e_ops.derivative(*e_ops.th, res, &jac);
        
        ElemOps<traits_complex_t> e_ops_c;
        
        complex_step_derivative
        <typename traits_complex_t::th_t, traits_t, traits_complex_t>
        (e_ops_c, e.get(), *e_ops_c.th, sol, res_cs, jac_cs);
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(res),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(res_cs)));
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(jac),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(jac_cs)));
    }

    for (uint_t i=0; i<nodes.size(); i++)
        delete nodes[i];
}

} // namespace PlateLinearAccelerationKernel
} // namespace LinearContinuum
} // namespace Elasticity
} // namespace Physics
} // namespace Test
} // namespace MAST


