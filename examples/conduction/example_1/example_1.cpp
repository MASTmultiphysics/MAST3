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

// MAST includes
#include <mast/base/exceptions.hpp>
#include <mast/util/perf_log.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/libmesh/fe_side_data.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/conduction/material_conductance.hpp>
#include <mast/physics/conduction/linear_conduction_kernel.hpp>
#include <mast/physics/conduction/source_kernel.hpp>
#include <mast/base/assembly/libmesh/residual_and_jacobian.hpp>
#include <mast/base/assembly/libmesh/residual_sensitivity.hpp>
#include <mast/numerics/libmesh/sparse_matrix_initialization.hpp>

// libMesh includes
#include <libmesh/replicated_mesh.h>
#include <libmesh/elem.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/boundary_info.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/zero_function.h>
#include <libmesh/exodusII_io.h>

// Eigen includes
#include <Eigen/SparseLU>

// BEGIN_TRANSLATE Two-dimensional Linear Conduction problem with volume source load

namespace MAST {
namespace Examples {
namespace Conduction {
namespace Example1 {

class Context {
    
public:
    
    Context(libMesh::Parallel::Communicator& comm):
    q_type    (libMesh::QGAUSS),
    q_order   (libMesh::FOURTH),
    fe_order  (libMesh::SECOND),
    fe_family (libMesh::LAGRANGE),
    mesh      (new libMesh::ReplicatedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    elem      (nullptr),
    qp        (-1),
    p_side_id (1) {



        libMesh::MeshTools::Generation::build_square(*mesh,
                                                     5, 5,
                                                     0.0, 10.0,
                                                     0.0, 10.0,
                                                     libMesh::QUAD9);

        sys->add_variable("T", libMesh::FEType(fe_order, fe_family));

        sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({0, 1, 2, 3}, {0}, libMesh::ZeroFunction<real_t>()));
        
        eq_sys->init();

        mesh->print_info(std::cout);
        eq_sys->print_info(std::cout);
    }

    virtual ~Context() {
        
        delete eq_sys;
        delete mesh;
    }
    
    uint_t elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    inline bool elem_is_quad() const {return (elem->type() == libMesh::QUAD4 ||
                                              elem->type() == libMesh::QUAD8 ||
                                              elem->type() == libMesh::QUAD9);}
    inline bool if_compute_pressure_load_on_side(const uint_t s)
    { return mesh->boundary_info->has_boundary_id(elem, s, p_side_id);}

    libMesh::QuadratureType           q_type;
    libMesh::Order                    q_order;
    libMesh::Order                    fe_order;
    libMesh::FEFamily                 fe_family;
    libMesh::ReplicatedMesh          *mesh;
    libMesh::EquationSystems         *eq_sys;
    libMesh::NonlinearImplicitSystem *sys;
    const libMesh::Elem              *elem;
    uint_t                            qp;
    uint_t                            p_side_id;
};



template <typename BasisScalarType,
          typename NodalScalarType,
          typename SolScalarType,
          uint_t   Dim>
struct Traits {

    using scalar_t          = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using fe_basis_t        = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, Dim>;
    using fe_shape_t        = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, Dim, Dim, fe_basis_t>;
    using fe_data_t         = typename MAST::FEBasis::libMeshWrapper::FEData<Dim, fe_basis_t, fe_shape_t>;
    using fe_side_data_t    = typename MAST::FEBasis::libMeshWrapper::FESideData<Dim, fe_basis_t, fe_shape_t>;
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 1, Dim, Context, fe_shape_t>;
    using conductance_t     = typename MAST::Base::ScalarConstant<SolScalarType>;
    using source_t          = typename MAST::Base::ScalarConstant<SolScalarType>;
    using area_t            = typename MAST::Base::ScalarConstant<SolScalarType>;
    using prop_t            = typename MAST::Physics::Conduction::IsotropicMaterialConductance<SolScalarType, conductance_t, Context>;
    using conduction_t      = typename MAST::Physics::Conduction::ConductionKernel<fe_var_t, prop_t, Dim, Context, true, true>;
    using source_load_t     = typename MAST::Physics::Conduction::SourceHeatLoad<fe_var_t, source_t, area_t, Dim, Context>;
    using element_vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using element_matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using assembled_vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using assembled_matrix_t = Eigen::SparseMatrix<scalar_t>;
};



template <typename TraitsType>
class ElemOps {
  
public:
    
    using scalar_t = typename TraitsType::scalar_t;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    

    ElemOps(libMesh::Order          q_order,
            libMesh::QuadratureType q_type,
            libMesh::Order          fe_order,
            libMesh::FEFamily       fe_family):
    k             (nullptr),
    q             (nullptr),
    area          (nullptr),
    _fe_data      (nullptr),
    _fe_side_data (nullptr),
    _fe_var       (nullptr),
    _fe_side_var  (nullptr),
    _prop         (nullptr),
    _kernel       (nullptr),
    _q_load       (nullptr) {
        
        _fe_data       = new typename TraitsType::fe_data_t;
        _fe_data->init(q_order, q_type, fe_order, fe_family);
        _fe_side_data  = new typename TraitsType::fe_side_data_t;
        _fe_side_data->init(q_order, q_type, fe_order, fe_family);
        _fe_var        = new typename TraitsType::fe_var_t;
        _fe_side_var   = new typename TraitsType::fe_var_t;

        // associate variables with the shape functions
        _fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _fe_side_var->set_fe_shape_data(_fe_side_data->fe_derivative());

        // tell the FE computations which quantities are needed for computation
        _fe_data->fe_basis().set_compute_dphi_dxi(true);
        
        _fe_data->fe_derivative().set_compute_dphi_dx(true);
        _fe_data->fe_derivative().set_compute_detJxW(true);
        
        _fe_side_data->fe_basis().set_compute_dphi_dxi(true);
        _fe_side_data->fe_derivative().set_compute_normal(true);
        _fe_side_data->fe_derivative().set_compute_detJxW(true);

        _fe_var->set_compute_du_dx(true);
        
        // variables for physics
        k        = new typename TraitsType::conductance_t(3.5e3);
        q        = new typename TraitsType::source_t(1.e2);
        area     = new typename TraitsType::area_t(1.0);
        _prop    = new typename TraitsType::prop_t;
        
        _prop->set_conductance(*k);
        _kernel   = new typename TraitsType::conduction_t;
        _kernel->set_section_property(*_prop);
        _q_load   = new typename TraitsType::source_load_t;
        _q_load->set_section_area(*area);
        _q_load->set_source(*q);
        
        // tell physics kernels about the FE discretization information
        _kernel->set_fe_var_data(*_fe_var);
        _q_load->set_fe_var_data(*_fe_var);
    }
    
    virtual ~ElemOps() {
        
        delete _q_load;
        delete area;
        delete q;
        delete _kernel;
        delete _prop;
        delete k;
        delete _fe_var;
        delete _fe_side_var;
        delete _fe_side_data;
        delete _fe_data;
    }
    

    template <typename ContextType, typename AccessorType>
    inline void compute(ContextType                       &c,
                        const AccessorType                &v,
                        typename TraitsType::element_vector_t &res,
                        typename TraitsType::element_matrix_t *jac) {
        
        _fe_data->reinit(c);
        _fe_var->init(c, v);
        _kernel->compute(c, res, jac);
        
        //_fe_side_data->reinit_for_side(c, s);
        //_fe_side_var->init(c, v);
        _q_load->compute(c, res, jac);
    }

    
    template <typename ContextType, typename AccessorType, typename ScalarFieldType>
    inline void derivative(ContextType                       &c,
                           const ScalarFieldType             &f,
                           const AccessorType                &v,
                           typename TraitsType::element_vector_t &res,
                           typename TraitsType::element_matrix_t *jac) {
        
        _fe_data->reinit(c);
        _fe_var->init(c, v);
        _kernel->derivative(c, f, res, jac);
        
        //_fe_side_data->reinit_for_side(c, s);
        //_fe_side_var->init(c, v);
        _q_load->derivative(c, f, res, jac);
    }

    // parameters
    typename TraitsType::conductance_t    *k;
    typename TraitsType::source_t         *q;
    typename TraitsType::area_t           *area;
    
private:

    // variables for quadrature and shape function
    typename TraitsType::fe_data_t         *_fe_data;
    typename TraitsType::fe_side_data_t    *_fe_side_data;
    typename TraitsType::fe_var_t          *_fe_var;
    typename TraitsType::fe_var_t          *_fe_side_var;
    typename TraitsType::prop_t            *_prop;
    typename TraitsType::conduction_t      *_kernel;
    typename TraitsType::source_load_t     *_q_load;
};


template <typename TraitsType>
inline void
compute_residual(Context                                        &c,
                 ElemOps<TraitsType>                            &e_ops,
                 const typename TraitsType::assembled_vector_t  &sol,
                 typename TraitsType::assembled_vector_t        &res) {
    
    using scalar_t   = typename TraitsType::scalar_t;

    MAST::Base::Assembly::libMeshWrapper::ResidualAndJacobian<scalar_t, ElemOps<TraitsType>>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    typename TraitsType::assembled_matrix_t
    *jac = nullptr;
    
    res = TraitsType::assembled_vector_t::Zero(c.sys->n_dofs());
    
    assembly.assemble(c, sol, &res, jac);
}



template <typename TraitsType, typename ScalarFieldType>
inline void
compute_residual_sensitivity(Context                                        &c,
                             ElemOps<TraitsType>                            &e_ops,
                             const ScalarFieldType                          &f,
                             const typename TraitsType::assembled_vector_t  &sol,
                             typename TraitsType::assembled_vector_t        &dres) {
    
    using scalar_t   = typename TraitsType::scalar_t;

    MAST::Base::Assembly::libMeshWrapper::ResidualSensitivity<scalar_t, ElemOps<TraitsType>>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    typename TraitsType::assembled_matrix_t
    *jac = nullptr;
    
    dres = TraitsType::assembled_vector_t::Zero(c.sys->n_dofs());
    
    assembly.assemble(c, f, sol, &dres, jac);
}



template <typename TraitsType>
inline void
compute_sol(Context                                  &c,
            ElemOps<TraitsType>                      &e_ops,
            typename TraitsType::assembled_vector_t  &sol) {
    
    using scalar_t   = typename TraitsType::scalar_t;

    MAST::Base::Assembly::libMeshWrapper::ResidualAndJacobian<scalar_t, ElemOps<TraitsType>>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    typename TraitsType::assembled_vector_t
    res;
    typename TraitsType::assembled_matrix_t
    jac;
    
    sol = TraitsType::assembled_vector_t::Zero(c.sys->n_dofs());
    res = TraitsType::assembled_vector_t::Zero(c.sys->n_dofs());
    MAST::Numerics::libMeshWrapper::init_sparse_matrix(c.sys->get_dof_map(), jac);
    
    assembly.assemble(c, sol, &res, &jac);
    
    sol = Eigen::SparseLU<typename TraitsType::assembled_matrix_t>(jac).solve(-res);
}


template <typename TraitsType, typename ScalarFieldType>
inline void
compute_sol_sensitivity(Context                                        &c,
                        ElemOps<TraitsType>                            &e_ops,
                        const ScalarFieldType                          &f,
                        const typename TraitsType::assembled_vector_t  &sol,
                        typename TraitsType::assembled_vector_t        &dsol) {
    
    using scalar_t   = typename TraitsType::scalar_t;
    
    typename TraitsType::assembled_vector_t
    *res = nullptr,
    dres;
    typename TraitsType::assembled_matrix_t
    jac,
    *djac = nullptr;
    
    dsol = TraitsType::assembled_vector_t::Zero(c.sys->n_dofs());
    dres = TraitsType::assembled_vector_t::Zero(c.sys->n_dofs());
    MAST::Numerics::libMeshWrapper::init_sparse_matrix(c.sys->get_dof_map(), jac);

    // assembly of Jacobian matrix
    {
        MAST::Base::Assembly::libMeshWrapper::ResidualAndJacobian<scalar_t, ElemOps<TraitsType>>
        assembly;
        assembly.set_elem_ops(e_ops);
        assembly.assemble(c, sol, res, &jac);
    }

    // assembly of sensitivity RHS
    {
        MAST::Base::Assembly::libMeshWrapper::ResidualSensitivity<scalar_t, ElemOps<TraitsType>>
        sens_assembly;
        sens_assembly.set_elem_ops(e_ops);
        sens_assembly.assemble(c, f, sol, &dres, djac);
    }
    
    dsol = Eigen::SparseLU<typename TraitsType::assembled_matrix_t>(jac).solve(-dres);
}


} // namespace Example1
} // namespace Conduction
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

int main(int argc, const char** argv) {

    libMesh::LibMeshInit init(argc, argv);
    
    using traits_t           = MAST::Examples::Conduction::Example1::Traits<real_t, real_t,    real_t, 2>;
    using traits_complex_t   = MAST::Examples::Conduction::Example1::Traits<real_t, real_t, complex_t, 2>;


    MAST::Examples::Conduction::Example1::Context c(init.comm());
    MAST::Examples::Conduction::Example1::ElemOps<traits_t>
    e_ops(c.q_order, c.q_type, c.fe_order, c.fe_family);
    MAST::Examples::Conduction::Example1::ElemOps<traits_complex_t>
    e_ops_c(c.q_order, c.q_type, c.fe_order, c.fe_family);

    typename traits_t::assembled_vector_t
    sol,
    dsol;

    typename traits_complex_t::assembled_vector_t
    sol_c;

    // compute the solution
    MAST::Examples::Conduction::Example1::compute_sol<traits_t>(c, e_ops, sol);
    
    // write solution as first time-step
    libMesh::ExodusII_IO writer(*c.mesh);
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, sol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 1, 0.);
    }

    // compute the solution sensitivity wrt k
    (*e_ops_c.k)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Conduction::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.k)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Conduction::Example1::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.k, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 2, 1.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    // compute the solution sensitivity wrt q
    (*e_ops_c.q)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Conduction::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.q)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Conduction::Example1::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.q, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 3, 2.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    // compute the solution sensitivity wrt section area
    (*e_ops_c.area)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Conduction::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.area)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Conduction::Example1::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.area, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 4, 3.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
