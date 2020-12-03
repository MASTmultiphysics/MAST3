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
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/base/assembly/libmesh/eigenproblem_assembly.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/physics/elasticity/mindlin_plate_strain_energy.hpp>
#include <mast/physics/elasticity/plate_linear_acceleration.hpp>
#include <mast/physics/elasticity/plate_bending_section_property.hpp>
#include <mast/solvers/eigen/constrained_generalized_hermitian_eigen_solver.hpp>
#include <mast/numerics/libmesh/unconstrained_dofs.hpp>

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

// BEGIN_TRANSLATE Modal eigensolution of Mindlin plate

namespace MAST {
namespace Examples {
namespace Structural {
namespace Example3 {

class Context {
    
public:
    
    Context(libMesh::Parallel::Communicator& comm):
    L         (10.),
    q_type    (libMesh::QGAUSS),
    q_order_b (libMesh::SECOND),
    q_order_s (libMesh::FIRST),
    fe_order  (libMesh::FIRST),
    fe_family (libMesh::LAGRANGE),
    mesh      (new libMesh::ReplicatedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    elem      (nullptr),
    qp        (-1) {



        libMesh::MeshTools::Generation::build_square(*mesh,
                                                     20, 20,
                                                     0.0, L,
                                                     0.0, L,
                                                     libMesh::QUAD4);

        sys->add_variable("w", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("t_x", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("t_y", libMesh::FEType(fe_order, fe_family));

        sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({0, 1, 2, 3},
                                    {0},
                                    libMesh::ZeroFunction<real_t>()));
        
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

    real_t                            L;
    libMesh::QuadratureType           q_type;
    libMesh::Order                    q_order_b;
    libMesh::Order                    q_order_s;
    libMesh::Order                    fe_order;
    libMesh::FEFamily                 fe_family;
    libMesh::ReplicatedMesh          *mesh;
    libMesh::EquationSystems         *eq_sys;
    libMesh::NonlinearImplicitSystem *sys;
    const libMesh::Elem              *elem;
    uint_t                            qp;
};



template <typename BasisScalarType,
          typename NodalScalarType,
          typename SolScalarType>
struct Traits {

    using scalar_t          = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using fe_basis_t        = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, 2>;
    using fe_shape_t        = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, 2, 2, fe_basis_t>;
    using fe_data_t         = typename MAST::FEBasis::libMeshWrapper::FEData<2, fe_basis_t, fe_shape_t>;
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 3, 2, Context, fe_shape_t>;
    using modulus_t         = typename MAST::Base::ScalarConstant<SolScalarType>;
    using nu_t              = typename MAST::Base::ScalarConstant<SolScalarType>;
    using rho_t             = typename MAST::Base::ScalarConstant<SolScalarType>;
    using thickness_t       = typename MAST::Base::ScalarConstant<SolScalarType>;
    using material_t        = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, 2, modulus_t, nu_t, Context>;
    using section_b_t       = typename MAST::Physics::Elasticity::PlateBendingSectionProperty<SolScalarType, material_t, thickness_t, Context>;
    using section_in_t      = typename MAST::Physics::Elasticity::PlateInertiaSectionProperty<SolScalarType, rho_t, thickness_t, Context>;
    using energy_t          = typename MAST::Physics::Elasticity::MindlinPlate::StrainEnergy<fe_var_t, section_b_t, Context>;
    using acc_t             = typename MAST::Physics::Elasticity::Plate::LinearAcceleration<fe_var_t, section_in_t, Context>;
    using element_vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using element_matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using assembled_vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using assembled_matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
};



template <typename TraitsType>
class ElemOps {
  
public:
    
    using scalar_t = typename TraitsType::scalar_t;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    

    ElemOps(libMesh::Order          q_order_b,
            libMesh::Order          q_order_s,
            libMesh::QuadratureType q_type,
            libMesh::Order          fe_order,
            libMesh::FEFamily       fe_family):
    E             (nullptr),
    nu            (nullptr),
    thickness     (nullptr),
    _fe_data_b    (nullptr),
    _fe_data_s    (nullptr),
    _fe_var_b     (nullptr),
    _fe_var_s     (nullptr),
    _material     (nullptr),
    _section_b    (nullptr),
    _section_in   (nullptr),
    _energy       (nullptr),
    _acc          (nullptr) {
        
        _fe_data_b       = new typename TraitsType::fe_data_t;
        _fe_data_b->init(q_order_b, q_type, fe_order, fe_family);
        _fe_var_b        = new typename TraitsType::fe_var_t;

        _fe_data_s       = new typename TraitsType::fe_data_t;
        _fe_data_s->init(q_order_s, q_type, fe_order, fe_family);
        _fe_var_s        = new typename TraitsType::fe_var_t;

        // associate variables with the shape functions
        _fe_var_b->set_fe_shape_data(_fe_data_b->fe_derivative());
        _fe_var_s->set_fe_shape_data(_fe_data_s->fe_derivative());

        // tell the FE computations which quantities are needed for computation
        _fe_data_b->fe_basis().set_compute_dphi_dxi(true);
        _fe_data_s->fe_basis().set_compute_dphi_dxi(true);

        _fe_data_b->fe_derivative().set_compute_dphi_dx(true);
        _fe_data_b->fe_derivative().set_compute_detJxW(true);
        _fe_data_s->fe_derivative().set_compute_dphi_dx(true);
        _fe_data_s->fe_derivative().set_compute_detJxW(true);

        _fe_var_b->set_compute_du_dx(true);
        _fe_var_s->set_compute_du_dx(true);

        // variables for physics
        E           = new typename TraitsType::modulus_t(72.e9);
        nu          = new typename TraitsType::nu_t(0.33);
        rho         = new typename TraitsType::nu_t(2.7e3);
        thickness   = new typename TraitsType::thickness_t(3.0e-3);
        _material   = new typename TraitsType::material_t;
        _section_b  = new typename TraitsType::section_b_t;
        _section_in = new typename TraitsType::section_in_t;

        _material->set_modulus_and_nu(*E, *nu);
        _section_b->set_material_and_thickness(*_material, *thickness);
        _section_in->set_density_and_thickness(*rho, *thickness);

        _energy   = new typename TraitsType::energy_t;
        _energy->set_section_property(*_section_b);
        
        _acc      = new typename TraitsType::acc_t;
        _acc->set_section_property(*_section_in);
        
        // tell physics kernels about the FE discretization information
        _energy->set_fe_var_data(*_fe_var_b, *_fe_var_s);
        _acc->set_fe_var_data(*_fe_var_b);
    }
    
    virtual ~ElemOps() {
        
        delete thickness;
        delete _acc;
        delete _energy;
        delete _section_in;
        delete _section_b;
        delete _material;
        delete rho;
        delete nu;
        delete E;
        delete _fe_var_b;
        delete _fe_data_b;
        delete _fe_var_s;
        delete _fe_data_s;
    }
    

    template <typename ContextType, typename AccessorType>
    inline void compute(ContextType                       &c,
                        const AccessorType                &v,
                        typename TraitsType::element_vector_t &res,
                        typename TraitsType::element_matrix_t *jac) {
        
        _fe_data_b->reinit(c);
        _fe_var_b->init(c, v);
        _fe_data_s->reinit(c);
        _fe_var_s->init(c, v);
        
        _energy->compute(c, res, jac);
    }


    template <typename ContextType, typename AccessorType>
    inline void compute(ContextType                       &c,
                        const AccessorType                &v,
                        typename TraitsType::element_matrix_t &A,
                        typename TraitsType::element_matrix_t &B) {
        
        _fe_data_b->reinit(c);
        _fe_var_b->init(c, v);
        _fe_data_s->reinit(c);
        _fe_var_s->init(c, v);
        
        typename TraitsType::element_vector_t
        res = TraitsType::element_vector_t::Zero(A.rows());

        _energy->compute(c, res, &A); // stiffness matrix
        _acc->compute   (c, res, &B); // mass matrix
    }

    
    template <typename ContextType,
              typename AccessorType,
              typename ScalarFieldType>
    inline void derivative(ContextType                       &c,
                           const ScalarFieldType             &f,
                           const AccessorType                &v,
                           typename TraitsType::element_vector_t &res,
                           typename TraitsType::element_matrix_t *jac) {
        
        _fe_data_b->reinit(c);
        _fe_var_b->init(c, v);
        _fe_data_s->reinit(c);
        _fe_var_s->init(c, v);
        _energy->derivative(c, f, res, jac);
    }

    
    template <typename ContextType,
              typename AccessorType,
              typename ScalarFieldType>
    inline void derivative(ContextType                           &c,
                           const ScalarFieldType                 &f,
                           const AccessorType                    &v,
                           typename TraitsType::element_matrix_t &A,
                           typename TraitsType::element_matrix_t &B) {
        
        _fe_data_b->reinit(c);
        _fe_var_b->init(c, v);
        _fe_data_s->reinit(c);
        _fe_var_s->init(c, v);

        typename TraitsType::element_vector_t
        res = TraitsType::element_vector_t::Zero(A.rows());

        _energy->derivative(c, f, res, &A);
        _acc->derivative   (c, f, res, &B);
    }

    
    // parameters
    typename TraitsType::modulus_t    *E;
    typename TraitsType::nu_t         *nu;
    typename TraitsType::rho_t        *rho;
    typename TraitsType::thickness_t  *thickness;
    
private:

    // variables for quadrature and shape function
    typename TraitsType::fe_data_t         *_fe_data_b;
    typename TraitsType::fe_data_t         *_fe_data_s;
    typename TraitsType::fe_var_t          *_fe_var_b;
    typename TraitsType::fe_var_t          *_fe_var_s;
    typename TraitsType::material_t        *_material;
    typename TraitsType::section_b_t       *_section_b;
    typename TraitsType::section_in_t      *_section_in;
    typename TraitsType::energy_t          *_energy;
    typename TraitsType::acc_t             *_acc;
};

} // namespace Example3
} // namespace Structural
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

int main(int argc, const char** argv) {

    libMesh::LibMeshInit init(argc, argv);
    
    using traits_t   = MAST::Examples::Structural::Example3::Traits<real_t, real_t, real_t>;
    using elem_ops_t = MAST::Examples::Structural::Example3::ElemOps<traits_t>;

    MAST::Examples::Structural::Example3::Context
    c(init.comm());

    elem_ops_t
    e_ops(c.q_order_b, c.q_order_s, c.q_type, c.fe_order, c.fe_family);

    uint_t
    n    = c.sys->get_dof_map().n_dofs(),
    n_ev = 5;
    
    real_t
    pi   = acos(-1),
    E    = (*e_ops.E)(),
    nu   = (*e_ops.nu)(),
    th   = (*e_ops.thickness)(),
    rho  = (*e_ops.rho)();
    
    MAST::Base::Assembly::libMeshWrapper::EigenProblemAssembly<real_t, elem_ops_t>
    assembly;
    typename traits_t::assembled_matrix_t
    A   = traits_t::assembled_matrix_t::Zero(n, n),
    B   = traits_t::assembled_matrix_t::Zero(n, n),
    dA  = traits_t::assembled_matrix_t::Zero(n, n),
    dB  = traits_t::assembled_matrix_t::Zero(n, n);

    typename traits_t::assembled_vector_t
    sol = traits_t::assembled_vector_t::Zero(n),
    vec = traits_t::assembled_vector_t::Zero(n);
    
    assembly.set_elem_ops(e_ops);

    // assembly of matrices
    assembly.assemble(c, sol, A, B);
    
    // sensitivity analysis with respect to thickness
    assembly.sensitivity_assemble(c, *e_ops.thickness, sol, dA, dB);

    // vector of unconstrained dofs
    std::vector<uint_t>
    unconstrained_dofs;
    
    MAST::Numerics::libMeshWrapper::unconstrained_dofs(c.sys->get_dof_map(),
                                                       unconstrained_dofs);
    
    // compute the solution
    MAST::Solvers::EigenWrapper::ConstrainedGeneralizedHermitianEigenSolver
    <real_t,
    Eigen::GeneralizedSelfAdjointEigenSolver<traits_t::assembled_matrix_t>,
    traits_t::assembled_matrix_t,
    traits_t::assembled_vector_t>
    solver(unconstrained_dofs);
    
    solver.solve(A, B, true);
    
    std::vector<real_t>
    eig            (n_ev, 0.),
    eig_analytical (n_ev, 0.),
    deig           (n_ev, 0.),
    deig_analytical(n_ev, 0.);
    
    // write solution as first time-step. The first 5 modes are written
    libMesh::ExodusII_IO writer(*c.mesh);
    for (uint_t j=0; j<5; j++) {

        // get the numerical eigenvalue
        eig[j] = solver.eig(j);
        
        // the analytical eigenvalue of a simply supported Kirchoff plate
        // with dimensions 10 x 10
        //eig_analytical[j] = (E*pow(th,2)/12./(1.-pow(nu,2)) *
        //                     (m * pi));
        
        // get the eigenvector from the solver.
        solver.getEigenVector(j, vec);

        // compute the sensitivity
        solver.sensitivity_solve(B, dA, dB, j);

        // copy to libmesh::System::solution for output
        c.sys->solution->zero();
        for (uint_t i=0; i<n; i++) c.sys->solution->set(i, vec(i));
        c.sys->solution->close();

        // write mode to output file.
        writer.write_timestep("modes.exo", *c.eq_sys, j+1, j);
    }

    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
