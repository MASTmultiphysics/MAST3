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
#include <mast/fe/scalar_field_wrapper.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/linear_strain_energy.hpp>
#include <mast/physics/elasticity/pressure_load.hpp>
#include <mast/optimization/topology/simp/penalized_density.hpp>
#include <mast/optimization/topology/simp/penalized_youngs_modulus.hpp>
#include <mast/optimization/topology/simp/libmesh/residual_and_jacobian.hpp>
#include <mast/optimization/topology/simp/libmesh/assemble_output_sensitivity.hpp>
#include <mast/optimization/topology/simp/libmesh/volume.hpp>
#include <mast/numerics/libmesh/sparse_matrix_initialization.hpp>
#include <mast/util/getpot_wrapper.hpp>
#include <mast/mesh/libmesh/geometric_filter.hpp>
#include <mast/optimization/design_parameter.hpp>
#include <mast/optimization/solvers/gcmma_interface.hpp>
#include <mast/optimization/utility/design_history.hpp>

// topology optimization benchmark cases
#include <mast/mesh/generation/bracket2d.hpp>

// libMesh includes
#include <libmesh/replicated_mesh.h>
#include <libmesh/elem.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/boundary_info.h>
#include <libmesh/exodusII_io.h>

// Eigen includes
#include <Eigen/SparseLU>

// BEGIN_TRANSLATE SIMP Minimum Compliance Topology Optimization


namespace MAST {
namespace Examples {
namespace Structural {
namespace Example5 {

template <typename ModelType>
class InitExample {
    
public:

    InitExample(libMesh::Parallel::Communicator &mpi_comm,
                MAST::Utility::GetPotWrapper    &inp):
    comm      (mpi_comm),
    input     (inp),
    model     (new ModelType),
    q_type    (libMesh::QGAUSS),
    q_order   (libMesh::SECOND),
    fe_order  (libMesh::FIRST),
    fe_family (libMesh::LAGRANGE),
    mesh      (new libMesh::ReplicatedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    rho_sys   (&eq_sys->add_system<libMesh::ExplicitSystem>("density")),
    filter    (nullptr),
    p_side_id (-1),
    penalty   (0.) {
        
        model->init_analysis_mesh(*this, *mesh);

        // displacement variables for elasticity solution
        sys->add_variable("u_x", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("u_y", libMesh::FEType(fe_order, fe_family));
        
        // density field
        rho_sys->add_variable("rho", libMesh::FEType(fe_order, fe_family));
        
        model->init_analysis_dirichlet_conditions(*this);
        
        eq_sys->init();
        
        real_t
        filter_r = input("filter_radius",
                         "radius of geometric filter for level set field", 0.015);
        filter = new MAST::Mesh::libMeshWrapper::GeometricFilter(*rho_sys, filter_r);
        
        mesh->print_info(std::cout);
        eq_sys->print_info(std::cout);
        
        penalty  = input("rho_penalty",
                         "SIMP modulus of elasticity penalty", 3.);
    }
    
    virtual ~InitExample() {
        
        delete eq_sys;
        delete mesh;
        delete model;
        delete filter;
    }
    
    libMesh::Parallel::Communicator             &comm;
    MAST::Utility::GetPotWrapper                &input;
    ModelType                                   *model;
    libMesh::QuadratureType                      q_type;
    libMesh::Order                               q_order;
    libMesh::Order                               fe_order;
    libMesh::FEFamily                            fe_family;
    libMesh::ReplicatedMesh                     *mesh;
    libMesh::EquationSystems                    *eq_sys;
    libMesh::NonlinearImplicitSystem            *sys;
    libMesh::ExplicitSystem                     *rho_sys;
    MAST::Mesh::libMeshWrapper::GeometricFilter *filter;
    uint_t                                       p_side_id;
    real_t                                       penalty;
};



template <typename TraitsType>
class Context {
    
public:
    using model_t = typename TraitsType::model_t;
    
    Context(InitExample<model_t>& init):
    ex_init   (init),
    mesh      (init.mesh),
    eq_sys    (init.eq_sys),
    sys       (init.sys),
    rho_sys   (init.rho_sys),
    elem      (nullptr),
    qp        (-1),
    fe        (nullptr)
    { }
    
    virtual ~Context() { }
    
    // assembly methods
    uint_t  elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    real_t  qp_location(uint_t i) const { return fe->xyz(qp, i);}
    inline bool elem_is_quad() const {return (elem->type() == libMesh::QUAD4 ||
                                              elem->type() == libMesh::QUAD8 ||
                                              elem->type() == libMesh::QUAD9);}
    inline bool if_compute_pressure_load_on_side(const uint_t s)
    { return ex_init.mesh->boundary_info->has_boundary_id(elem, s, ex_init.p_side_id);}
    
    
    InitExample<model_t>                &ex_init;
    libMesh::ReplicatedMesh             *mesh;
    libMesh::EquationSystems            *eq_sys;
    libMesh::NonlinearImplicitSystem    *sys;
    libMesh::ExplicitSystem             *rho_sys;
    const libMesh::Elem                 *elem;
    uint_t                               qp;
    typename TraitsType::fe_shape_t     *fe;
};



template <typename BasisScalarType,
typename NodalScalarType,
typename SolScalarType,
typename ModelType>
struct Traits {
    
    static const uint_t dim = ModelType::dim;
    using traits_t          = MAST::Examples::Structural::Example5::Traits<BasisScalarType, NodalScalarType, SolScalarType, ModelType>;
    using model_t           = ModelType;
    using context_t         = Context<traits_t>;
    using ex_init_t         = InitExample<model_t>;
    using scalar_t          = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using fe_basis_t        = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, dim>;
    using fe_shape_t        = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, dim, dim, fe_basis_t>;
    using fe_data_t         = typename MAST::FEBasis::libMeshWrapper::FEData<dim, fe_basis_t, fe_shape_t>;
    using fe_side_data_t    = typename MAST::FEBasis::libMeshWrapper::FESideData<dim, fe_basis_t, fe_shape_t>;
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, dim, dim, context_t, fe_shape_t>;
    using density_fe_var_t  = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 1, dim, context_t, fe_shape_t>;
    using density_field_t   = typename MAST::FEBasis::ScalarFieldWrapper<scalar_t, density_fe_var_t>;
    using density_t         = typename MAST::Optimization::Topology::SIMP::PenalizedDensity<SolScalarType, density_field_t>;
    using modulus_t         = typename MAST::Optimization::Topology::SIMP::PenalizedYoungsModulus<SolScalarType, density_t>;
    using nu_t              = typename MAST::Base::ScalarConstant<SolScalarType>;
    using press_t           = typename ModelType::template pressure_t<scalar_t>;
    using area_t            = typename MAST::Base::ScalarConstant<SolScalarType>;
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, dim, modulus_t, nu_t, context_t>;
    using energy_t          = typename MAST::Physics::Elasticity::LinearContinuum::StrainEnergy<fe_var_t, prop_t, dim, context_t>;
    using press_load_t      = typename MAST::Physics::Elasticity::SurfacePressureLoad<fe_var_t, press_t, area_t, dim, context_t>;
    using element_vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using element_matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using assembled_vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using assembled_matrix_t = Eigen::SparseMatrix<scalar_t>;
};



template <typename TraitsType>
class ElemOps {
    
public:
    
    using scalar_t  = typename TraitsType::scalar_t;
    using context_t = typename TraitsType::context_t;
    using vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    ElemOps(context_t  &c):
    density         (nullptr),
    E               (nullptr),
    nu              (nullptr),
    press           (nullptr),
    area            (nullptr),
    _fe_data        (nullptr),
    _fe_side_data   (nullptr),
    _fe_var         (nullptr),
    _fe_side_var    (nullptr),
    _density_fe_var (nullptr),
    _density_field  (nullptr),
    _prop           (nullptr),
    _energy         (nullptr),
    _p_load         (nullptr) {
        
        _fe_data       = new typename TraitsType::fe_data_t;
        _fe_data->init(c.ex_init.q_order,
                       c.ex_init.q_type,
                       c.ex_init.fe_order,
                       c.ex_init.fe_family);
        _fe_side_data  = new typename TraitsType::fe_side_data_t;
        _fe_side_data->init(c.ex_init.q_order,
                            c.ex_init.q_type,
                            c.ex_init.fe_order,
                            c.ex_init.fe_family);
        _fe_var        = new typename TraitsType::fe_var_t;
        _fe_side_var   = new typename TraitsType::fe_var_t;
        
        _density_fe_var      = new typename TraitsType::density_fe_var_t;
        _density_sens_fe_var = new typename TraitsType::density_fe_var_t;
        _density_field       = new typename TraitsType::density_field_t;
        
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
        
        _density_fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _density_sens_fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _density_field->set_fe_object_and_component(*_density_fe_var, 0);
        _density_field->set_derivative_fe_object_and_component(*_density_sens_fe_var, 0);

        // variables for physics
        density  = new typename TraitsType::density_t;
        E        = new typename TraitsType::modulus_t;
        nu       = new typename TraitsType::nu_t(0.33);
        press    = c.ex_init.model->template build_pressure_load<scalar_t, typename TraitsType::ex_init_t>(c.ex_init).release();
        area     = new typename TraitsType::area_t(1.0);
        _prop    = new typename TraitsType::prop_t;
        
        density->set_penalty(c.ex_init.penalty);
        density->set_density_field(*_density_field);
        E->set_density(*density);
        E->set_modulus(72.e9, 72.e2);
        
        _prop->set_modulus_and_nu(*E, *nu);
        _energy   = new typename TraitsType::energy_t;
        _energy->set_section_property(*_prop);
        _p_load   = new typename TraitsType::press_load_t;
        _p_load->set_section_area(*area);
        _p_load->set_pressure(*press);
        
        // tell physics kernels about the FE discretization information
        _energy->set_fe_var_data(*_fe_var);
        _p_load->set_fe_var_data(*_fe_side_var);
    }
    
    virtual ~ElemOps() {
        
        delete area;
        delete press;
        delete nu;
        delete E;
        delete density;

        delete _energy;
        delete _prop;
        delete _p_load;

        delete _density_field;
        delete _density_fe_var;
        
        delete _fe_var;
        delete _fe_side_var;
        delete _fe_side_data;
        delete _fe_data;
    }
    
    
    template <typename ContextType,
              typename Accessor1Type,
              typename Accessor2Type>
    inline void compute(ContextType                       &c,
                        const Accessor1Type               &sol_v,
                        const Accessor2Type               &density_v,
                        typename TraitsType::element_vector_t &res,
                        typename TraitsType::element_matrix_t *jac) {
        

        c.fe = &_fe_data->fe_derivative();
        _fe_data->reinit(c);
        _fe_var->init(c, sol_v);
        _density_fe_var->init(c, density_v);

        _energy->compute(c, res, jac);
        
        for (uint_t s=0; s<c.elem->n_sides(); s++)
            if (c.if_compute_pressure_load_on_side(s)) {
                
                c.fe = &_fe_side_data->fe_derivative();
                _fe_side_data->reinit_for_side(c, s);
                _fe_side_var->init(c, sol_v);
                _p_load->compute(c, res, jac);
            }
    }
    
    
    template <typename ContextType,
              typename Accessor1Type,
              typename Accessor2Type,
              typename Accessor3Type,
              typename ScalarFieldType>
    inline void derivative(ContextType                       &c,
                           const ScalarFieldType             &f,
                           const Accessor1Type               &sol_v,
                           const Accessor2Type               &density_v,
                           const Accessor3Type               &density_sens,
                           typename TraitsType::element_vector_t &res,
                           typename TraitsType::element_matrix_t *jac) {
        
        c.fe = &_fe_data->fe_derivative();
        _fe_data->reinit(c);
        _fe_var->init(c, sol_v);
        _density_fe_var->init(c, density_v);
        _density_sens_fe_var->init(c, density_sens);

        _energy->derivative(c, f, res, jac);
    }

    
    template <typename ContextType,
              typename Accessor1Type,
              typename Accessor2Type,
              typename Accessor3Type,
              typename ScalarFieldType>
    inline scalar_t
    derivative(ContextType                       &c,
               const ScalarFieldType             &f,
               const Accessor1Type               &sol_v,
               const Accessor2Type               &density_v,
               const Accessor3Type               &density_sens) {

        // nothing to be done here since external work done due to pressure
        // is independent of topology parameter
        return 0.;
    }

    
    // parameters
    typename TraitsType::density_t    *density;
    typename TraitsType::modulus_t    *E;
    typename TraitsType::nu_t         *nu;
    typename TraitsType::press_t      *press;
    typename TraitsType::area_t       *area;
    
private:
    
    // variables for quadrature and shape function
    typename TraitsType::fe_data_t         *_fe_data;
    typename TraitsType::fe_side_data_t    *_fe_side_data;
    typename TraitsType::fe_var_t          *_fe_var;
    typename TraitsType::fe_var_t          *_fe_side_var;
    typename TraitsType::density_fe_var_t  *_density_fe_var;
    typename TraitsType::density_fe_var_t  *_density_sens_fe_var;
    typename TraitsType::density_field_t   *_density_field;
    typename TraitsType::prop_t            *_prop;
    typename TraitsType::energy_t          *_energy;
    typename TraitsType::press_load_t      *_p_load;
};



template <typename TraitsType>
class FunctionEvaluation {
    
public:
    
    using scalar_t  = typename TraitsType::scalar_t;
    using context_t = typename TraitsType::context_t;
    
    FunctionEvaluation(ElemOps<TraitsType> &e_ops,
                       context_t           &c):
    _e_ops        (e_ops),
    _c            (c),
    _volume       (_c.ex_init.model->reference_volume(_c.ex_init)),
    _vf           (_c.ex_init.input("volume_fraction",
                                    "upper limit for the volume fraction", 0.2)) {
        
        // initialize the design variable vector
        _c.ex_init.model->init_simp_dvs(_c.ex_init, _dvs);
    }
    
    virtual ~FunctionEvaluation() {}
    
    
    inline uint_t n_vars() const {return _dvs.size();}
    inline uint_t   n_eq() const {return 0;}
    inline uint_t n_ineq() const {return 1;}
    virtual void init_dvar(std::vector<scalar_t>& x,
                           std::vector<scalar_t>& xmin,
                           std::vector<scalar_t>& xmax) {

        Assert1(_dvs.size(), _dvs.size(), "Design variables must be initialized");
        
        x.resize(_dvs.size());
        xmin.resize(_dvs.size());
        xmax.resize(_dvs.size());
        
        std::fill(xmin.begin(), xmin.end(),      0.);
        std::fill(xmax.begin(), xmax.end(),    1.e0);

        //
        // now, check if the user asked to initialize dvs from a previous file
        //
        std::string
        nm    =  _c.ex_init.input("restart_optimization_file",
                                  "filename with optimization history for restart",
                                  "");
        
        if (nm.length()) {
            
            uint_t
            iter = _c.ex_init.input("restart_optimization_iter",
                                    "restart iteration number from file", 0);
            MAST::Optimization::Utility::initialize_dv_from_output_file(*this,
                                                                        nm,
                                                                        iter,
                                                                        x);
        }
        else {
            
            for (uint_t i=0; i<_dvs.size(); i++)
                x[i] = _dvs[i]();
        }
    }
    
    
    virtual void evaluate(const std::vector<scalar_t> &x,
                          scalar_t                    &obj,
                          bool                         eval_obj_grad,
                          std::vector<scalar_t>       &obj_grad,
                          std::vector<scalar_t>       &fvals,
                          std::vector<bool>           &eval_grads,
                          std::vector<scalar_t>       &grads) {

        std::cout << "New Evaluation" << std::endl;
        
        Assert2(x.size() == _dvs.size(),
                x.size(), _dvs.size(),
                "Incompatible design variable vector size.");

        libMesh::ExplicitSystem
        &str_sys = *_c.ex_init.sys,
        &rho_sys = *_c.ex_init.rho_sys;

        const uint_t
        n_dofs          = str_sys.n_dofs(),
        n_rho_vals      = rho_sys.n_dofs(),
        first_local_rho = rho_sys.get_dof_map().first_dof(rho_sys.comm().rank()),
        last_local_rho  = rho_sys.get_dof_map().end_dof(rho_sys.comm().rank());
        
        
        typename TraitsType::assembled_vector_t
        rho_base     = TraitsType::assembled_vector_t::Ones(n_rho_vals),
        rho_filtered = TraitsType::assembled_vector_t::Zero(n_rho_vals),
        res          = TraitsType::assembled_vector_t::Zero(n_dofs),
        sol          = TraitsType::assembled_vector_t::Zero(n_dofs);
        
        typename TraitsType::assembled_matrix_t
        jac;
        
        for (uint_t i=0; i<_dvs.size(); i++) {
            
            uint_t dof_id = _dvs.template get_parameter_for_dv<int>(i, "dof_id");
            
            if (dof_id >= first_local_rho && dof_id <  last_local_rho)
                rho_base(dof_id) = x[i];
        }
        //base_phi.close();
        _c.ex_init.filter->template compute_filtered_values
        <scalar_t,
        typename TraitsType::assembled_vector_t,
        typename TraitsType::assembled_vector_t>
        (_dvs, rho_base, rho_filtered);

        // this will create a localized vector in _level_set_sys->curret_local_solution
        //_density_sys->update();
        
        //_sys.solution->zero();
        
        //////////////////////////////////////////////////////////////////////
        // check to see if the sensitivity of constraint is requested
        //////////////////////////////////////////////////////////////////////
        bool if_grad_sens = false;
        for (uint_t i=0; i<eval_grads.size(); i++)
            if_grad_sens = (if_grad_sens || eval_grads[i]);
        
        //*********************************************************************
        // DO NOT zero out the gradient vector, since GCMMA needs it for the  *
        // subproblem solution                                                *
        //*********************************************************************
        
        std::cout << "Static Solve" << std::endl;

        // set the elasticity penalty for solution
        //_Ef->set_penalty_val(penalty);

        MAST::Optimization::Topology::SIMP::libMeshWrapper::ResidualAndJacobian<scalar_t, ElemOps<TraitsType>>
        assembly;
        
        assembly.set_elem_ops(_e_ops);
        
        MAST::Numerics::libMeshWrapper::init_sparse_matrix(str_sys.get_dof_map(), jac);
        
        // the residual is assembled as \f$ R(x) = K x - f \f$. Since \f$ x= 0 \f$ we have
        // \f$ R(x) = - f \f$.
        assembly.assemble(_c, sol, rho_filtered, &res, &jac);
        // We multiply with -1 before solving for \f$ x \f$.
        res *= -1;
        // This solves for \f$ x\f$ from the system of equations \f$ K x = f \f$.
        sol = Eigen::SparseLU<typename TraitsType::assembled_matrix_t>(jac).solve(res);

        {
            libMesh::ExodusII_IO writer(*_c.mesh);
            Eigen::Matrix<real_t, Eigen::Dynamic, 1> sol_r = sol.real();
            for (uint_t i=0; i<sol.size(); i++) _c.sys->solution->set(i, sol_r(i));

            sol_r = rho_base.real();
            for (uint_t i=0; i<rho_filtered.size(); i++) _c.rho_sys->solution->set(i, sol_r(i));

            writer.write_timestep("solution.exo", *_c.eq_sys, 1, 1.);

            sol_r = rho_filtered.real();
            for (uint_t i=0; i<rho_filtered.size(); i++) _c.rho_sys->solution->set(i, sol_r(i));
            writer.write_timestep("solution.exo", *_c.eq_sys, 2, 2.);
        }

        // compliance is defined using the external work done \f$ c = x^T f \f$
        scalar_t
        vol    = 0.,
        comp   = sol.dot(res);

        // ask the system to update so that the localized solution is available for
        // further processing
        //_sys->update();

        //////////////////////////////////////////////////////////////////////
        // evaluate the functions
        //////////////////////////////////////////////////////////////////////
        
        // evaluate the volume for used in the problem setup
        MAST::Optimization::Topology::SIMP::libMeshWrapper::Volume<scalar_t>
        volume;
        
        vol = volume.compute(_c, rho_filtered);
        std::cout << "volume: " << vol << std::endl;
        
        // evaluate the output based on specified problem type
        //nonlinear_assembly.calculate_output(*_sys->current_local_solution, false, compliance);
        //comp      = compliance.output_total();
        obj       = comp;
        fvals[0]  = vol/_volume - _vf; // vol/vol0 - a <=
        std::cout << "compliance: " << comp << std::endl;
        

        //////////////////////////////////////////////////////////////////////
        // evaluate the objective sensitivities, if requested
        //////////////////////////////////////////////////////////////////////
        if (eval_obj_grad) {
            
            MAST::Optimization::Topology::SIMP::libMeshWrapper::AssembleOutputSensitivity
            <scalar_t, ElemOps<TraitsType>, ElemOps<TraitsType>>
            compliance_sens;
            
            compliance_sens.set_elem_ops(_e_ops, _e_ops);

            // the adjoint solution for compliance is the negative of displacement. We copy the
            // negative of solution in vector \p res.
            res = -sol;
            
            // This solves for the sensitivity of compliance, \f$ c=x^T f \f$, with respect to
            // a parameter \f$ \alpha \f$.
            // \f{eqnarray*}{ \frac{dc}{d\alpha}
            //    & = & \frac{\partial c}{\partial \alpha} + \frac{\partial c}{\partial x}
            //     \frac{dx}{d\alpha} \\
            //    & = & \frac{\partial c}{\partial \alpha} +
            //     \lambda^T \frac{\partial R(x)}{\partial \alpha}
            // \f}
            // Note that the adjoint solution for compliance is \f$ \lambda = -x \f$.
            compliance_sens.assemble(_c,
                                     sol,                 // solution
                                     rho_filtered,        // filtered density
                                     res,                 // adjoint solution
                                     *_c.ex_init.filter,  // geometric filter
                                     _dvs,
                                     obj_grad);
        }
        
        
        //////////////////////////////////////////////////////////////////////
        // evaluate the sensitivities for constraints
        //////////////////////////////////////////////////////////////////////
        if (if_grad_sens) {
            
            //////////////////////////////////////////////////////////////////
            // indices used by GCMMA follow this rule:
            // grad_k = dfi/dxj  ,  where k = j*NFunc + i
            //////////////////////////////////////////////////////////////////

            volume.derivative(_c,
                              rho_filtered,
                              *_c.ex_init.filter,
                              _dvs,
                              grads);
            for (uint_t i=0; i<grads.size(); i++)
                grads[i] /= _volume;
        }
        
        //
        // also the stress data for plotting
        //
        //_Ef->set_penalty_val(stress_penalty);
        //stress_assembly.update_stress_strain_data(stress, *_sys->solution);
    }
    
    inline void output(const uint_t                iter,
                       const std::vector<real_t>  &dvars,
                       real_t                     &o,
                       std::vector<real_t>        &fvals) {
        
    }
    
private:
    
    ElemOps<TraitsType>                                 &_e_ops;
    context_t                                           &_c;
    MAST::Optimization::DesignParameterVector<scalar_t>  _dvs;
    real_t                                               _volume;
    real_t                                               _vf;
};
} // namespace Example5
} // namespace Structural
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

int main(int argc, char** argv) {
    
    libMesh::LibMeshInit init(argc, argv);
    MAST::Utility::GetPotWrapper input(argc, argv);
    
    using model_t            = MAST::Mesh::Generation::Bracket2D;
    
    using traits_t           = MAST::Examples::Structural::Example5::Traits<real_t, real_t,    real_t, model_t>;
    using elem_ops_t         = MAST::Examples::Structural::Example5::ElemOps<traits_t>;
    using func_eval_t        = MAST::Examples::Structural::Example5::FunctionEvaluation<traits_t>;

    traits_t::ex_init_t ex_init(init.comm(), input);

    traits_t::context_t  c(ex_init);
    elem_ops_t           e_ops(c);
    func_eval_t          f_eval(e_ops, c);
    
    /*using traits_complex_t   = MAST::Examples::Structural::Example5::Traits<real_t, real_t, complex_t, model_t>;
    using elem_ops_complex_t = MAST::Examples::Structural::Example5::ElemOps<traits_complex_t>;
    using func_eval_complex_t= MAST::Examples::Structural::Example5::FunctionEvaluation<traits_complex_t>;

    traits_complex_t::context_t  c_cmplx(ex_init);
    elem_ops_complex_t           e_ops_c(c_cmplx);
    func_eval_complex_t          f_eval_c(e_ops_c, c_cmplx);

    std::vector<real_t>
    dvs (f_eval_c.n_vars(), 0.),
    xlow (f_eval_c.n_vars(), 0.),
    xup  (f_eval_c.n_vars(), 0.),
    g    (1, 0.),
    o_sens(f_eval_c.n_vars(), 0.),
    g_sens(f_eval_c.n_vars(), 0.),
    o_cs(f_eval_c.n_vars(), 0.),
    g_cs(f_eval_c.n_vars(), 0.);

    std::vector<complex_t>
    dvs_c (f_eval_c.n_vars(), 0.),
    xlow_c (f_eval_c.n_vars(), 0.),
    xup_c  (f_eval_c.n_vars(), 0.),
    g_c    (1, 0.),
    d_c;
    
    real_t
    obj;
    
    complex_t
    obj_c;
    
    std::vector<bool> b_vec(1, true);
    
    f_eval.init_dvar(dvs, xlow, xup);
    f_eval.evaluate(dvs, obj, true, o_sens, g, b_vec, g_sens);

    b_vec[0] = false;
    f_eval_c.init_dvar(dvs_c, xlow_c, xup_c);
    
    std::cout << "obj: " << obj << std::endl;
    
    for (uint_t i=0; i<f_eval.n_vars(); i++) {
        
        dvs_c[i] += complex_t(0., ComplexStepDelta);
        f_eval_c.evaluate(dvs_c, obj_c, false, d_c, g_c, b_vec, d_c);
        dvs_c[i] -= complex_t(0., ComplexStepDelta);
        o_cs[i] = obj_c.imag() /ComplexStepDelta;
        g_cs[i] = g_c[0].imag()/ComplexStepDelta;
        
        std::cout
        << std::setw(5) << i
        << std::setw(20) << o_sens[i]
        << std::setw(20) << o_cs[i]
        << std::setw(20) << std::fabs(o_sens[i]-o_cs[i])
        << std::setw(20) << g_sens[i]
        << std::setw(20) << g_cs[i]
        << std::setw(20) << std::fabs(g_sens[i]-g_cs[i]) << std::endl;
    }*/

    
    // create an optimizer, attach the function evaluation
    MAST::Optimization::Solvers::GCMMAInterface<func_eval_t> optimizer;
    optimizer.max_inner_iters = 6;
    optimizer.constr_penalty  = 1.e5;
    optimizer.set_function_evaluation(f_eval);
    
    // optimize
    optimizer.optimize();
    
    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
