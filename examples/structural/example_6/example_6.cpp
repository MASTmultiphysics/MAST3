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
#include <mast/base/scalar_constant.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/libmesh/fe_side_data.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/fe/scalar_field_wrapper.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/physics/elasticity/linear_strain_energy.hpp>
#include <mast/physics/elasticity/pressure_load.hpp>
#include <mast/physics/elasticity/linear_thermoelastic_load.hpp>
#include <mast/physics/elasticity/libmesh/mat_null_space.hpp>
#include <mast/optimization/topology/simp/heaviside_filter.hpp>
#include <mast/optimization/topology/simp/penalized_density.hpp>
#include <mast/optimization/topology/simp/penalized_scalar.hpp>
#include <mast/optimization/topology/simp/libmesh/residual_and_jacobian.hpp>
#include <mast/optimization/topology/simp/libmesh/assemble_output_sensitivity.hpp>
#include <mast/optimization/topology/simp/libmesh/volume.hpp>
#include <mast/optimization/design_parameter.hpp>
#include <mast/optimization/solvers/gcmma_interface.hpp>
#include <mast/optimization/utility/design_history.hpp>
#include <mast/util/getpot_wrapper.hpp>
#include <mast/mesh/libmesh/geometric_filter.hpp>
#include <mast/solvers/petsc/linear_solver.hpp>

// topology optimization benchmark cases
#include <mast/mesh/generation/truss2d.hpp>
#include <mast/mesh/generation/truss3d.hpp>
#include <mast/mesh/generation/inplane2d.hpp>
#include <mast/mesh/generation/bracket2d.hpp>
#include <mast/mesh/generation/bracket3d.hpp>
#include <mast/mesh/generation/panel2d.hpp>
#include <mast/mesh/generation/panel3d.hpp>

// libMesh includes
#include <libmesh/distributed_mesh.h>
#include <libmesh/elem.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/boundary_info.h>
#include <libmesh/nemesis_io.h>
#include <libmesh/petsc_matrix.h>


// BEGIN_TRANSLATE SIMP Minimum Compliance Topology Optimization with MPI based solvers


namespace MAST {
namespace Examples {
namespace Structural {
namespace Example6 {

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
    mesh      (new libMesh::DistributedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    rho_sys   (&eq_sys->add_system<libMesh::ExplicitSystem>("density")),
    filter    (nullptr),
    p_side_id (-1),
    penalty   (0.),
    beta      (0.),
    eta       (0.) {
        
        std::string
        t = input("q_order", "quadrature order", "second");
        q_order = libMesh::Utility::string_to_enum<libMesh::Order>(t);

        t = input("fe_order", "finite element interpolation order", "first");
        fe_order = libMesh::Utility::string_to_enum<libMesh::Order>(t);

        model->init_analysis_mesh(*this, *mesh);

        // displacement variables for elasticity solution
        sys->add_variable("u_x", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("u_y", libMesh::FEType(fe_order, fe_family));
        if (ModelType::dim == 3)
            sys->add_variable("u_z", libMesh::FEType(fe_order, fe_family));
        
        // density field
        rho_sys->add_variable("rho", libMesh::FEType(fe_order, fe_family));
        
        model->init_analysis_dirichlet_conditions(*this);
        
        eq_sys->init();
        
        mesh->print_info(std::cout);
        eq_sys->print_info(std::cout);

        real_t
        filter_r = input("filter_radius",
                         "radius of geometric filter for level set field", 0.015);
        filter = new MAST::Mesh::libMeshWrapper::GeometricFilter(*rho_sys, filter_r);
        eq_sys->reinit();

        // create and attach the null space to the matrix
        MAST::Physics::Elasticity::libMeshWrapper::NullSpace
        null_sp(*sys, ModelType::dim);
        
        Mat m = dynamic_cast<libMesh::PetscMatrix<real_t>*>(sys->matrix)->mat();
        null_sp.attach_to_matrix(m);
        
        eta      = input("heaviside_eta",
                         "Smoothed heaviside eta parameter", 0.5);
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
    libMesh::DistributedMesh                    *mesh;
    libMesh::EquationSystems                    *eq_sys;
    libMesh::NonlinearImplicitSystem            *sys;
    libMesh::ExplicitSystem                     *rho_sys;
    MAST::Mesh::libMeshWrapper::GeometricFilter *filter;
    uint_t                                       p_side_id;
    real_t                                       penalty;
    real_t                                       beta;
    real_t                                       eta;
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
    inline bool elem_is_hex() const  {return (elem->type() == libMesh::HEX8 ||
                                              elem->type() == libMesh::HEX20 ||
                                              elem->type() == libMesh::HEX27);}
    inline bool if_compute_pressure_load_on_side(const uint_t s)
    { return ex_init.mesh->boundary_info->has_boundary_id(elem, s, ex_init.p_side_id);}
    
    
    InitExample<model_t>                &ex_init;
    libMesh::DistributedMesh            *mesh;
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
    using traits_t          = MAST::Examples::Structural::Example6::Traits<BasisScalarType, NodalScalarType, SolScalarType, ModelType>;
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
    using heaviside_t       = typename MAST::Optimization::Topology::SIMP::HeavisideFilter<SolScalarType, density_field_t>;
    using density_t         = typename MAST::Optimization::Topology::SIMP::PenalizedDensity<SolScalarType, heaviside_t>;
    using modulus_t         = typename MAST::Optimization::Topology::SIMP::PenalizedScalar<SolScalarType, density_t>;
    using nu_t              = typename MAST::Base::ScalarConstant<SolScalarType>;
    using alpha_t           = typename MAST::Base::ScalarConstant<SolScalarType>;
    using press_t           = typename ModelType::template pressure_t<scalar_t>;
    using temp_t            = typename MAST::Optimization::Topology::SIMP::PenalizedScalar<SolScalarType, density_t>;
    using area_t            = typename MAST::Base::ScalarConstant<SolScalarType>;
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, dim, modulus_t, nu_t, context_t>;
    using energy_t          = typename MAST::Physics::Elasticity::LinearContinuum::StrainEnergy<fe_var_t, prop_t, dim, context_t>;
    using press_load_t      = typename MAST::Physics::Elasticity::SurfacePressureLoad<fe_var_t, press_t, area_t, dim, context_t>;
    using temp_load_t       = typename MAST::Physics::Elasticity::LinearContinuum::ThermoelasticLoad<fe_var_t, temp_t, alpha_t, prop_t, dim, context_t>;
    using element_vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using element_matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using assembled_vector_t = libMesh::NumericVector<scalar_t>;
    using assembled_matrix_t = libMesh::SparseMatrix<scalar_t>;
};



template <typename TraitsType>
class ElemOps {
    
public:
    
    using scalar_t  = typename TraitsType::scalar_t;
    using context_t = typename TraitsType::context_t;
    using vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    
    ElemOps(context_t  &c):
    heaviside       (nullptr),
    density         (nullptr),
    E               (nullptr),
    dt              (nullptr),
    nu              (nullptr),
    alpha           (nullptr),
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
    _temp_load      (nullptr),
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
        heaviside= new typename TraitsType::heaviside_t;
        density  = new typename TraitsType::density_t;
        E        = new typename TraitsType::modulus_t;
        dt       = new typename TraitsType::temp_t;
        nu       = new typename TraitsType::nu_t(0.33);
        alpha    = new typename TraitsType::alpha_t(2.e-5);
        press    = c.ex_init.model->template build_pressure_load<scalar_t, typename TraitsType::ex_init_t>(c.ex_init).release();
        area     = new typename TraitsType::area_t(1.0);
        _prop    = new typename TraitsType::prop_t;
        
        heaviside->set_parameters(c.ex_init.beta, c.ex_init.eta);
        heaviside->set_field(*_density_field);
        density->set_penalty(c.ex_init.penalty);
        density->set_density_field(*heaviside);
        E->set_density(*density);
        E->set_scalar(72.e9, 72.e2);
        dt->set_density(*density);
        dt->set_scalar(c.ex_init.input("temperature",
                                       "temperature over domain", 0.), 0.);

        _prop->set_modulus_and_nu(*E, *nu);
        _energy   = new typename TraitsType::energy_t;
        _energy->set_section_property(*_prop);
        _p_load   = new typename TraitsType::press_load_t;
        _p_load->set_section_area(*area);
        _p_load->set_pressure(*press); 
        _temp_load = new typename TraitsType::temp_load_t;
        _temp_load->set_section_property(*_prop);
        _temp_load->set_coeff_thermal_expansion(*alpha);
        _temp_load->set_temperature(*dt);
        
        // tell physics kernels about the FE discretization information
        _energy->set_fe_var_data(*_fe_var);
        _p_load->set_fe_var_data(*_fe_side_var);
        _temp_load->set_fe_var_data(*_fe_var);
    }
    
    virtual ~ElemOps() {
        
        delete area;
        delete press;
        delete nu;
        delete alpha;
        delete E;
        delete dt;
        delete density;
        delete heaviside;

        delete _energy;
        delete _prop;
        delete _p_load;
        delete _temp_load;

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
        _temp_load->compute(c, res, jac);
        
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
        _temp_load->derivative(c, f, res, jac);
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

        // pressure load is independent of design parameters but thermoelastic load
        // depends on it. So, we compute the partial derivative of compliance contribution
        // from that term
        c.fe = &_fe_data->fe_derivative();
        _fe_data->reinit(c);
        _fe_var->init(c, sol_v);
        _density_fe_var->init(c, density_v);
        _density_sens_fe_var->init(c, density_sens);

        typename TraitsType::element_vector_t
        res = TraitsType::element_vector_t::Zero(_temp_load->n_dofs());
        
        _temp_load->derivative(c, f, res);
        
        return -sol_v.dot(res);
    }

    
    // parameters
    typename TraitsType::heaviside_t  *heaviside;
    typename TraitsType::density_t    *density;
    typename TraitsType::modulus_t    *E;
    typename TraitsType::temp_t       *dt;
    typename TraitsType::nu_t         *nu;
    typename TraitsType::alpha_t      *alpha;
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
    typename TraitsType::temp_load_t       *_temp_load;
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
    _dvs          (new MAST::Optimization::DesignParameterVector<scalar_t>(c.rho_sys->comm())),
    _volume       (_c.ex_init.model->reference_volume(_c.ex_init)),
    _vf           (_c.ex_init.input("volume_fraction",
                                    "upper limit for the volume fraction", 0.2)) {
        
        // initialize the design variable vector
        _c.ex_init.model->init_simp_dvs(_c.ex_init, *_dvs);
        
        // open the file where the history will be stored
        _history.open("optim_history.txt", std::ostream::out);
    }
    
    virtual ~FunctionEvaluation() {
        
        _history.close();
        delete _dvs;
    }
    
    
    inline uint_t n_vars() const {return _dvs->size();}
    inline uint_t   n_eq() const {return 0;}
    inline uint_t n_ineq() const {return 1;}
    virtual void init_dvar(std::vector<scalar_t>& x,
                           std::vector<scalar_t>& xmin,
                           std::vector<scalar_t>& xmax) {

        Assert1(_dvs->size(), _dvs->size(), "Design variables must be initialized");
        
        x.resize(_dvs->size(), 0.);
        xmin.resize(_dvs->size());
        xmax.resize(_dvs->size());
        
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
            
            for (uint_t i=_dvs->local_begin(); i<_dvs->local_end(); i++)
                x[i] = (*_dvs)[i]();
            
            _c.rho_sys->comm().sum(x);
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
        
        Assert2(x.size() == _dvs->size(),
                x.size(), _dvs->size(),
                "Incompatible design variable vector size.");

        libMesh::ExplicitSystem
        &str_sys = *_c.ex_init.sys,
        &rho_sys = *_c.ex_init.rho_sys;

        const uint_t
        n_dofs          = str_sys.n_dofs(),
        n_rho_vals      = rho_sys.n_dofs(),
        first_local_rho = rho_sys.get_dof_map().first_dof(rho_sys.comm().rank()),
        last_local_rho  = rho_sys.get_dof_map().end_dof(rho_sys.comm().rank());
        
        
        std::unique_ptr<typename TraitsType::assembled_vector_t>
        rho_base(_c.rho_sys->current_local_solution->clone().release()),
        res(_c.sys->solution->clone().release());

        // set a unit value for density. Values provided by design parameters
        // will be overwritten in \p rho_base.
        *rho_base = 1.;
        

        for (uint_t i=_dvs->local_begin(); i<_dvs->local_end(); i++)
            rho_base->set(_dvs->get_data_for_parameter((*_dvs)[i]).template get<int>("dof_id"),
                          x[i]);
        rho_base->close();
        
        _c.ex_init.filter->compute_filtered_values
        (dynamic_cast<libMesh::PetscVector<scalar_t>*>(rho_base.get())->vec(),
         dynamic_cast<libMesh::PetscVector<scalar_t>*>(_c.rho_sys->solution.get())->vec());
        _c.rho_sys->solution->close();
        
        // this will copy the solution to libMesh::System::current_local_soluiton
        _c.rho_sys->update();

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

        MAST::Optimization::Topology::SIMP::libMeshWrapper::ResidualAndJacobian<scalar_t, ElemOps<TraitsType>>
        assembly;
        
        assembly.set_elem_ops(_e_ops);
        _c.sys->solution->zero();
        
        // this will copy the solution to libMesh::System::current_local_soluiton
        _c.sys->update();
        
        // the residual is assembled as \f$ R(x) = K x - f \f$. Since \f$ x= 0 \f$ we have
        // \f$ R(x) = - f \f$.
        assembly.assemble(_c,
                          *_c.sys->current_local_solution,
                          *_c.rho_sys->current_local_solution,
                          res.get(),
                          _c.sys->matrix);
        // We multiply with -1 before solving for \f$ x \f$.
        res->scale(-1);
        // This solves for \f$ x\f$ from the system of equations \f$ K x = f \f$.
        Mat
        m   = dynamic_cast<libMesh::PetscMatrix<real_t>*>(_c.sys->matrix)->mat();
        Vec
        b   = dynamic_cast<libMesh::PetscVector<real_t>*>(res.get())->vec(),
        sol = dynamic_cast<libMesh::PetscVector<real_t>*>(_c.sys->solution.get())->vec();
        
        MAST::Solvers::PETScWrapper::LinearSolver
        linear_solver(_c.eq_sys->comm().get());
        linear_solver.init(m, &_c.sys->name());
        linear_solver.solve(sol, b);

        _c.sys->update();

        // compliance is defined using the external work done \f$ c = x^T f \f$
        scalar_t
        vol    = 0.,
        comp   = _c.sys->solution->dot(*res);
        
        //////////////////////////////////////////////////////////////////////
        // evaluate the functions
        //////////////////////////////////////////////////////////////////////
        
        // evaluate the volume for used in the problem setup
        MAST::Optimization::Topology::SIMP::libMeshWrapper::Volume<scalar_t>
        volume;
        
        vol = volume.compute(_c, *_c.rho_sys->current_local_solution,
                             *_e_ops.heaviside);
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
            res.reset(_c.sys->current_local_solution->clone().release());
            res->scale(-1.);
            
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
                                     *_c.sys->current_local_solution,      // solution
                                     *_c.rho_sys->current_local_solution,  // filtered density
                                     *res,                                 // adjoint solution
                                     *_c.ex_init.filter,                   // geometric filter
                                     *_dvs,
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
                              *_c.rho_sys->current_local_solution,
                              *_e_ops.heaviside,
                              *_c.ex_init.filter,
                              *_dvs,
                              grads);
            for (uint_t i=0; i<grads.size(); i++)
                grads[i] /= _volume;
        }
    }
    
    inline void output(const uint_t                iter,
                       const std::vector<real_t>  &dvars,
                       real_t                     &o,
                       std::vector<real_t>        &fvals) {

        // pass all desntity values through the heaviside filter for plotting.
        for (uint_t i=_c.rho_sys->solution->first_local_index();
             i<_c.rho_sys->solution->last_local_index(); i++) {

            real_t v = _c.rho_sys->solution->el(i);
            _c.rho_sys->solution->set(i, _e_ops.heaviside->filter(v));
        }
        _c.rho_sys->solution->close();
        _c.rho_sys->update();
        
        std::ostringstream oss;
        oss << "output_optim.e-s." << std::setfill('0') << std::setw(5) << iter ;
        
        _c.sys->time = iter;
        libMesh::Nemesis_IO writer(*_c.mesh);
        // "1" is the number of time-steps in the file,
        // as opposed to the time-step number.
        writer.write_timestep(oss.str(), *_c.eq_sys, 1, (real_t)iter);
        
        // also, save the design iteration to a text file
        MAST::Optimization::Utility::write_obj_constr_history_to_file(*this,
                                                                      _history,
                                                                      iter,
                                                                      o,
                                                                      fvals);

    }
    
private:
    
    ElemOps<TraitsType>                                 &_e_ops;
    context_t                                           &_c;
    MAST::Optimization::DesignParameterVector<scalar_t> *_dvs;
    real_t                                               _volume;
    real_t                                               _vf;
    std::ofstream                                        _history;
};
} // namespace Example6
} // namespace Structural
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

template <typename ModelType>
void run(libMesh::LibMeshInit& init, MAST::Utility::GetPotWrapper& input) {
    
    using traits_t    = MAST::Examples::Structural::Example6::Traits<real_t, real_t, real_t, ModelType>;
    using elem_ops_t  = MAST::Examples::Structural::Example6::ElemOps<traits_t>;
    using func_eval_t = MAST::Examples::Structural::Example6::FunctionEvaluation<traits_t>;

    typename traits_t::ex_init_t ex_init(init.comm(), input);

    typename traits_t::context_t  c(ex_init);
    elem_ops_t           e_ops(c);
    func_eval_t          f_eval(e_ops, c);
    
    // create an optimizer, attach the function evaluation
    MAST::Optimization::Solvers::GCMMAInterface<func_eval_t> optimizer(init.comm());
    optimizer.max_iters       = input("max_iters","maximum iterations for GCMMA", 50);
    optimizer.max_inner_iters = 6;
    optimizer.constr_penalty  = input("constr_penalty","constraint penalty for GCMMA", 1.e5);
    optimizer.initial_rel_step=1.e-2;
    optimizer.set_function_evaluation(f_eval);
    optimizer.init();
    
    real_t
    beta_base = input("heaviside_beta","Base value of beta parameter in Heaviside filter", 1.);
    
    // continuation approach to increase penalty parameters
    for (uint_t i=0; i<8; i++) {
        
        ex_init.penalty = 1. + 0.5 * i;
        ex_init.beta    = pow(beta_base, i);
     
        e_ops.heaviside->set_parameters(c.ex_init.beta, c.ex_init.eta);
        e_ops.density->set_penalty(c.ex_init.penalty);

        optimizer.optimize();
    }
}

int main(int argc, char** argv) {
    
    libMesh::LibMeshInit init(argc, argv);
    MAST::Utility::GetPotWrapper input(argc, argv);

    std::string
    nm = input("model",
               "model to run for topology optimization: bracket2d/bracket3d/truss2d/truss3d/inplane2d",
               "bracket2d");
    
    if (nm == "bracket2d")
        run<MAST::Mesh::Generation::Bracket2D>(init, input);
    else if (nm == "bracket3d")
        run<MAST::Mesh::Generation::Bracket3D>(init, input);
    else if (nm == "truss2d")
        run<MAST::Mesh::Generation::Truss2D>(init, input);
    else if (nm == "truss3d")
        run<MAST::Mesh::Generation::Truss3D>(init, input);
    else if (nm == "inplane2d")
        run<MAST::Mesh::Generation::Inplane2D>(init, input);
    else if (nm == "panel2d")
        run<MAST::Mesh::Generation::Panel2D>(init, input);
    else if (nm == "panel3d")
        run<MAST::Mesh::Generation::Panel3D>(init, input);
    else
        std::cout << "Invalid model" << std::endl;
    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
