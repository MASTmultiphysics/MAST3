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
#include <mast/fe/fe_var_data.hpp>
#include <mast/fe/scalar_field_wrapper.hpp>
#include <mast/physics/conduction/material_conductance.hpp>
#include <mast/physics/conduction/linear_conduction_kernel.hpp>
#include <mast/physics/conduction/source_kernel.hpp>
#include <mast/physics/conduction/libmesh/mat_null_space.hpp>
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
#include <mast/mesh/generation/heat_sink2d.hpp>
#include <mast/mesh/generation/heat_sink3d.hpp>

// libMesh includes
#include <libmesh/distributed_mesh.h>
#include <libmesh/elem.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/boundary_info.h>
#include <libmesh/nemesis_io.h>
#include <libmesh/petsc_matrix.h>


// BEGIN_TRANSLATE SIMP Heat Sink Topology Optimization with MPI based solvers


namespace MAST {
namespace Examples {
namespace Conduction {
namespace Example2 {

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
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("conduction")),
    rho_sys   (&eq_sys->add_system<libMesh::ExplicitSystem>("density")),
    filter    (nullptr),
    penalty   (0.) {
        
        model->init_analysis_mesh(*this, *mesh);

        // displacement variables for elasticity solution
        sys->add_variable("T", libMesh::FEType(fe_order, fe_family));
        
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
        MAST::Physics::Conduction::libMeshWrapper::NullSpace
        null_sp(*sys, ModelType::dim);
        
        Mat m = dynamic_cast<libMesh::PetscMatrix<real_t>*>(sys->matrix)->mat();
        null_sp.attach_to_matrix(m);
        
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
    libMesh::DistributedMesh                    *mesh;
    libMesh::EquationSystems                    *eq_sys;
    libMesh::NonlinearImplicitSystem            *sys;
    libMesh::ExplicitSystem                     *rho_sys;
    MAST::Mesh::libMeshWrapper::GeometricFilter *filter;
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
    inline bool elem_is_hex() const  {return (elem->type() == libMesh::HEX8 ||
                                              elem->type() == libMesh::HEX20 ||
                                              elem->type() == libMesh::HEX27);}
    
    
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
    using traits_t          = MAST::Examples::Conduction::Example2::Traits<BasisScalarType, NodalScalarType, SolScalarType, ModelType>;
    using model_t           = ModelType;
    using context_t         = Context<traits_t>;
    using ex_init_t         = InitExample<model_t>;
    using scalar_t          = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type, SolScalarType>::type;
    using fe_basis_t        = typename MAST::FEBasis::libMeshWrapper::FEBasis<BasisScalarType, dim>;
    using fe_shape_t        = typename MAST::FEBasis::Evaluation::FEShapeDerivative<BasisScalarType, NodalScalarType, dim, dim, fe_basis_t>;
    using fe_data_t         = typename MAST::FEBasis::libMeshWrapper::FEData<dim, fe_basis_t, fe_shape_t>;
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 1, dim, context_t, fe_shape_t>;
    using density_fe_var_t  = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, 1, dim, context_t, fe_shape_t>;
    using density_field_t   = typename MAST::FEBasis::ScalarFieldWrapper<scalar_t, density_fe_var_t>;
    using density_t         = typename MAST::Optimization::Topology::SIMP::PenalizedDensity<SolScalarType, density_field_t>;
    using conductance_t     = typename MAST::Optimization::Topology::SIMP::PenalizedScalar<SolScalarType, density_t>;
    using source_t          = typename MAST::Base::ScalarConstant<SolScalarType>;
    using area_t            = typename MAST::Base::ScalarConstant<SolScalarType>;
    using prop_t            = typename MAST::Physics::Conduction::IsotropicMaterialConductance<SolScalarType, conductance_t, context_t>;
    using conduction_t      = typename MAST::Physics::Conduction::ConductionKernel<fe_var_t, prop_t, dim, context_t, true, true>;
    using source_load_t     = typename MAST::Physics::Conduction::SourceHeatLoad<fe_var_t, source_t, area_t, dim, context_t>;
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
    density         (nullptr),
    k               (nullptr),
    qv              (nullptr),
    area            (nullptr),
    _fe_data        (nullptr),
    _fe_var         (nullptr),
    _density_fe_var (nullptr),
    _density_field  (nullptr),
    _prop           (nullptr),
    _conduction     (nullptr),
    _source_load    (nullptr) {
        
        _fe_data       = new typename TraitsType::fe_data_t;
        _fe_data->init(c.ex_init.q_order,
                       c.ex_init.q_type,
                       c.ex_init.fe_order,
                       c.ex_init.fe_family);
        _fe_var        = new typename TraitsType::fe_var_t;
        
        _density_fe_var      = new typename TraitsType::density_fe_var_t;
        _density_sens_fe_var = new typename TraitsType::density_fe_var_t;
        _density_field       = new typename TraitsType::density_field_t;
        
        // associate variables with the shape functions
        _fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        
        // tell the FE computations which quantities are needed for computation
        _fe_data->fe_basis().set_compute_dphi_dxi(true);
        
        _fe_data->fe_derivative().set_compute_dphi_dx(true);
        _fe_data->fe_derivative().set_compute_detJxW(true);
                
        _fe_var->set_compute_du_dx(true);
        
        _density_fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _density_sens_fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _density_field->set_fe_object_and_component(*_density_fe_var, 0);
        _density_field->set_derivative_fe_object_and_component(*_density_sens_fe_var, 0);

        // variables for physics
        density  = new typename TraitsType::density_t;
        k        = new typename TraitsType::conductance_t;
        qv       = new typename TraitsType::source_t(1.);
        area     = new typename TraitsType::area_t(1.0);
        _prop    = new typename TraitsType::prop_t;
        
        density->set_penalty(c.ex_init.penalty);
        density->set_density_field(*_density_field);
        k->set_density(*density);
        k->set_scalar(1.0, 1.e-3);
        
        _prop->set_conductance(*k);
        _conduction   = new typename TraitsType::conduction_t;
        _conduction->set_section_property(*_prop);
        _source_load  = new typename TraitsType::source_load_t;
        _source_load->set_source(*qv);

        if (TraitsType::dim < 3)
            _source_load->set_section_area(*area);

        // tell physics kernels about the FE discretization information
        _conduction->set_fe_var_data(*_fe_var);
        _source_load->set_fe_var_data(*_fe_var);
    }
    
    virtual ~ElemOps() {
        
        delete area;
        delete qv;
        delete k;
        delete density;

        delete _conduction;
        delete _prop;
        delete _source_load;

        delete _density_field;
        delete _density_fe_var;
        
        delete _fe_var;
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

        _conduction->compute(c, res, jac);
        _source_load->compute(c, res, jac);
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

        _conduction->derivative(c, f, res, jac);
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
    typename TraitsType::density_t        *density;
    typename TraitsType::conductance_t    *k;
    typename TraitsType::source_t         *qv;
    typename TraitsType::area_t           *area;
    
private:
    
    // variables for quadrature and shape function
    typename TraitsType::fe_data_t         *_fe_data;
    typename TraitsType::fe_var_t          *_fe_var;
    typename TraitsType::density_fe_var_t  *_density_fe_var;
    typename TraitsType::density_fe_var_t  *_density_sens_fe_var;
    typename TraitsType::density_field_t   *_density_field;
    typename TraitsType::prop_t            *_prop;
    typename TraitsType::conduction_t      *_conduction;
    typename TraitsType::source_load_t     *_source_load;
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
                                    "upper limit for the volume fraction", 0.3)) {
        
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
        &cnd_sys = *_c.ex_init.sys,
        &rho_sys = *_c.ex_init.rho_sys;

        const uint_t
        n_dofs          = cnd_sys.n_dofs(),
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
        
        _c.ex_init.filter->template compute_filtered_values
        <scalar_t,
        typename TraitsType::assembled_vector_t,
        typename TraitsType::assembled_vector_t>
        (*_dvs, *rho_base, *_c.rho_sys->solution);
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
        libMesh::SparseMatrix<real_t>
        *pc  = _c.sys->request_matrix("Preconditioner");

        std::pair<unsigned int, real_t>
        solver_params = _c.sys->get_linear_solve_parameters();

        Mat
        m   = dynamic_cast<libMesh::PetscMatrix<real_t>*>(_c.sys->matrix)->mat();
        Vec
        b   = dynamic_cast<libMesh::PetscVector<real_t>*>(res.get())->vec(),
        sol = dynamic_cast<libMesh::PetscVector<real_t>*>(_c.sys->solution.get())->vec();
        
        MAST::Solvers::PETScWrapper::LinearSolver
        linear_solver(_c.eq_sys->comm().get());
        linear_solver.init(m);
        linear_solver.solve(sol, b);

        _c.sys->update();

        scalar_t
        vol        = 0.,
        temp_sum   = _c.sys->solution->sum()/(1.*n_dofs);
        
        //////////////////////////////////////////////////////////////////////
        // evaluate the functions
        //////////////////////////////////////////////////////////////////////
        
        // evaluate the volume for used in the problem setup
        MAST::Optimization::Topology::SIMP::libMeshWrapper::Volume<scalar_t>
        volume;
        
        vol = volume.compute(_c, *_c.rho_sys->current_local_solution);
        std::cout << "volume: " << vol << std::endl;
        
        // evaluate the output based on specified problem type
        obj       = temp_sum;
        fvals[0]  = vol/_volume - _vf; // vol/vol0 - a <=
        std::cout << "Sum_i Temperature: " << temp_sum << std::endl;
        

        //////////////////////////////////////////////////////////////////////
        // evaluate the objective sensitivities, if requested
        //////////////////////////////////////////////////////////////////////
        if (eval_obj_grad) {
            
            MAST::Optimization::Topology::SIMP::libMeshWrapper::AssembleOutputSensitivity
            <scalar_t, ElemOps<TraitsType>, ElemOps<TraitsType>>
            temp_sum_sens;
            
            temp_sum_sens.set_elem_ops(_e_ops, _e_ops);

            // the adjoint solution for sum of temperature is obtained using a RHS vector
            // of unit values scaled by the number of degrees-of-freedom, \f$ N \f$.
            // \f[ K^T \lambda = - \{1\}/N \f]
            //
            (*res) = -1./(1.*n_dofs);
            
            std::unique_ptr<libMesh::NumericVector<real_t>>
            adj(_c.sys->solution->zero_clone().release()),
            adj_localized(_c.sys->current_local_solution->zero_clone().release());
            Vec
            adj_v = dynamic_cast<libMesh::PetscVector<real_t>*>(adj.get())->vec();
            b     = dynamic_cast<libMesh::PetscVector<real_t>*>(res.get())->vec();
            linear_solver.solve(adj_v, b);
            adj->localize(*adj_localized, _c.sys->get_dof_map().get_send_list());
            
            // This solves for the sensitivity of sum of temperature,
            // \f$ T_s=\sum_{i=1}^N T_i \f$, with respect to a parameter \f$ \alpha \f$.
            // \f{eqnarray*}{ \frac{dT_s)}{d\alpha}
            //    & = & \frac{\partial T_s}{\partial \alpha} + \frac{\partial T_s}{\partial T}
            //     \frac{dT}{d\alpha} \\
            //    & = & 0 + \lambda^T \frac{\partial R(x)}{\partial \alpha}
            // \f}
            temp_sum_sens.assemble(_c,
                                   *_c.sys->current_local_solution,      // solution
                                   *_c.rho_sys->current_local_solution,  // filtered density
                                   *adj_localized,                       // adjoint solution
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
} // namespace Example2
} // namespace Conduction
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

template <typename ModelType>
void run(libMesh::LibMeshInit& init, MAST::Utility::GetPotWrapper& input) {
    
    using traits_t    = MAST::Examples::Conduction::Example2::Traits<real_t, real_t, real_t, ModelType>;
    using elem_ops_t  = MAST::Examples::Conduction::Example2::ElemOps<traits_t>;
    using func_eval_t = MAST::Examples::Conduction::Example2::FunctionEvaluation<traits_t>;

    typename traits_t::ex_init_t ex_init(init.comm(), input);

    typename traits_t::context_t  c(ex_init);
    elem_ops_t           e_ops(c);
    func_eval_t          f_eval(e_ops, c);
    
    // create an optimizer, attach the function evaluation
    MAST::Optimization::Solvers::GCMMAInterface<func_eval_t> optimizer(init.comm());
    optimizer.max_inner_iters = 6;
    optimizer.constr_penalty  = 1.e5;
    optimizer.set_function_evaluation(f_eval);
    
    // optimize
    optimizer.optimize();
}

int main(int argc, char** argv) {
    
    libMesh::LibMeshInit init(argc, argv);
    MAST::Utility::GetPotWrapper input(argc, argv);

    std::string
    nm = input("model", "model to run for topology optimization: heat_sink2d/heat_sink3d", "heat_sink2d");
    
    if (nm == "heat_sink2d")
        run<MAST::Mesh::Generation::HeatSink2D>(init, input);
    if (nm == "heat_sink3d")
        run<MAST::Mesh::Generation::HeatSink3D>(init, input);
    else
        std::cout << "Invalid model" << std::endl;
    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
