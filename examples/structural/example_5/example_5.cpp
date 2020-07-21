
// MAST includes
#include <mast/base/exceptions.hpp>
#include <mast/util/perf_log.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/libmesh/fe_side_data.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/linear_strain_energy.hpp>
#include <mast/physics/elasticity/pressure_load.hpp>
#include <mast/base/assembly/libmesh/residual_and_jacobian.hpp>
#include <mast/base/assembly/libmesh/residual_sensitivity.hpp>
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
    q_order   (libMesh::FOURTH),
    fe_order  (libMesh::SECOND),
    fe_family (libMesh::LAGRANGE),
    mesh      (new libMesh::ReplicatedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    rho_sys   (&eq_sys->add_system<libMesh::ExplicitSystem>("density")),
    p_side_id (-1) {
        
        model->init_analysis_mesh(*this, *mesh);

        // displacement variables for elasticity solution
        sys->add_variable("u_x", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("u_y", libMesh::FEType(fe_order, fe_family));
        
        // density field
        rho_sys->add_variable("rho", libMesh::FEType(fe_order, fe_family));
        
        model->init_analysis_dirichlet_conditions(*this);
        
        sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({3}, {0, 1}, libMesh::ZeroFunction<real_t>()));
        
        eq_sys->init();
        
        mesh->print_info(std::cout);
        eq_sys->print_info(std::cout);
    }
    
    virtual ~InitExample() {
        
        delete eq_sys;
        delete mesh;
        delete model;
    }
    
    libMesh::Parallel::Communicator  &comm;
    MAST::Utility::GetPotWrapper     &input;
    ModelType                        *model;
    libMesh::QuadratureType           q_type;
    libMesh::Order                    q_order;
    libMesh::Order                    fe_order;
    libMesh::FEFamily                 fe_family;
    libMesh::ReplicatedMesh          *mesh;
    libMesh::EquationSystems         *eq_sys;
    libMesh::NonlinearImplicitSystem *sys;
    libMesh::ExplicitSystem          *rho_sys;
    uint_t                            p_side_id;
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
    elem      (nullptr),
    qp        (-1)
    { }
    
    virtual ~Context() { }
    
    // assembly methods
    uint_t  elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    real_t  qp_location(uint_t i) const {;}
    inline bool elem_is_quad() const {return (elem->type() == libMesh::QUAD4 ||
                                              elem->type() == libMesh::QUAD8 ||
                                              elem->type() == libMesh::QUAD9);}
    inline bool if_compute_pressure_load_on_side(const uint_t s)
    { return ex_init.mesh->boundary_info->has_boundary_id(elem, s, ex_init.p_side_id);}
    
    
    InitExample<model_t>            &ex_init;
    libMesh::ReplicatedMesh          *mesh;
    libMesh::EquationSystems         *eq_sys;
    libMesh::NonlinearImplicitSystem *sys;
    const libMesh::Elem              *elem;
    uint_t                            qp;
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
    using modulus_t         = typename MAST::Base::ScalarConstant<SolScalarType>;
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
    E             (nullptr),
    nu            (nullptr),
    press         (nullptr),
    area          (nullptr),
    _fe_data      (nullptr),
    _fe_side_data (nullptr),
    _fe_var       (nullptr),
    _fe_side_var  (nullptr),
    _prop         (nullptr),
    _energy       (nullptr),
    _p_load       (nullptr) {
        
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
        E        = new typename TraitsType::modulus_t(72.e9);
        nu       = new typename TraitsType::nu_t(0.33);
        press    = c.ex_init.model->template build_pressure_load<scalar_t, typename TraitsType::ex_init_t>(c.ex_init).release();
        area     = new typename TraitsType::area_t(1.0);
        _prop    = new typename TraitsType::prop_t;
        
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
        
        delete _p_load;
        delete area;
        delete press;
        delete _energy;
        delete _prop;
        delete nu;
        delete E;
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
        _energy->compute(c, res, jac);
        
        for (uint_t s=0; s<c.elem->n_sides(); s++)
            if (c.if_compute_pressure_load_on_side(s)) {
                
                _fe_side_data->reinit_for_side(c, s);
                _fe_side_var->init(c, v);
                _p_load->compute(c, res, jac);
            }
    }
    
    
    template <typename ContextType, typename AccessorType, typename ScalarFieldType>
    inline void derivative(ContextType                       &c,
                           const ScalarFieldType             &f,
                           const AccessorType                &v,
                           typename TraitsType::element_vector_t &res,
                           typename TraitsType::element_matrix_t *jac) {
        
        _fe_data->reinit(c);
        _fe_var->init(c, v);
        _energy->derivative(c, f, res, jac);
        
        for (uint_t s=0; s<c.elem->n_sides(); s++)
            if (c.if_compute_pressure_load_on_side(s)) {
                
                _fe_side_data->reinit_for_side(c, s);
                _fe_side_var->init(c, v);
                _p_load->derivative(c, f, res, jac);
            }
    }
    
    // parameters
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
    _e_ops  (e_ops),
    _c      (c) {
        
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
            
            unsigned int
            iter = _c.ex_init.input("restart_optimization_iter",
                                    "restart iteration number from file", 0);
            MAST::Optimization::Utility::initialize_dv_from_output_file(*this,
                                                                        nm,
                                                                        iter,
                                                                        x);
        }
        else {
            
            for (unsigned int i=0; i<_dvs.size(); i++)
                x[i] = (*_dvs[i].second)();
        }
    }
    
    
    virtual void evaluate(const std::vector<real_t>& x,
                          real_t& obj,
                          bool eval_obj_grad,
                          std::vector<real_t>& obj_grad,
                          std::vector<real_t>& fvals,
                          std::vector<bool>& eval_grads,
                          std::vector<real_t>& grads) {
        
    }
    
    
    inline void output(const uint_t                iter,
                       const std::vector<real_t>  &dvars,
                       real_t                     &o,
                       std::vector<real_t>        &fvals) {
        
    }
    
private:
    
    ElemOps<TraitsType>   &_e_ops;
    context_t             &_c;
    std::vector<std::pair<uint_t, MAST::Optimization::DesignParameter<scalar_t>*>> _dvs;
};


template <typename TraitsType>
inline void
compute_residual(typename TraitsType::context_t                 &c,
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
compute_residual_sensitivity(typename TraitsType::context_t                 &c,
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
compute_sol(typename TraitsType::context_t           &c,
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
    
    sol = TraitsType::assembled_vector_t::Zero(c.ex_init.sys->n_dofs());
    res = TraitsType::assembled_vector_t::Zero(c.ex_init.sys->n_dofs());
    MAST::Numerics::libMeshWrapper::init_sparse_matrix(c.ex_init.sys->get_dof_map(),
                                                       jac);
    
    assembly.assemble(c, sol, &res, &jac);
    
    sol = Eigen::SparseLU<typename TraitsType::assembled_matrix_t>(jac).solve(-res);
}


template <typename TraitsType, typename ScalarFieldType>
inline void
compute_sol_sensitivity(typename TraitsType::context_t                 &c,
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
    
    dsol = TraitsType::assembled_vector_t::Zero(c.ex_init.sys->n_dofs());
    dres = TraitsType::assembled_vector_t::Zero(c.ex_init.sys->n_dofs());
    MAST::Numerics::libMeshWrapper::init_sparse_matrix(c.ex_init.sys->get_dof_map(), jac);
    
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

    using traits_complex_t   = MAST::Examples::Structural::Example5::Traits<real_t, real_t, complex_t, model_t>;
    using elem_ops_complex_t = MAST::Examples::Structural::Example5::ElemOps<traits_complex_t>;
    using func_eval_complex_t= MAST::Examples::Structural::Example5::FunctionEvaluation<traits_complex_t>;

    traits_t::ex_init_t ex_init(init.comm(), input);

    traits_t::context_t  c(ex_init);
    elem_ops_t           e_ops(c);
    func_eval_t          f_eval(e_ops, c);
    
    traits_complex_t::context_t  c_cmplx(ex_init);
    elem_ops_complex_t           e_ops_c(c_cmplx);
    func_eval_complex_t          f_eval_c(e_ops_c, c_cmplx);

    
    // create an optimizer, attach the function evaluation
    MAST::Optimization::Solvers::GCMMAInterface<func_eval_t> optimizer;
    optimizer.set_function_evaluation(f_eval);
    
    // optimize
    optimizer.optimize();
    
    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
