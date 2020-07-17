
// MAST includes
#include <mast/base/exceptions.hpp>
#include <mast/util/perf_log.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/base/scalar_constant.hpp>
#include <mast/physics/elasticity/mindlin_plate_strain_energy.hpp>
#include <mast/physics/elasticity/plate_bending_section_property.hpp>
#include <mast/physics/elasticity/shell_face_pressure_load.hpp>
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

// BEGIN_TRANSLATE Bending with Mindlin Plate

namespace MAST {
namespace Examples {
namespace Structural {
namespace Example2 {

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
    qp        (-1) {



        libMesh::MeshTools::Generation::build_square(*mesh,
                                                     2, 2,
                                                     0.0, 10.0,
                                                     0.0, 10.0,
                                                     libMesh::QUAD9);

        sys->add_variable("w", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("t_x", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("t_y", libMesh::FEType(fe_order, fe_family));

        sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({3}, {0, 1, 2}, libMesh::ZeroFunction<real_t>()));
        
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

    libMesh::QuadratureType           q_type;
    libMesh::Order                    q_order;
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
    using press_t           = typename MAST::Base::ScalarConstant<SolScalarType>;
    using thickness_t       = typename MAST::Base::ScalarConstant<SolScalarType>;
    using material_t        = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, 2, modulus_t, nu_t, Context>;
    using section_t         = typename MAST::Physics::Elasticity::PlateBendingSectionProperty<SolScalarType, material_t, thickness_t, Context>;
    using energy_t          = typename MAST::Physics::Elasticity::MindlinPlate::StrainEnergy<fe_var_t, section_t, Context>;
    using press_load_t      = typename MAST::Physics::Elasticity::ShellFacePressureLoad<fe_var_t, press_t, Context>;
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
    E             (nullptr),
    nu            (nullptr),
    press         (nullptr),
    thickness     (nullptr),
    _fe_data      (nullptr),
    _fe_var       (nullptr),
    _material     (nullptr),
    _section      (nullptr),
    _energy       (nullptr),
    _p_load       (nullptr) {
        
        _fe_data       = new typename TraitsType::fe_data_t;
        _fe_data->init(q_order, q_type, fe_order, fe_family);
        _fe_var        = new typename TraitsType::fe_var_t;

        // associate variables with the shape functions
        _fe_var->set_fe_shape_data(_fe_data->fe_derivative());

        // tell the FE computations which quantities are needed for computation
        _fe_data->fe_basis().set_compute_dphi_dxi(true);
        
        _fe_data->fe_derivative().set_compute_dphi_dx(true);
        _fe_data->fe_derivative().set_compute_detJxW(true);
        
        _fe_var->set_compute_du_dx(true);
        
        // variables for physics
        E         = new typename TraitsType::modulus_t(72.e9);
        nu        = new typename TraitsType::nu_t(0.33);
        press     = new typename TraitsType::press_t(1.e2);
        thickness = new typename TraitsType::thickness_t(3.0e-4);
        _material = new typename TraitsType::material_t;
        _section  = new typename TraitsType::section_t;
        
        _material->set_modulus_and_nu(*E, *nu);
        _section->set_material_and_thickness(*_material, *thickness);
        _energy   = new typename TraitsType::energy_t;
        _energy->set_section_property(*_section);
        _p_load   = new typename TraitsType::press_load_t;
        _p_load->set_pressure(*press);
        
        // tell physics kernels about the FE discretization information
        _energy->set_fe_var_data(*_fe_var);
        _p_load->set_fe_var_data(*_fe_var);
    }
    
    virtual ~ElemOps() {
        
        delete _p_load;
        delete thickness;
        delete press;
        delete _energy;
        delete _section;
        delete _material;
        delete nu;
        delete E;
        delete _fe_var;
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
        _p_load->compute(c, res, jac);
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
        _p_load->derivative(c, f, res, jac);
    }

    // parameters
    typename TraitsType::modulus_t    *E;
    typename TraitsType::nu_t         *nu;
    typename TraitsType::press_t      *press;
    typename TraitsType::thickness_t  *thickness;
    
private:

    // variables for quadrature and shape function
    typename TraitsType::fe_data_t         *_fe_data;
    typename TraitsType::fe_var_t          *_fe_var;
    typename TraitsType::material_t        *_material;
    typename TraitsType::section_t         *_section;
    typename TraitsType::thickness_t       *_thickness;
    typename TraitsType::energy_t          *_energy;
    typename TraitsType::press_load_t      *_p_load;
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


} // namespace Example2
} // namespace Structural
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

int main(int argc, const char** argv) {

    libMesh::LibMeshInit init(argc, argv);
    
    using traits_t           = MAST::Examples::Structural::Example2::Traits<real_t, real_t,    real_t>;
    using traits_complex_t   = MAST::Examples::Structural::Example2::Traits<real_t, real_t, complex_t>;


    MAST::Examples::Structural::Example2::Context c(init.comm());
    MAST::Examples::Structural::Example2::ElemOps<traits_t>
    e_ops(c.q_order, c.q_type, c.fe_order, c.fe_family);
    MAST::Examples::Structural::Example2::ElemOps<traits_complex_t>
    e_ops_c(c.q_order, c.q_type, c.fe_order, c.fe_family);

    typename traits_t::assembled_vector_t
    sol,
    dsol;

    typename traits_complex_t::assembled_vector_t
    sol_c;

    // compute the solution
    MAST::Examples::Structural::Example2::compute_sol<traits_t>(c, e_ops, sol);
    
    // write solution as first time-step
    libMesh::ExodusII_IO writer(*c.mesh);
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, sol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 1, 1.);
    }

    // compute the solution sensitivity wrt E
    (*e_ops_c.E)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.E)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.E, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 2, 2.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    // compute the solution sensitivity wrt nu
    (*e_ops_c.nu)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.nu)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.nu, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 3, 3.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    // compute the solution sensitivity wrt p
    (*e_ops_c.press)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.press)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.press, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 4, 4.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    // compute the solution sensitivity wrt section area
    (*e_ops_c.thickness)() += complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
    (*e_ops_c.thickness)() -= complex_t(0., ComplexStepDelta);
    MAST::Examples::Structural::Example2::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.thickness, sol, dsol);
    
    // write solution as first time-step
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 5, 5.);
    }
    dsol -= sol_c.imag()/ComplexStepDelta;
    std::cout << dsol.norm() << std::endl;

    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
