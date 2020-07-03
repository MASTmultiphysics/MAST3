
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

// BEGIN_TRANSLATE Extension of bar

class Context {
    
public:
    
    Context(libMesh::Parallel::Communicator& comm):
    mesh      (new libMesh::ReplicatedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    elem      (nullptr),
    qp        (-1),
    p_side_id (1) {

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
    using fe_var_t          = typename MAST::FEBasis::FEVarData<BasisScalarType, NodalScalarType, SolScalarType, Dim, Dim, Context, fe_shape_t>;
    using modulus_t         = typename MAST::Base::ScalarConstant<SolScalarType>;
    using nu_t              = typename MAST::Base::ScalarConstant<SolScalarType>;
    using press_t           = typename MAST::Base::ScalarConstant<SolScalarType>;
    using area_t            = typename MAST::Base::ScalarConstant<SolScalarType>;
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, Dim, modulus_t, nu_t, Context>;
    using energy_t          = typename MAST::Physics::Elasticity::LinearContinuum::StrainEnergy<fe_var_t, prop_t, Dim, Context>;
    using press_load_t      = typename MAST::Physics::Elasticity::SurfacePressureLoad<fe_var_t, press_t, area_t, Dim, Context>;
};



template <typename Traits>
class ElemOps {
  
public:
    
    using scalar_t = typename Traits::scalar_t;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    

    ElemOps(libMesh::Order          q_order,
            libMesh::QuadratureType q_type,
            libMesh::Order          fe_order,
            libMesh::FEFamily       fe_family):
    _fe_data      (nullptr),
    _fe_side_data (nullptr),
    _fe_var       (nullptr),
    _fe_side_var  (nullptr),
    _E            (nullptr),
    _nu           (nullptr),
    _prop         (nullptr),
    _energy       (nullptr),
    _press        (nullptr),
    _area         (nullptr),
    _p_load       (nullptr) {
        
        _fe_data       = new typename Traits::fe_data_t;
        _fe_data->init(q_order, q_type, fe_order, fe_family);
        _fe_side_data  = new typename Traits::fe_side_data_t;
        _fe_side_data->init(q_order, q_type, fe_order, fe_family);
        _fe_var        = new typename Traits::fe_var_t;
        _fe_side_var   = new typename Traits::fe_var_t;

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
        _E        = new typename Traits::modulus_t(72.e9);
        _nu       = new typename Traits::nu_t(0.33);
        _press    = new typename Traits::press_t(1.e2);
        _area     = new typename Traits::area_t(1.0);
        _prop     = new typename Traits::prop_t;
        _prop->set_modulus_and_nu(*_E, *_nu);
        _energy   = new typename Traits::energy_t;
        _energy->set_section_property(*_prop);
        _p_load   = new typename Traits::press_load_t;
        _p_load->set_section_area(*_area);
        _p_load->set_pressure(*_press);
        
        // tell physics kernels about the FE discretization information
        _energy->set_fe_var_data(*_fe_var);
        _p_load->set_fe_var_data(*_fe_side_var);
    }
    
    virtual ~ElemOps() {
        
        delete _p_load;
        delete _area;
        delete _press;
        delete _energy;
        delete _prop;
        delete _nu;
        delete _E;
        delete _fe_var;
        delete _fe_side_var;
        delete _fe_side_data;
        delete _fe_data;
    }
    

    template <typename ContextType, typename AccessorType>
    inline void compute(ContextType& c,
                        const AccessorType& v,
                        vector_t& res,
                        matrix_t* jac) {
        
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

    
    template <typename ScalarFieldType>
    inline void derivative(Context& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac) {
        
        _energy->derivative(c, f, res, jac);
    }

private:

    // variables for quadrature and shape function
    typename Traits::fe_data_t         *_fe_data;
    typename Traits::fe_side_data_t    *_fe_side_data;
    typename Traits::fe_var_t          *_fe_var;
    typename Traits::fe_var_t          *_fe_side_var;

    // variables for physics
    typename Traits::modulus_t    *_E;
    typename Traits::nu_t         *_nu;
    typename Traits::prop_t       *_prop;
    typename Traits::energy_t     *_energy;
    typename Traits::press_t      *_press;
    typename Traits::area_t       *_area;
    typename Traits::press_load_t *_p_load;
};


int main(int argc, const char** argv) {

    libMesh::LibMeshInit init(argc, argv);
    
    libMesh::QuadratureType q_type    = libMesh::QGAUSS;
    libMesh::Order          q_order   = libMesh::FOURTH;
    libMesh::Order          fe_order  = libMesh::SECOND;
    libMesh::FEFamily       fe_family = libMesh::LAGRANGE;

    Context c(init.comm());

    libMesh::MeshTools::Generation::build_square(*c.mesh,
                                                 5, 5,
                                                 0.0, 10.0,
                                                 0.0, 10.0,
                                                 libMesh::QUAD9);

    c.sys->add_variable("u_x", libMesh::FEType(fe_order, fe_family));
    c.sys->add_variable("u_y", libMesh::FEType(fe_order, fe_family));

    c.sys->get_dof_map().add_dirichlet_boundary
    (libMesh::DirichletBoundary({3}, {0, 1}, libMesh::ZeroFunction<real_t>()));
    
    c.eq_sys->init();

    c.mesh->print_info(std::cout);
    c.eq_sys->print_info(std::cout);
    
    using basis_scalar_t = real_t;
    using nodal_scalar_t = real_t;
    using sol_scalar_t   = real_t;
    using res_vec_t      = Eigen::Matrix<sol_scalar_t, Eigen::Dynamic, 1>;
    using jac_mat_t      = Eigen::SparseMatrix<sol_scalar_t>;
    using elem_ops_t     = ElemOps<Traits<basis_scalar_t, nodal_scalar_t, sol_scalar_t, 2>>;
    
    elem_ops_t e_ops(q_order, q_type, fe_order, fe_family);

    MAST::Base::Assembly::libMeshWrapper::ResidualAndJacobian<real_t, elem_ops_t>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    res_vec_t sol, res;
    jac_mat_t jac;
    
    sol = res_vec_t::Zero(c.sys->n_dofs());
    res = res_vec_t::Zero(c.sys->n_dofs());
    MAST::Numerics::libMeshWrapper::init_sparse_matrix(c.sys->get_dof_map(), jac);
    
    assembly.assemble(c, sol, &res, &jac);
    
    sol = Eigen::SparseLU<jac_mat_t>(jac).solve(-res);
    
    
    for (uint_t i=0; i<sol.size(); i++)
        c.sys->solution->set(i, sol(i));

    libMesh::ExodusII_IO(*c.mesh).write_equation_systems("solution.exo", *c.eq_sys);
    
    // END_TRANSLATE
    return 0;
}
