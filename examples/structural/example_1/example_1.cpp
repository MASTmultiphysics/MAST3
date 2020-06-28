
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

// libMesh includes
#include <libmesh/replicated_mesh.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/equation_systems.h>
#include <libmesh/boundary_info.h>

// BEGIN_TRANSLATE Extension of bar

class Context {
    
public:
    
    Context(libMesh::Parallel::Communicator& comm):
    mesh    (new libMesh::ReplicatedMesh(comm)),
    eq_sys  (new libMesh::EquationSystems(*mesh)),
    sys     (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    elem    (nullptr),
    qp      (-1) {

    }

    virtual ~Context() {
        
        delete eq_sys;
        delete mesh;
    }
    
    libMesh::ReplicatedMesh          *mesh;
    libMesh::EquationSystems         *eq_sys;
    libMesh::NonlinearImplicitSystem *sys;
    const libMesh::Elem              *elem;
    uint_t                            qp;
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
    using prop_t            = typename MAST::Physics::Elasticity::IsotropicMaterialStiffness<SolScalarType, Dim, modulus_t, nu_t, Context>;
    using energy_t          = typename MAST::Physics::Elasticity::LinearContinuumStrainEnergy<fe_var_t, prop_t, Dim, Context>;
};



template <typename Traits>
class ElemOps {
  
public:
    
    using scalar_t = typename Traits::scalar_t;
    using vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    

    ElemOps(Context& c):
    _fe_data      (nullptr),
    _fe_side_data (nullptr),
    _fe_var       (nullptr),
    _fe_var_side  (nullptr),
    _E            (nullptr),
    _nu           (nullptr),
    _prop         (nullptr),
    _energy       (nullptr),
    _press        (nullptr),
    _c            (&c) {
        
        _fe_data       = new typename Traits::fe_data_t;
        _fe_data->init(libMesh::SECOND,
                       libMesh::QGAUSS,
                       libMesh::FIRST,
                       libMesh::LAGRANGE);
        _fe_side_data  = new typename Traits::fe_side_data_t;
        _fe_side_data->init(libMesh::SECOND,
                            libMesh::QGAUSS,
                            libMesh::FIRST,
                            libMesh::LAGRANGE);
        _fe_var        = new typename Traits::fe_var_t;
        _fe_var_side   = new typename Traits::fe_var_t;

        // variables for physics
        _E        = new typename Traits::modulus_t(72.e9);
        _nu       = new typename Traits::nu_t(0.33);
        _press    = new typename Traits::nu_t(1.e2);
        _prop     = new typename Traits::prop_t;
        _prop->set_modulus_and_nu(*_E, *_nu);
        _energy   = new typename Traits::energy_t;
    }
    
    virtual ~ElemOps() {
        
        delete _press;
        delete _energy;
        delete _prop;
        delete _nu;
        delete _E;
        delete _fe_var;
        delete _fe_var_side;
        delete _fe_side_data;
        delete _fe_data;
        delete _c;
    }
    
    template <typename AccessorType>
    inline void init(Context& c,
                     const AccessorType& v) {
        
    }
    
    inline void clear() {
        
    }
    
    inline void compute(Context& c,
                        vector_t& res,
                        matrix_t* jac) {
        
    }

    inline void derivative(Context& c,
                           vector_t& res,
                           matrix_t* jac) {
        
    }

private:

    // variables for quadrature and shape function
    typename Traits::fe_data_t         *_fe_data;
    typename Traits::fe_side_data_t    *_fe_side_data;
    typename Traits::fe_var_t          *_fe_var;
    typename Traits::fe_var_t          *_fe_var_side;

    // variables for physics
    typename Traits::modulus_t   *_E;
    typename Traits::nu_t        *_nu;
    typename Traits::prop_t      *_prop;
    typename Traits::energy_t    *_energy;
    typename Traits::press_t     *_press;
    
    Context                      *_c;
};


int main(int argc, const char** argv) {

    libMesh::LibMeshInit init(argc, argv);
    
    libMesh::QuadratureType q_type    = libMesh::QGAUSS;
    libMesh::Order          q_order   = libMesh::SECOND;
    libMesh::Order          fe_order  = libMesh::FIRST;
    libMesh::FEFamily       fe_family = libMesh::LAGRANGE;

    Context c(init.comm());

    c.sys->add_variable("u_x", libMesh::FEType(fe_order, fe_family));
    c.sys->add_variable("u_y", libMesh::FEType(fe_order, fe_family));

    libMesh::MeshTools::Generation::build_line(*c.mesh, 5, 0.0, 10.0);
    c.mesh->print_info();
    c.mesh->boundary_info->print_info();

    using basis_scalar_t = real_t;
    using nodal_scalar_t = real_t;
    using sol_scalar_t   = real_t;
    using res_vec_t      = Eigen::Matrix<sol_scalar_t, Eigen::Dynamic, 1>;
    using jac_mat_t      = Eigen::Matrix<sol_scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using elem_ops_t = ElemOps<Traits<basis_scalar_t, nodal_scalar_t, sol_scalar_t, 2>>;
    
    elem_ops_t e_ops(c);

    MAST::Base::Assembly::libMeshWrapper::ResidualAndJacobian<real_t, elem_ops_t>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    res_vec_t sol, res;
    jac_mat_t jac;

    assembly.assemble(c, sol, &res, &jac);
    
    // END_TRANSLATE
    return 0;
}
