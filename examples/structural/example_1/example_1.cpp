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
#include <mast/base/material_point/material_point_storage.hpp>
#include <mast/base/material_point/libmesh/indexing.hpp>
#include <mast/util/perf_log.hpp>
#include <mast/fe/eval/fe_basis_derivatives.hpp>
#include <mast/fe/libmesh/fe_data.hpp>
#include <mast/fe/libmesh/fe_side_data.hpp>
#include <mast/fe/fe_var_data.hpp>
#include <mast/physics/elasticity/isotropic_stiffness.hpp>
#include <mast/physics/elasticity/linear_strain_energy.hpp>
#include <mast/physics/elasticity/pressure_load.hpp>
#include <mast/physics/elasticity/continuum_stress.hpp>
#include <mast/physics/elasticity/von_mises_stress.hpp>
#include <mast/base/assembly/libmesh/residual_and_jacobian.hpp>
#include <mast/base/assembly/libmesh/residual_sensitivity.hpp>
#include <mast/base/assembly/libmesh/stress_assembly.hpp>
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

// BEGIN_TRANSLATE Two-dimensional Linear Continuum with Pressure Load

namespace MAST {
namespace Examples {
namespace Structural {
namespace Example1 {

class Context {
    
public:
    
    using mp_indexing_t = MAST::Base::MaterialPoint::libMeshWrapper::Indexing;

    Context(libMesh::Parallel::Communicator& comm):
    q_type    (libMesh::QGAUSS),
    q_order   (libMesh::FOURTH),
    fe_order  (libMesh::SECOND),
    fe_family (libMesh::LAGRANGE),
    mesh      (new libMesh::ReplicatedMesh(comm)),
    eq_sys    (new libMesh::EquationSystems(*mesh)),
    sys       (&eq_sys->add_system<libMesh::NonlinearImplicitSystem>("structural")),
    stress_sys(&eq_sys->add_system<libMesh::ExplicitSystem>("stress")),
    elem      (nullptr),
    qp        (-1),
    p_side_id (1),
    index     (nullptr) {


        // initialize the mesh on a two-dimensional domanin of size
        // \f$ [0,10]\times[0,10]\f$ with a \f$ 10\times 10\f$ mesh of nine-noded
        // quadrilateral elements.
        libMesh::MeshTools::Generation::build_square(*mesh,
                                                     10, 10,
                                                     0.0, 10.0,
                                                     0.0, 10.0,
                                                     libMesh::QUAD9);

        // We add two variables to the system, one for each component of the displacement.
        // The variables are discretized using C0 continuous Lagrange shape functions
        // of second order, as defined in the constructor above.
        sys->add_variable("u_x", libMesh::FEType(fe_order, fe_family));
        sys->add_variable("u_y", libMesh::FEType(fe_order, fe_family));

        // we add constant variables for plotting stress. One variable for each component
        // of stress \f$ \{ \sigma_xx, \sigma_yy, \sigma_xy \} \f$ and one for von Mises
        // stress.
        stress_sys->add_variable("s_xx", libMesh::FEType(libMesh::CONSTANT, libMesh::MONOMIAL));
        stress_sys->add_variable("s_yy", libMesh::FEType(libMesh::CONSTANT, libMesh::MONOMIAL));
        stress_sys->add_variable("s_xy", libMesh::FEType(libMesh::CONSTANT, libMesh::MONOMIAL));
        stress_sys->add_variable("s_vm", libMesh::FEType(libMesh::CONSTANT, libMesh::MONOMIAL));

        // this constrains both displacement variables on boundary 3 (left edge) to zero.
        sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({3}, {0, 1}, libMesh::ZeroFunction<real_t>()));
        
        // initializing the equation system sets up the discretization in libMesh.
        eq_sys->init();

        mesh->print_info(std::cout);
        eq_sys->print_info(std::cout);
        
        index = new mp_indexing_t;
    }

    virtual ~Context() {
        
        delete eq_sys;
        delete mesh;
        delete index;
    }
    
    uint_t elem_dim() const {return elem->dim();}
    uint_t  n_nodes() const {return elem->n_nodes();}
    real_t  nodal_coord(uint_t nd, uint_t c) const {return elem->point(nd)(c);}
    inline bool elem_is_quad() const {return (elem->type() == libMesh::QUAD4 ||
                                              elem->type() == libMesh::QUAD8 ||
                                              elem->type() == libMesh::QUAD9);}
    // since we use a mesh two-dimensional quad9 elements, the number of quadrature
    // points per element can be a-priori determined.
    inline uint_t n_qpoints_per_elem() const {
        if (q_order == libMesh::SECOND) return 4;
        else if (q_order == libMesh::FOURTH) return 9;
        else Error(false, "Quadrature order not implemented");
    }
    inline bool if_compute_pressure_load_on_side(const uint_t s)
    { return mesh->boundary_info->has_boundary_id(elem, s, p_side_id);}
    inline void init_index() { index->init(*mesh, this->n_qpoints_per_elem());}

    libMesh::QuadratureType           q_type;
    libMesh::Order                    q_order;
    libMesh::Order                    fe_order;
    libMesh::FEFamily                 fe_family;
    libMesh::ReplicatedMesh          *mesh;
    libMesh::EquationSystems         *eq_sys;
    libMesh::NonlinearImplicitSystem *sys;
    libMesh::ExplicitSystem          *stress_sys;
    const libMesh::Elem              *elem;
    uint_t                            qp;
    uint_t                            p_side_id;
    mp_indexing_t                    *index;
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
    using stress_t          = typename MAST::Physics::Elasticity::LinearContinuum::Stress<fe_var_t, prop_t, Dim, Context>;
    using press_load_t      = typename MAST::Physics::Elasticity::SurfacePressureLoad<fe_var_t, press_t, area_t, Dim, Context>;
    using element_vector_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using element_matrix_t  = Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;
    using assembled_vector_t = Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using assembled_matrix_t = Eigen::SparseMatrix<scalar_t>;
    using mp_indexing_t      = MAST::Base::MaterialPoint::libMeshWrapper::Indexing;
    using mp_storage_t       = MAST::Base::MaterialPoint::Storage<scalar_t, stress_t::n_strain>;
    using mp_vm_storage_t    = MAST::Base::MaterialPoint::Storage<scalar_t, 1>;
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
    area          (nullptr),
    _fe_data      (nullptr),
    _fe_side_data (nullptr),
    _fe_var       (nullptr),
    _sens_fe_var  (nullptr),
    _fe_side_var  (nullptr),
    _prop         (nullptr),
    _energy       (nullptr),
    _p_load       (nullptr),
    _stress       (nullptr) {
        
        _fe_data       = new typename TraitsType::fe_data_t;
        _fe_data->init(q_order, q_type, fe_order, fe_family);
        _fe_side_data  = new typename TraitsType::fe_side_data_t;
        _fe_side_data->init(q_order, q_type, fe_order, fe_family);
        _fe_var        = new typename TraitsType::fe_var_t;
        _sens_fe_var   = new typename TraitsType::fe_var_t;
        _fe_side_var   = new typename TraitsType::fe_var_t;

        // associate variables with the shape functions
        _fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _sens_fe_var->set_fe_shape_data(_fe_data->fe_derivative());
        _fe_side_var->set_fe_shape_data(_fe_side_data->fe_derivative());

        // tell the FE computations which quantities are needed for computation
        _fe_data->fe_basis().set_compute_dphi_dxi(true);
        
        _fe_data->fe_derivative().set_compute_dphi_dx(true);
        _fe_data->fe_derivative().set_compute_detJxW(true);
        
        _fe_side_data->fe_basis().set_compute_dphi_dxi(true);
        _fe_side_data->fe_derivative().set_compute_normal(true);
        _fe_side_data->fe_derivative().set_compute_detJxW(true);

        _fe_var->set_compute_du_dx(true);
        _sens_fe_var->set_compute_du_dx(true);
        
        // variables for physics
        E        = new typename TraitsType::modulus_t(72.e9);
        nu       = new typename TraitsType::nu_t(0.33);
        press    = new typename TraitsType::press_t(1.e2);
        area     = new typename TraitsType::area_t(1.0);
        _prop    = new typename TraitsType::prop_t;
        
        _prop->set_modulus_and_nu(*E, *nu);
        _energy   = new typename TraitsType::energy_t;
        _energy->set_section_property(*_prop);
        _p_load   = new typename TraitsType::press_load_t;
        _p_load->set_section_area(*area);
        _p_load->set_pressure(*press);
        _stress   = new typename TraitsType::stress_t;
        _stress->set_section_property(*_prop);

        // tell physics kernels about the FE discretization information
        _energy->set_fe_var_data(*_fe_var);
        _p_load->set_fe_var_data(*_fe_side_var);
        _stress->set_fe_var_data(*_fe_var);
        _stress->set_fe_var_sensitivity_data(*_sens_fe_var);
    }
    
    virtual ~ElemOps() {
        
        delete _p_load;
        delete _stress;
        delete area;
        delete press;
        delete _energy;
        delete _prop;
        delete nu;
        delete E;
        delete _fe_var;
        delete _sens_fe_var;
        delete _fe_side_var;
        delete _fe_side_data;
        delete _fe_data;
    }
    
    // this method computes the residual and Jacobian for an element
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

    
    // this method computes the derivative of residual and Jacobian for an element
    // with respect to the parameter \p f. This uses the hand-coded derivatives
    // in the compute kernels that through the \p derivative() methods.
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

    
    // this method computes the stress at the material (quadrature) points in an element.
    // The storage object \p storage stores the stress vector in Voigt representation.
    // and \p index provides the mapping from element and quadrature point to the
    // local/global index of the material point.
    template <typename ContextType,
              typename AccessorType,
              typename IndexingType,
              typename StorageType>
    inline void compute(ContextType             &c,
                        const AccessorType      &v,
                        const IndexingType      &index,
                        StorageType             &storage) {
        
        _fe_data->reinit(c);
        _fe_var->init(c, v);
        
        uint_t
        id = 0;
        
        for (uint_t i=0; i<_fe_var->n_q_points(); i++) {
            
            c.qp = i;
            
            id = index.local_id_for_point_on_elem(c.elem, i);
            typename StorageType::view_t
            stress_qp = storage.data(id);
            
            _stress->compute(c, stress_qp);
        }
    }

    // this method computes the derivative of stress with respect to parameter \p f.
    template <typename ContextType,
              typename AccessorType,
              typename ScalarFieldType,
              typename IndexingType,
              typename StorageType>
    inline void derivative(ContextType             &c,
                           const ScalarFieldType   &f,
                           const AccessorType      &v,
                           const AccessorType      &dv,
                           const IndexingType      &index,
                           StorageType             &storage) {
        
        _fe_data->reinit(c);
        _fe_var->init(c, v);
        _sens_fe_var->init(c, dv);
        
        uint_t
        id = 0;
        
        for (uint_t i=0; i<_fe_var->n_q_points(); i++) {
            
            c.qp = i;
            
            id = index.local_id_for_point_on_elem(c.elem, i);
            typename StorageType::view_t
            stress_qp = storage.data(id);
            
            _stress->derivative(c, f, stress_qp);
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
    typename TraitsType::fe_var_t          *_sens_fe_var;
    typename TraitsType::fe_var_t          *_fe_side_var;
    typename TraitsType::prop_t            *_prop;
    typename TraitsType::energy_t          *_energy;
    typename TraitsType::press_load_t      *_p_load;
    typename TraitsType::stress_t          *_stress;
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


// computes the vonMises stress from the stress tensor and stores it in \p stress_vm
template <typename IndexingType,
          typename StressStorageType,
          typename vonMisesStressStorageType>
inline void
compute_vonMises_stress(libMesh::System           &sys,
                        const IndexingType        &index,
                        const StressStorageType   &stress,
                        const uint_t               n_stress_comp,
                        const uint_t               n_qp,
                        vonMisesStressStorageType &stress_vm) {
    
    using scalar_t = typename StressStorageType::scalar_t;
    
    stress_vm.zero();
    
    libMesh::MeshBase::const_element_iterator
    it  = sys.get_mesh().active_local_elements_begin(),
    end = sys.get_mesh().active_local_elements_end();
    
    for ( ; it != end; it++) {
        
        const libMesh::Elem *e = *it;

        uint_t
        id = 0;
        
        for (uint_t i=0; i<n_qp; i++) {
         
            id = index.local_id_for_point_on_elem(e, i);
            
            const typename StressStorageType::view_t
            stress_e = stress.data(id);

            typename vonMisesStressStorageType::view_t
            vm_stress_e = stress_vm.data(id);

            // compute the von Mises stress and put that in the last row
            vm_stress_e(0) =
            MAST::Physics::Elasticity::LinearContinuum::vonMises_stress
            <scalar_t, 2, typename StressStorageType::view_t>(stress_e);
        }
    }
}



// computes the sensitivity of vonMises stress from
//  the stress tensor and its sensitivity and stores it in \p dstress_vm
template <typename IndexingType,
          typename StressStorageType,
          typename vonMisesStressStorageType>
inline void
compute_vonMises_stress_sensitivity(libMesh::System           &sys,
                                    const IndexingType        &index,
                                    const StressStorageType   &stress,
                                    const StressStorageType   &dstress,
                                    const uint_t               n_stress_comp,
                                    const uint_t               n_qp,
                                    vonMisesStressStorageType &dstress_vm) {
    
    dstress_vm.zero();
    
    libMesh::MeshBase::const_element_iterator
    it  = sys.get_mesh().active_local_elements_begin(),
    end = sys.get_mesh().active_local_elements_end();
    
    for ( ; it != end; it++) {
        
        const libMesh::Elem *e = *it;

        uint_t
        id = 0;
        
        for (uint_t i=0; i<n_qp; i++) {
         
            id = index.local_id_for_point_on_elem(e, i);
            
            const typename StressStorageType::view_t
            stress_e  = stress.data(id),
            dstress_e = dstress.data(id);

            typename vonMisesStressStorageType::view_t
            dvm_stress_e = dstress_vm.data(id);

            // compute the von Mises stress and put that in the last row
            dvm_stress_e(0) =
            MAST::Physics::Elasticity::LinearContinuum::vonMises_stress_derivative
            <real_t, 2, typename StressStorageType::view_t>(stress_e, dstress_e);
        }
    }
}



template <typename TraitsType, typename IndexingType>
inline void
compute_stress(Context                                        &c,
               ElemOps<TraitsType>                            &e_ops,
               const typename TraitsType::assembled_vector_t  &sol,
               const IndexingType                             &index,
               typename TraitsType::mp_storage_t              &stress) {
    
    using scalar_t   = typename TraitsType::scalar_t;

    MAST::Base::Assembly::libMeshWrapper::StressAssembly<scalar_t, ElemOps<TraitsType>>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    stress.zero();
    
    assembly.assemble(c, sol, index, stress);
}


// compute sensitivity of stress with respect to parameter \p f.
template <typename TraitsType, typename IndexingType, typename ScalarFieldType>
inline void
compute_stress_sensitivity(Context                                        &c,
                           ElemOps<TraitsType>                            &e_ops,
                           const ScalarFieldType                          &f,
                           const typename TraitsType::assembled_vector_t  &sol,
                           const typename TraitsType::assembled_vector_t  &dsol,
                           const IndexingType                             &index,
                           typename TraitsType::mp_storage_t              &dstress) {
    
    using scalar_t   = typename TraitsType::scalar_t;

    MAST::Base::Assembly::libMeshWrapper::StressAssembly<scalar_t, ElemOps<TraitsType>>
    assembly;
    
    assembly.set_elem_ops(e_ops);

    dstress.zero();
    
    assembly.sensitivity_assemble(c, f, sol, dsol, index, dstress);
}






// this identifies maximum value of each stress component on the element
// and copies it to the system for plotting
template <typename IndexingType,
          typename StressStorageType,
          typename vonMisesStressStorageType>
inline void
copy_stress_to_system(libMesh::System                  &sys,
                      const IndexingType               &index,
                      const StressStorageType          &stress,
                      const vonMisesStressStorageType  &stress_vm,
                      const uint_t                      n_stress_comp,
                      const uint_t                      n_qp) {
    
    libMesh::MeshBase::const_element_iterator
    it  = sys.get_mesh().active_local_elements_begin(),
    end = sys.get_mesh().active_local_elements_end();
    
    for ( ; it != end; it++) {
        
        const libMesh::Elem *e = *it;

        // this stores the components of stress for each quadrature point
        // in the matrix column
        Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>
        s_vals = Eigen::Matrix<real_t, Eigen::Dynamic, Eigen::Dynamic>::Zero(n_stress_comp+1, n_qp);
        
        uint_t
        id = 0;
        
        for (uint_t i=0; i<n_qp; i++) {
         
            id = index.local_id_for_point_on_elem(e, i);
            
            const typename StressStorageType::view_t
            stress_e = stress.data(id);
            
            const typename vonMisesStressStorageType::view_t
            vm_stress_e = stress_vm.data(id);

            // copy the stress components
            s_vals.col(i).topRows(n_stress_comp) = stress_e;
            
            // the von Mises stress is put in the last row
            s_vals(n_stress_comp, i) = vm_stress_e(0);
        }
        
        // identify the maximum absolute value of stress
        for (uint_t i=0; i<n_stress_comp+1; i++) {
            real_t
            v = s_vals(i,0);
            
            for (uint_t j=1; j<n_qp; j++) {
                if (fabs(v) < fabs(s_vals(i,j))) v = s_vals(i,j);
            }
            
            // set the value in the system solution vector
            sys.solution->set(e->dof_number(sys.number(), i, 0), v);
        }
        
        // copy the maximum value for each stress component to the system
        sys.solution->close();
    }
}

} // namespace Example1
} // namespace Structural
} // namespace Examples
} // namespace MAST

#ifndef MAST_TESTING

int main(int argc, const char** argv) {

    libMesh::LibMeshInit init(argc, argv);
    
    using traits_t           = MAST::Examples::Structural::Example1::Traits<real_t, real_t,    real_t, 2>;
    using traits_complex_t   = MAST::Examples::Structural::Example1::Traits<real_t, real_t, complex_t, 2>;


    MAST::Examples::Structural::Example1::Context c(init.comm());
    MAST::Examples::Structural::Example1::ElemOps<traits_t>
    e_ops(c.q_order, c.q_type, c.fe_order, c.fe_family);
    MAST::Examples::Structural::Example1::ElemOps<traits_complex_t>
    e_ops_c(c.q_order, c.q_type, c.fe_order, c.fe_family);

    typename traits_t::assembled_vector_t
    sol,
    dsol;

    typename traits_t::mp_storage_t
    stress(c.mesh->comm().get()),
    dstress(c.mesh->comm().get());

    typename traits_t::mp_vm_storage_t
    vm_stress(c.mesh->comm().get()),
    dvm_stress(c.mesh->comm().get());

    typename traits_complex_t::mp_storage_t
    stress_cs(c.mesh->comm().get());

    typename traits_complex_t::mp_vm_storage_t
    vm_stress_cs(c.mesh->comm().get());

    c.init_index();
    stress.init(c.index->n_local_points());
    dstress.init(c.index->n_local_points());
    vm_stress.init(c.index->n_local_points());
    dvm_stress.init(c.index->n_local_points());
    stress_cs.init(c.index->n_local_points());
    vm_stress_cs.init(c.index->n_local_points());

    typename traits_complex_t::assembled_vector_t
    sol_c;

    // compute the solution
    MAST::Examples::Structural::Example1::compute_sol<traits_t>(c, e_ops, sol);
    
    // compute the stresses from the solution
    MAST::Examples::Structural::Example1::compute_stress<traits_t>
    (c, e_ops, sol, *c.index, stress);
    
    // compute the vonMises stress from the stresses
    MAST::Examples::Structural::Example1::compute_vonMises_stress
    (*c.stress_sys, *c.index, stress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem(), vm_stress);

    // copy the solution to system for plotting
    MAST::Examples::Structural::Example1::copy_stress_to_system
    (*c.stress_sys, *c.index, stress, vm_stress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem());
    
    // write solution as first time-step
    libMesh::ExodusII_IO writer(*c.mesh);
    {
        for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, sol(i));
        writer.write_timestep("solution.exo", *c.eq_sys, 1, 1.);
    }

    // print the header for the table
    std::cout
    << std::setw(60) << " **********  Norm of Difference in Sensitivity ********  "
    << std::endl
    << std::setw(20) << "Solution"
    << std::setw(20) << "Stress"
    << std::setw(20) << "vonMises Stress"
    << std::endl;

    ///////////////////////////////////////////////////////////////////////////////////////
    // compute the solution sensitivity wrt E
    ///////////////////////////////////////////////////////////////////////////////////////
    {
        // add a complex perturbation to the modulus-of-elasticity
        (*e_ops_c.E)() += complex_t(0., ComplexStepDelta);
        
        // compute the solution with the complex-perturbation
        MAST::Examples::Structural::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
        
        // compute the stresses from the solution
        MAST::Examples::Structural::Example1::compute_stress<traits_complex_t>
        (c, e_ops_c, sol_c, *c.index, stress_cs);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress
        (*c.stress_sys, *c.index, stress_cs, traits_complex_t::stress_t::n_strain,
         c.n_qpoints_per_elem(), vm_stress_cs);
        
        // remove the complex perturbation
        (*e_ops_c.E)() -= complex_t(0., ComplexStepDelta);
        
        MAST::Examples::Structural::Example1::compute_sol_sensitivity<traits_t>
        (c, e_ops, *e_ops.E, sol, dsol);
        
        // compute the sensitivity stresses from the solution
        MAST::Examples::Structural::Example1::compute_stress_sensitivity<traits_t>
        (c, e_ops, *e_ops.E, sol, dsol, *c.index, dstress);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress_sensitivity
        (*c.stress_sys, *c.index, stress, dstress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem(), dvm_stress);
        
        // copy the sensitivity of stress to system for plotting
        MAST::Examples::Structural::Example1::copy_stress_to_system
        (*c.stress_sys, *c.index, dstress, dvm_stress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem());
        
        // write solution as first time-step
        {
            for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
            writer.write_timestep("solution.exo", *c.eq_sys, 2, 2.);
        }
        
        // the complex-step sensitivity is obtained from the imaginary part of the solution.
        // We compute the difference between the analytical sensitivity in \p dsol and
        // the complex-step sensitivity.
        dsol -= sol_c.imag()/ComplexStepDelta;
        
        // compute difference in sensitivity of stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        stress_vec(stress.data(), stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        stress_vec_cs(stress_cs.data(), stress_cs.size(), 1);
        
        stress_vec -= stress_vec_cs.imag()/ComplexStepDelta;
        
        // similarly, compute the difference between the analytical and complex-step
        // sensitivity of vonMises stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        vm_stress_vec(vm_stress.data(), vm_stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        vm_stress_vec_cs(vm_stress_cs.data(), vm_stress_cs.size(), 1);
        
        // print out the norm of the difference. The difference should be zero to machine
        // precision. This proceduce is followed for all the other parameters.
        std::cout
        << std::setw(20) << dsol.norm()
        << std::setw(20) << stress_vec.norm()
        << std::setw(20) << vm_stress_vec.norm()
        << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // compute the solution sensitivity wrt nu
    ///////////////////////////////////////////////////////////////////////////////////////
    {
        (*e_ops_c.nu)() += complex_t(0., ComplexStepDelta);
        
        // compute the solution with the complex-perturbation
        MAST::Examples::Structural::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
        
        // compute the stresses from the solution
        MAST::Examples::Structural::Example1::compute_stress<traits_complex_t>
        (c, e_ops_c, sol_c, *c.index, stress_cs);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress
        (*c.stress_sys, *c.index, stress_cs, traits_complex_t::stress_t::n_strain,
         c.n_qpoints_per_elem(), vm_stress_cs);
        
        // remove the complex perturbation
        (*e_ops_c.nu)() -= complex_t(0., ComplexStepDelta);
        
        MAST::Examples::Structural::Example1::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.nu, sol, dsol);
        
        MAST::Examples::Structural::Example1::compute_stress_sensitivity<traits_t>
        (c, e_ops, *e_ops.nu, sol, dsol, *c.index, stress);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress_sensitivity
        (*c.stress_sys, *c.index, stress, dstress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem(), dvm_stress);
        
        // copy the sensitivity of stress to system for plotting
        MAST::Examples::Structural::Example1::copy_stress_to_system
        (*c.stress_sys, *c.index, dstress, dvm_stress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem());
        
        // write solution as first time-step
        {
            for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
            writer.write_timestep("solution.exo", *c.eq_sys, 3, 3.);
        }
        
        // the complex-step sensitivity is obtained from the imaginary part of the solution.
        // We compute the difference between the analytical sensitivity in \p dsol and
        // the complex-step sensitivity.
        dsol -= sol_c.imag()/ComplexStepDelta;
        
        // compute difference in sensitivity of stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        stress_vec(stress.data(), stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        stress_vec_cs(stress_cs.data(), stress_cs.size(), 1);
        
        stress_vec -= stress_vec_cs.imag()/ComplexStepDelta;
        
        // similarly, compute the difference between the analytical and complex-step
        // sensitivity of vonMises stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        vm_stress_vec(vm_stress.data(), vm_stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        vm_stress_vec_cs(vm_stress_cs.data(), vm_stress_cs.size(), 1);
        
        // print out the norm of the difference. The difference should be zero to machine
        // precision. This proceduce is followed for all the other parameters.
        std::cout
        << std::setw(20) << dsol.norm()
        << std::setw(20) << stress_vec.norm()
        << std::setw(20) << vm_stress_vec.norm()
        << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // compute the solution sensitivity wrt p
    ///////////////////////////////////////////////////////////////////////////////////////
    {
        (*e_ops_c.press)() += complex_t(0., ComplexStepDelta);
        
        // compute the solution with the complex-perturbation
        MAST::Examples::Structural::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
        
        // compute the stresses from the solution
        MAST::Examples::Structural::Example1::compute_stress<traits_complex_t>
        (c, e_ops_c, sol_c, *c.index, stress_cs);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress
        (*c.stress_sys, *c.index, stress_cs, traits_complex_t::stress_t::n_strain,
         c.n_qpoints_per_elem(), vm_stress_cs);
        
        // remove the complex perturbation
        (*e_ops_c.press)() -= complex_t(0., ComplexStepDelta);
        
        MAST::Examples::Structural::Example1::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.press, sol, dsol);
        
        MAST::Examples::Structural::Example1::compute_stress_sensitivity<traits_t>
        (c, e_ops, *e_ops.press, sol, dsol, *c.index, stress);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress_sensitivity
        (*c.stress_sys, *c.index, stress, dstress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem(), dvm_stress);
        
        // copy the sensitivity of stress to system for plotting
        MAST::Examples::Structural::Example1::copy_stress_to_system
        (*c.stress_sys, *c.index, dstress, dvm_stress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem());
        
        // write solution as first time-step
        {
            for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
            writer.write_timestep("solution.exo", *c.eq_sys, 4, 4.);
        }
        
        // the complex-step sensitivity is obtained from the imaginary part of the solution.
        // We compute the difference between the analytical sensitivity in \p dsol and
        // the complex-step sensitivity.
        dsol -= sol_c.imag()/ComplexStepDelta;
        
        // compute difference in sensitivity of stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        stress_vec(stress.data(), stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        stress_vec_cs(stress_cs.data(), stress_cs.size(), 1);
        
        stress_vec -= stress_vec_cs.imag()/ComplexStepDelta;
        
        // similarly, compute the difference between the analytical and complex-step
        // sensitivity of vonMises stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        vm_stress_vec(vm_stress.data(), vm_stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        vm_stress_vec_cs(vm_stress_cs.data(), vm_stress_cs.size(), 1);
        
        // print out the norm of the difference. The difference should be zero to machine
        // precision. This proceduce is followed for all the other parameters.
        std::cout
        << std::setw(20) << dsol.norm()
        << std::setw(20) << stress_vec.norm()
        << std::setw(20) << vm_stress_vec.norm()
        << std::endl;
    }

    ///////////////////////////////////////////////////////////////////////////////////////
    // compute the solution sensitivity wrt section area
    ///////////////////////////////////////////////////////////////////////////////////////
    {
        
        (*e_ops_c.area)() += complex_t(0., ComplexStepDelta);
        
        // compute the solution with the complex-perturbation
        MAST::Examples::Structural::Example1::compute_sol<traits_complex_t>(c, e_ops_c, sol_c);
        
        // compute the stresses from the solution
        MAST::Examples::Structural::Example1::compute_stress<traits_complex_t>
        (c, e_ops_c, sol_c, *c.index, stress_cs);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress
        (*c.stress_sys, *c.index, stress_cs, traits_complex_t::stress_t::n_strain,
         c.n_qpoints_per_elem(), vm_stress_cs);
        
        // remove the complex perturbation
        (*e_ops_c.area)() -= complex_t(0., ComplexStepDelta);
        
        MAST::Examples::Structural::Example1::compute_sol_sensitivity<traits_t>(c, e_ops, *e_ops.area, sol, dsol);
        
        MAST::Examples::Structural::Example1::compute_stress_sensitivity<traits_t>
        (c, e_ops, *e_ops.area, sol, dsol, *c.index, stress);
        
        // compute the vonMises stress from the stresses
        MAST::Examples::Structural::Example1::compute_vonMises_stress_sensitivity
        (*c.stress_sys, *c.index, stress, dstress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem(), dvm_stress);
        
        // copy the sensitivity of stress to system for plotting
        MAST::Examples::Structural::Example1::copy_stress_to_system
        (*c.stress_sys, *c.index, dstress, dvm_stress, traits_t::stress_t::n_strain, c.n_qpoints_per_elem());
        
        
        // write solution as first time-step
        {
            for (uint_t i=0; i<sol.size(); i++) c.sys->solution->set(i, dsol(i));
            writer.write_timestep("solution.exo", *c.eq_sys, 5, 5.);
        }
        
        // the complex-step sensitivity is obtained from the imaginary part of the solution.
        // We compute the difference between the analytical sensitivity in \p dsol and
        // the complex-step sensitivity.
        dsol -= sol_c.imag()/ComplexStepDelta;
        
        // compute difference in sensitivity of stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        stress_vec(stress.data(), stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        stress_vec_cs(stress_cs.data(), stress_cs.size(), 1);
        
        stress_vec -= stress_vec_cs.imag()/ComplexStepDelta;
        
        // similarly, compute the difference between the analytical and complex-step
        // sensitivity of vonMises stress
        Eigen::Map<Eigen::Matrix<real_t, Eigen::Dynamic, 1>>
        vm_stress_vec(vm_stress.data(), vm_stress.size(), 1);
        
        Eigen::Map<Eigen::Matrix<complex_t, Eigen::Dynamic, 1>>
        vm_stress_vec_cs(vm_stress_cs.data(), vm_stress_cs.size(), 1);
        
        // print out the norm of the difference. The difference should be zero to machine
        // precision. This proceduce is followed for all the other parameters.
        std::cout
        << std::setw(20) << dsol.norm()
        << std::setw(20) << stress_vec.norm()
        << std::setw(20) << vm_stress_vec.norm()
        << std::endl;
    }
    
    // END_TRANSLATE
    return 0;
}

#endif // MAST_TESTING
