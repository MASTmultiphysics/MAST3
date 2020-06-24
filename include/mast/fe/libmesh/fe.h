
#ifndef __mast__libmesh_fe_h__
#define __mast__libmesh_fe_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// libMesh includes
#include "libmesh/fe_base.h"

namespace MAST {

namespace FEBasis     {
namespace libMeshWrapper {


template <uint_t Dim>
void
init_basis_derivative_map(const libMesh::FEMap& fe_m,
                          std::vector<const std::vector<std::vector<double>>*>& map)
{ Assert0(false, "Implementation for specific dimensions");}

template <>
void
init_basis_derivative_map<1>(const libMesh::FEMap& fe_m,
                             std::vector<const std::vector<std::vector<double>>*>& dphi_dxi)
{ dphi_dxi = {&fe_m.get_dphidxi_map()};}

template <>
void
init_basis_derivative_map<2>(const libMesh::FEMap& fe_m,
                             std::vector<const std::vector<std::vector<double>>*>& dphi_dxi)
{ dphi_dxi = {&fe_m.get_dphidxi_map(), &fe_m.get_dphideta_map()};}

template <>
void
init_basis_derivative_map<3>(const libMesh::FEMap& fe_m,
                             std::vector<const std::vector<std::vector<double>>*>& dphi_dxi)
{ dphi_dxi = {&fe_m.get_dphidxi_map(), &fe_m.get_dphideta_map(), &fe_m.get_dphidzeta_map()};}




template <typename ScalarType, uint_t Dim>
class FEBasis {
    
public:

    using scalar_t              = ScalarType;
    using fe_t                  = libMesh::FEBase;
    using quadrature_t          = typename MAST::Quadrature::libMeshWrapper::Quadrature<ScalarType, Dim>;
    using side_quadrature_t     = typename MAST::Quadrature::libMeshWrapper::Quadrature<ScalarType, Dim-1>;
    using elem_t                = libMesh::Elem;
    static const uint_t dim     = Dim;
    static_assert (std::is_same<scalar_t, double>::value,
                   "Class only implemented for scalar type = double.");
    
    FEBasis(fe_t& fe):
    _fe               (&fe),
    _own_pointer      (false),
    _compute_dphi_dxi (false),
    _q                (nullptr),
    _q_side           (nullptr),
    _elem             (nullptr),
    _side             (-1) {
        
        _fe->get_phi();
    }

    FEBasis(const libMesh::FEType fe_type):
    _fe               (libMesh::FEBase::build (Dim, fe_type).release()),
    _own_pointer      (true),
    _compute_dphi_dxi (false),
    _q                (nullptr),
    _q_side           (nullptr),
    _elem             (nullptr),
    _side             (-1) {
        
        _fe->get_phi();
    }

    virtual ~FEBasis() {
        
        if (_own_pointer)
            delete _fe;
    }
    
    inline uint_t order() const { return _fe->get_order();}

    inline void set_compute_dphi_dxi(bool f) {
        
        _compute_dphi_dxi = f;
        _fe->get_JxW();
    }

    virtual inline uint_t n_q_points() const {
        
        Assert0(_q || _q_side, "Quadrature not initialized.");
        return _q?_q->n_points():_q_side->n_points();
    }

    inline void reinit(const libMesh::Elem& e,
                       const quadrature_t&  q) {
     
        // reinitialize only if needed
        if (&e != _elem || &q != _q) {
            
            _fe->attach_quadrature_rule(&q.quadrature_object());
            _fe->reinit(&e);
            _q      = &q;
            _elem   = &e;
            _side   = -1;
            _q_side = nullptr;
            
            if (_compute_dphi_dxi)
                MAST::FEBasis::libMeshWrapper::init_basis_derivative_map<Dim>(_dphi_dxi,
                                                                              _fe->get_fe_map());
            else
                _dphi_dxi.clear();
        }
    }

    inline void reinit_side(const libMesh::Elem&     e,
                            const side_quadrature_t& q,
                            const uint_t             s) {
        
        // reinitialize only if needed
        if (&e != _elem ||
            &q != _q_side ||
            s  != _side) {
            
            _fe->attach_quadrature_rule(&q.quadrature_object());
            _fe->reinit(&e, &q.quadrature_object(), s);
            
            _q      = nullptr;
            _elem   = &e;
            _side   = s;
            _q_side = &q;
            
            if (_compute_dphi_dxi)
                MAST::FEBasis::libMeshWrapper::init_basis_derivative_map<Dim>(_dphi_dxi,
                                                                              _fe->get_fe_map());
            else
                _dphi_dxi.clear();
        }
    }

    inline uint_t n_basis() const { return _fe->n_shape_functions();}
    
    inline scalar_t phi(uint_t qp, uint_t phi_i) const { return _fe->get_phi()[phi_i][qp];}
    
    inline scalar_t
    dphi_dxi(uint_t qp, uint_t phi_i, uint_t xi_i) const { return (*_dphi_dxi[xi_i])[phi_i][qp];}

private:
    
    fe_t                     *_fe;
    bool                      _own_pointer;
    bool                      _compute_dphi_dxi;
    const quadrature_t       *_q;
    const side_quadrature_t  *_q_side;
    const elem_t             *_elem;
    uint_t                    _side;
    std::vector<const std::vector<std::vector<scalar_t>>*> _dphi_dxi;
};

}  // namespace libMeshWrapper
}  // namespace FE
}  // namespace MAST

#endif // __mast__libmesh_fe_h__
