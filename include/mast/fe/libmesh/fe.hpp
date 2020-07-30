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

#ifndef __mast__libmesh_fe_h__
#define __mast__libmesh_fe_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/quadrature/libmesh/quadrature.hpp>

// libMesh includes
#include <libmesh/fe_base.h>

namespace MAST {

namespace FEBasis     {
namespace libMeshWrapper {

template <typename ScalarType, uint_t Dim>
class FEBasis {
    
public:

    using scalar_t              = ScalarType;
    using fe_t                  = libMesh::FEBase;
    using quadrature_t          = typename MAST::Quadrature::libMeshWrapper::Quadrature<ScalarType, Dim>;
    using side_quadrature_t     = typename MAST::Quadrature::libMeshWrapper::Quadrature<ScalarType, Dim-1>;
    using elem_t                = libMesh::Elem;
    using phi_vec_t             = typename Eigen::Map<const typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>>;
    using dphi_dxi_vec_t        = typename Eigen::Map<const typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>>;
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
                       quadrature_t&  q) {
     
        // reinitialize only if needed
        if (&e != _elem || &q != _q) {
            
            _fe->attach_quadrature_rule(&q.quadrature_object());
            _fe->reinit(&e);
            _q      = &q;
            _elem   = &e;
            _side   = -1;
            _q_side = nullptr;

            _phi.setZero(this->n_basis(), this->n_q_points());
            
            for (uint_t k=0; k<this->n_q_points(); k++)
                for (uint_t j=0; j<this->n_basis(); j++)
                    _phi(j, k) = _fe->get_phi()[j][k];

            if (_compute_dphi_dxi) {

                _dphi_dxi.setZero(Dim*this->n_basis(), this->n_q_points());
                
                for (uint_t k=0; k<this->n_q_points(); k++)
                    for (uint_t j=0; j<this->n_basis(); j++)
                        _dphi_dxi(j, k) = _fe->get_fe_map().get_dphidxi_map()[j][k];
                
                if (Dim > 1) {
                    for (uint_t k=0; k<this->n_q_points(); k++)
                        for (uint_t j=0; j<this->n_basis(); j++)
                            _dphi_dxi(this->n_basis()+j, k) = _fe->get_fe_map().get_dphideta_map()[j][k];
                }

                if (Dim > 2) {
                    for (uint_t k=0; k<this->n_q_points(); k++)
                        for (uint_t j=0; j<this->n_basis(); j++)
                            _dphi_dxi(2*this->n_basis()+j, k) = _fe->get_fe_map().get_dphidzeta_map()[j][k];
                }
            }
            else
                _dphi_dxi.setZero();
        }
    }

    inline void reinit_for_side(const libMesh::Elem     &e,
                                side_quadrature_t       &q,
                                const uint_t             s) {
        
        // reinitialize only if needed
        if (&e != _elem ||
            &q != _q_side ||
            s  != _side) {
            
            _fe->attach_quadrature_rule(&q.quadrature_object());
            _fe->reinit(&e, s);
            
            _q      = nullptr;
            _elem   = &e;
            _side   = s;
            _q_side = &q;
            
            _phi.setZero(this->n_basis(), this->n_q_points());
            
            for (uint_t k=0; k<this->n_q_points(); k++)
                for (uint_t j=0; j<this->n_basis(); j++)
                    _phi(j, k) = _fe->get_phi()[j][k];

            if (_compute_dphi_dxi) {
                
                _dphi_dxi.setZero(Dim*this->n_basis(), this->n_q_points());

                for (uint_t k=0; k<this->n_q_points(); k++)
                    for (uint_t j=0; j<this->n_basis(); j++)
                        _dphi_dxi(j, k) = _fe->get_fe_map().get_dphidxi_map()[j][k];
                
                if (Dim > 1) {
                    for (uint_t k=0; k<this->n_q_points(); k++)
                        for (uint_t j=0; j<this->n_basis(); j++)
                            _dphi_dxi(this->n_basis()+j, k) = _fe->get_fe_map().get_dphideta_map()[j][k];
                }

                if (Dim > 2) {
                    for (uint_t k=0; k<this->n_q_points(); k++)
                        for (uint_t j=0; j<this->n_basis(); j++)
                            _dphi_dxi(2*this->n_basis()+j, k) = _fe->get_fe_map().get_dphidzeta_map()[j][k];
                }
            }
            else
                _dphi_dxi.setZero();
        }
    }

    inline uint_t n_basis() const { return _fe->n_shape_functions();}
    
    inline scalar_t qp_weight(uint_t qp) const {

        Assert0(_q || _q_side, "Quadrature rule must be specified before quadrature weight can be obtained");
        return _q?_q->weight(qp):_q_side->weight(qp);
    }

    inline phi_vec_t phi(uint_t qp) const {

        Assert2(qp < this->n_q_points(),
                qp, this->n_q_points(),
                "Invalid quadrature point index");

        return phi_vec_t(_phi.col(qp).data(), this->n_basis());
    }

    inline scalar_t phi(uint_t qp, uint_t phi_i) const {
        
        Assert2(phi_i < this->n_basis(),
                phi_i, this->n_basis(),
                "Invalid shape function index");
        Assert2(qp < this->n_q_points(),
                qp, this->n_q_points(),
                "Invalid quadrature point index");
        
        return _phi(phi_i, qp);
    }
    
    inline const dphi_dxi_vec_t
    dphi_dxi(uint_t qp, uint_t xi_i) const {
        
        Assert0(_compute_dphi_dxi, "FE not initialized with basis derivatives.");
        return dphi_dxi_vec_t(_dphi_dxi.col(qp).segment
                              (xi_i*this->n_basis(), this->n_basis()).data(),
                              this->n_basis());
    }
    
    inline scalar_t
    dphi_dxi(uint_t qp, uint_t phi_i, uint_t xi_i) const {
        
        Assert0(_compute_dphi_dxi, "FE not initialized with basis derivatives.");
        return _dphi_dxi(xi_i*this->n_basis()+phi_i, qp);
    }

private:
    
    fe_t                                                      *_fe;
    bool                                                       _own_pointer;
    bool                                                       _compute_dphi_dxi;
    const quadrature_t                                        *_q;
    const side_quadrature_t                                   *_q_side;
    const elem_t                                              *_elem;
    uint_t                                                     _side;
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>  _phi;
    Eigen::Matrix<ScalarType, Eigen::Dynamic, Eigen::Dynamic>  _dphi_dxi;
};

}  // namespace libMeshWrapper
}  // namespace FE
}  // namespace MAST

#endif // __mast__libmesh_fe_h__
