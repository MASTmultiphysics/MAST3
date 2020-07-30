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

#ifndef __mast__libmesh_quadrature_h__
#define __mast__libmesh_quadrature_h__

// MAST includes
#include <mast/base/mast_data_types.h>

// libMesh includes
#include <libmesh/quadrature.h>
#include <libmesh/enum_quadrature_type.h>

namespace MAST {

namespace Quadrature     {
namespace libMeshWrapper {

template <typename ScalarType, uint_t Dim>
class Quadrature {
    
public:

    using scalar_t          = ScalarType;
    using quadrature_t      = libMesh::QBase;
    static const uint_t dim = Dim;
    static_assert (std::is_same<scalar_t, double>::value,
                   "Class only implemented for scalar type = double.");
    
    Quadrature(quadrature_t& q):
    _q           (&q),
    _own_pointer (false) {
        
    }

    Quadrature(const libMesh::QuadratureType qt,
               const libMesh::Order order):
    _q           (libMesh::QBase::build (qt, Dim, order).release()),
    _own_pointer (true) {
        
    }

    ~Quadrature() {
        
        if (_own_pointer)
            delete _q;
    }
    
    inline uint_t order() const { return _q->get_order();}
    inline uint_t n_points() const { return _q->n_points();}
    inline scalar_t qp_coord(uint_t qp, uint_t xi_i) const { return _q->get_points()[qp](xi_i);}
    inline scalar_t weight(uint_t qp) const { return _q->w(qp);}
    inline quadrature_t& quadrature_object() { return *_q;}
    inline const quadrature_t& quadrature_object() const { return *_q;}

private:
    
    quadrature_t   *_q;
    bool            _own_pointer;
};

}  // namespace libMeshWrapper
}  // namespace Quadrature
}  // namespace MAST

#endif // __mast__libmesh_quadrature_h__
