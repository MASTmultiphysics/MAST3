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

#ifndef __mast_material_point_libmesh_indexing_h__
#define __mast_material_point_libmesh_indexing_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// libMesh includes
#include <libmesh/mesh_base.h>

namespace MAST {
namespace Base {
namespace MaterialPoint {
namespace libMeshWrapper {

/*!
 * indexes the material point ID to quadrature points on each element. Only IDs for local elements are stored and
 * IDs are numbered contiguously across processors. The IDs on rank 0 will be in range
 *  [0, num_local_elements_on_rank_0 * num_quadrature_pts_per_elem], and on rank 0 will be in range
 *  num_local_elements_on_rank_0 * num_quadrature_pts_per_elem + [0, num_local_elements_on_rank_1 * num_quadrature_pts_per_elem].
 */
class Indexing {
    
public:
      
    Indexing():
    _initialized    (false),
    _mesh           (nullptr),
    _n_points       (0),
    _begin_local_id (0),
    _end_local_id   (0) {
        
    }
    
    virtual ~Indexing() { }

    inline uint_t n_local_points() const {
        
        Assert0(_initialized, "Object must be initialized");
        return _end_local_id - _begin_local_id;
    }
    
    inline void init(const libMesh::MeshBase &mesh, uint_t n_points_per_elem) {
        
        Assert0(!_initialized, "Object already initialized");
        
        _mesh     = &mesh;
        _n_points = n_points_per_elem;
        
        libMesh::MeshBase::const_element_iterator
        it   = mesh.active_local_elements_begin(),
        end  = mesh.active_local_elements_end();
        
        for ( ; it != end; it++) {
            
            const libMesh::Elem* e = *it;
            _elem_id_map[e] = _end_local_id;
            _end_local_id  += n_points_per_elem;
        }
        
        uint_t
        comm_rank = mesh.comm().rank(),
        comm_size = mesh.comm().size();
        
        
        std::vector<uint_t>
        n_ids_on_rank(comm_size);
        
        // number of points on this rank
        n_ids_on_rank[comm_rank] = _end_local_id;
        
        mesh.comm().sum(n_ids_on_rank);
        
        for (uint_t i=1; i<comm_size; i++)
            n_ids_on_rank[i] += n_ids_on_rank[i-1];
        
        if (comm_rank > 0)
            _begin_local_id = n_ids_on_rank[comm_rank-1];
        _end_local_id = n_ids_on_rank[comm_rank];
        
        _initialized = true;
    }

    inline uint_t
    local_id_for_point_on_elem(const libMesh::Elem *e,
                               uint_t               i) const {
        
        Assert0(_initialized, "Object must be initialized");
        Assert2(i < _n_points, i, _n_points,
                "Index must be less than points per element");
        
        std::map<const libMesh::Elem*, uint_t>::const_iterator
        it  = _elem_id_map.find(e);
        
        Assert0( it != _elem_id_map.end(), "Element not in map");
        
        return it->second;
    }


    inline uint_t
    global_id_for_point_on_elem(const libMesh::Elem *e,
                                uint_t               i) const {
        
        return _begin_local_id + this->local_id_for_point_on_elem(e, i);
    }

private:
    
    bool                                    _initialized;
    const libMesh::MeshBase                *_mesh;
    uint_t                                  _n_points;
    uint_t                                  _begin_local_id;
    uint_t                                  _end_local_id;
    std::map<const libMesh::Elem*, uint_t>  _elem_id_map;
};

}  // namespace libMeshWrapper
}  // namespace MaterialPoint
}  // namespace Base
}  // namespace MAST

#endif // __mast_material_point_libmesh_indexing_h__
