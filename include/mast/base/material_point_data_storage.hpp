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

#ifndef __mast_material_point_data_storage_h__
#define __mast_material_point_data_storage_h__


// libMesh includes
#include <libmesh/parallel.h>


namespace MAST {
namespace Base {

template <typename ScalarType>
class MaterialPointDataStorage {
    
public:
    
    using view_t       = Eigen::Map<typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>>;
    using const_view_t = Eigen::Map<const typename Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>>;
    
    
    MaterialPointDataStorage(libMesh::Parallel::Communicator &comm,
                             uint_t                          n_comp,
                             const std::string               &nm):
    _comm          (comm),
    _n_comp        (n_comp),
    _begin_elem_id (-1),
    _end_elem_id   (-1),
    _nm            (nm),
    _data          (nullptr) {
        
    }
    
    
    virtual ~MaterialPointDataStorage() {
        
        if (_data) delete _data;
    }
    
    
    inline void init(uint_t n_pts,
                     uint_t begin_elem_id,
                     uint_t end_elem_id) {
        
        Assert0(!_data, "Data must be cleared before reinitialization");
        
        
        _begin_elem_id  = begin_elem_id;
        _end_elem_id    = end_elem_id;
        
        _data = new ScalarType[n_pts*_n_comp];
    }
    
    
    inline view_t
    value(const uint_t e_id, const uint_t qp) {

        Assert2(e_id >= _begin_elem_id,
                e_id, _begin_elem_id,
                "Material point id out of bounds for this rank");
        Assert2(e_id < _end_elem_id,
                e_id, _end_elem_id,
                "Material point id out of bounds for this rank");

        uint_t
        begin_index = (e_id-_begin_elem_id) * _n_comp;
        
        return view_t(&(_data[begin_index]) , _n_comp, 1);
    }

    
    inline const_view_t
    value(const uint_t e_id, const uint_t qp) const {
        
        Assert2(e_id >= _begin_elem_id,
                e_id, _begin_elem_id,
                "Material point id out of bounds for this rank");
        Assert2(e_id < _end_elem_id,
                e_id, _end_elem_id,
                "Material point id out of bounds for this rank");

        uint_t
        begin_index = (e_id - _begin_elem_id) * _n_comp;
        
        return const_view_t(&(_data[begin_index]) , _n_comp, 1);
    }

    
private:
    
    libMesh::Parallel::Communicator& _comm;
    
    const uint_t                     _n_comp;
    
    /*!
     *  first element stored on this processor
     */
    uint_t                           _begin_elem_id;
    
    /*!
     *  one past the last element stored on this processor
     */
    uint_t                           _end_elem_id;
    
    const std::string                _nm;
    
    ScalarType                      *_data;
};

} // namespace Base
} // namespace MAST

#endif // __mast_material_point_data_storage_h__
