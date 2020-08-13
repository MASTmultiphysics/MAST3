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

#ifndef __mast_material_point_system_h__
#define __mast_material_point_system_h__

// MAST includes
#include <mast/base/mast_data_types.h>

// libMesh includes
#include <libmesh/mesh_base.h>

namespace MAST {
namespace Base {

class MaterialPointSystem {
  
public:
    
    MaterialPointSystem(libMesh::MeshBase& mesh):
    _mesh         (mesh),
    _initialized  (false)
    { }
    
    virtual ~MaterialPointSystem() {
        
        // delete the material points
        /*{
            std::vector<MAST::Base::MaterialPoint*>::iterator
            it   = _material_points.begin(),
            end  = _material_points.end();
            
            for (; it != end; it++) delete *it;
        }*/
        
        /*// delete the data
        {
            std::map<std::string, MAST::Base::MaterialPointDataStorage<ScalarType>*>::iterator
            it   = _data.begin(),
            end  = _data.end();
            
            for (; it != end; it++) delete it->second;
        }*/
    }

    void initialize(uint_t n_qp_per_elem) {

        Assert0(!_initialized, "System already initialization");
        
        _n_points_on_rank.resize(_mesh.comm().size(), 0);
        _begin_point_id.resize(_mesh.comm().size(), 0);

        uint_t
        prev_end_id = 0;
        
        for (uint_t i=0; i<_mesh.comm().size(); i++) {
            
            _n_points_on_rank[i] = _mesh.n_elem_on_proc(i) * n_qp_per_elem;
            _end_point_id[i]     = prev_end_id + _n_points_on_rank[i];
            prev_end_id          = _end_point_id[i];
        }
        
        for (uint_t i=1; i<_mesh.comm().size(); i++)
            _begin_point_id[i]   = _end_point_id[i-1];
    }
    
    
    /*inline MAST::Base::MaterialPointDataStorage<ScalarType>*>&
    add_variable(const std::string& nm, uint_t n_components)  {
        
        Assert0(!initialized, "Variables can be added prior to system initialization");

        
    }
    
    
    inline MAST::Base::MaterialPointDataStorage<ScalarType>*>&
    get_variable(const std::string& nm)  {
        
    }*/


private:

    libMesh::MeshBase                       &_mesh;

    bool                                     _initialized;
    
    std::vector<uint_t>                      _n_points_on_rank;
    
    std::vector<uint_t>                      _begin_point_id;

    std::vector<uint_t>                      _end_point_id;

    //std::vector<MAST::Base::MaterialPoint*>  _material_points;
    
    //std::map<std::string, MAST::Base::MaterialPointDataStorage<ScalarType>*> _data;
};

} // namespace Base
} // namespace MAST

#endif // __mast_material_point_system_h__
