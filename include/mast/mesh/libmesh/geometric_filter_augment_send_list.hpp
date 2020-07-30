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

#ifndef __mast_geometric_filter_augment_send_list_h__
#define __mast_geometric_filter_augment_send_list_h__


// libMesh includes
#include <libmesh/dof_map.h>


namespace MAST {
namespace Mesh {
namespace libMeshWrapper {


class GeometricFilterAugmentSendList:
public libMesh::DofMap::AugmentSendList {

public:
    
    GeometricFilterAugmentSendList(const std::vector<uint_t>& v):
    _list  (v)
    { }

    virtual ~GeometricFilterAugmentSendList() { }

    virtual void
    augment_send_list(std::vector<libMesh::dof_id_type>& send_list) override {
        
        for (uint_t i=0; i<_list.size(); i++)
            send_list.push_back(_list[i]);
    }

private:
    
    const std::vector<uint_t>& _list;
};

} // namespace libMeshWrapper
} // namespace Mesh
} // namespace MAST


#endif // __mast_geometric_filter_augment_send_list_h__
