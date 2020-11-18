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

#ifndef __mast_mesh_libmesh_utility_h__
#define __mast_mesh_libmesh_utility_h__

// MAST includes
#include <mast/base/mast_data_types.h>

// libMesh includes
#include <libmesh/elem.h>
#include <libmesh/enum_elem_type.h>

namespace MAST {
namespace Mesh {
namespace libMeshWrapper {
namespace Utility {


/*!
 * identifies number of ndoes on element
 */
inline uint_t n_linear_basis_nodes_on_elem(const libMesh::Elem& e) {
    
    switch (e.type()) {
        case libMesh::QUAD4:
        case libMesh::QUAD9:
            return 4;
            break;
            
        case libMesh::HEX8:
        case libMesh::HEX27:
            return 8;
            break;
            
        default:
            Error(false, "Elem type must be QUAD4/QUAD9 for 2D or HEX8/HEX27 for 3D");
    }
}



} // namespace Utility
} // namespace libMesh
} // namespace Mesh
} // namespace MAST

#endif // __mast_mesh_libmesh_utility_h__
