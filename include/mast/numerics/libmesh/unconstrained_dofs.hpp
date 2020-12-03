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

#ifndef __mast_libmesh_unconstrained_dofs_h__
#define __mast_libmesh_unconstrained_dofs_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// libMesh includes
#include <libmesh/dof_map.h>

namespace MAST {
namespace Numerics {
namespace libMeshWrapper {

/*!
 * copies the unconstrained dofs for the \p dof_map in the vector \p dofs. This is a
 * collective operation and each rank will store only its own set of unconstrained dofs.
 */
template <typename IntType>
void unconstrained_dofs(const libMesh::DofMap  &dof_map,
                        std::vector<IntType>   &dofs)  {
    
    std::set<IntType>
    local_non_condensed_dofs_set;
    
    for (IntType i=dof_map.first_dof(); i<dof_map.end_dof(); i++) {
        
        if (!dof_map.is_constrained_dof(i))
            local_non_condensed_dofs_set.insert(i);
    }

    std::set<IntType>::const_iterator
    it   = local_non_condensed_dofs.begin(),
    end  = local_non_condensed_dofs.end();
    
    dofs.clear();
    dofs.reserve(local_non_condensed_dofs().size());
    
    for ( ; it != end; it++)
        dofs.push_back(dof);
}

} // namespace libMeshWrapper
} // namespace Numerics
} // namespace MAST

#endif // __mast_libmesh_unconstrained_dofs_h__
