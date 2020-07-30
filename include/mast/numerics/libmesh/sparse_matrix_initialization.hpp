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

#ifndef __mast_libmesh_sparse_matrix_initialization_h__
#define __mast_libmesh_sparse_matrix_initialization_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// libMesh includes
#include <libmesh/dof_map.h>

namespace MAST {
namespace Numerics {
namespace libMeshWrapper {


template <typename P1, int P2, typename P3>
void init_sparse_matrix(const libMesh::DofMap& dof_map,
                        Eigen::SparseMatrix<P1, P2, P3>& m)  {

        Assert1(dof_map.comm().size() == 1,
                dof_map.comm().size(),
                "Eigen matrix can only be used for MPI communicator with rank 1.");
        
        m.resize(dof_map.n_dofs(), dof_map.n_dofs());
        
        m.reserve(dof_map.get_n_nz());
}

} // namespace libMeshWrapper
} // namespace Numerics
} // namespace MAST

#endif // __mast_libmesh_sparse_matrix_initialization_h__
