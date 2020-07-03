
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
