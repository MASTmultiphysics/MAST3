
#ifndef __mast_numerics_utility_h__
#define __mast_numerics_utility_h__

// C++ includes
#include <vector>
#include <type_traits>

// MAST includes
#include <mast/base/mast_data_types.h>

// libMesh includes
#include <libmesh/numeric_vector.h>
#include <libmesh/sparse_matrix.h>

namespace MAST {
namespace Numerics {
namespace Utility {

template <typename ValType>
void
setZero(ValType& m) { m.setZero();}

void
setZero(libMesh::NumericVector<real_t>& v) { v.zero();}

void
setZero(libMesh::SparseMatrix<real_t>& m) { m.zero();}

template <typename ValType>
void
finalize(ValType& m) { }


void
finalize(libMesh::NumericVector<real_t>& v) { v.close();}


void
finalize(libMesh::SparseMatrix<real_t>& m) { m.close();}


template <typename P1, int P2, typename P3>
void
finalize(Eigen::SparseMatrix<P1, P2, P3>& m) { m.makeCompressed();}


template <typename ScalarType, typename VecType>
void copy(const VecType& v_from,
          libMesh::DenseVector<ScalarType>& v_to) {
    
    v_to.resize(v_from.size());
    
    for (uint_t i=0; i<v_from.size(); i++)
        v_to(i) = v_from(i);
}

template <typename ScalarType, typename MatType>
void copy(const MatType& m_from,
          libMesh::DenseMatrix<ScalarType>& m_to) {
    
    m_to.resize(m_from.rows(), m_from.cols());
    
    for (uint_t i=0; i<m_from.cols(); i++)
        for (uint_t j=0; j<m_from.cols(); j++)
            m_to(i, j) = m_from(i, j);
}


} // namespace Utility
} // namespace Numerics
} // namespace MAST

#endif // __mast_numerics_utility_h__
