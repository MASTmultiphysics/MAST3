
#ifndef __mast_numerics_utility_h__
#define __mast_numerics_utility_h__

// C++ includes
#include <vector>

// libMesh includes
#include <libmesh/numeric_vector.h>
#include <libmesh/sparse_matrix.h>

namespace MAST {
namespace Numerics {
namespace Utility {

template <typename ValType>
void
setZero(ValType& m) { m.setZero();}

template <typename ValType>
typename std::enable_if<std::is_same<libMesh::NumericVector<real_t>, ValType>::value, void>::type
setZero(libMesh::NumericVector<real_t>& v) { v.zero();}

template <typename ValType>
typename std::enable_if<std::is_same<libMesh::SparseMatrix<real_t>, ValType>::value, void>::type
setZero(libMesh::SparseMatrix<real_t>& m) { m.zero();}

template <typename ValType>
void
finalize(ValType& m) { }

template <typename ValType>
typename std::enable_if<std::is_same<libMesh::NumericVector<real_t>, ValType>::value, void>::type
finalize(libMesh::NumericVector<real_t>& v) { v.close();}

template <typename ValType>
typename std::enable_if<std::is_same<libMesh::SparseMatrix<real_t>, ValType>::value, void>::type
finalize(libMesh::SparseMatrix<real_t>& m) { m.close();}

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
