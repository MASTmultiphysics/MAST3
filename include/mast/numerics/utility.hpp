
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
inline void
setZero(ValType& m) { m.setZero();}


inline void
setZero(libMesh::NumericVector<real_t>& v) { v.zero();}


inline void
setZero(libMesh::SparseMatrix<real_t>& m) { m.zero();}

template <typename ScalarType>
inline void
setZero(std::vector<ScalarType>& v) { std::fill(v.begin(), v.end(), ScalarType());}


template <typename ScalarType>
inline void
add(std::vector<ScalarType>& v, uint_t i, ScalarType s) {
    v[i] += s;
}


inline void
add(libMesh::NumericVector<real_t>& v, uint_t i, real_t s) {
    v.add(i, s);
}


template <typename ScalarType>
inline void
add(Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& v,
    uint_t i,
    ScalarType s) {
    v(i) += s;
}


template <typename ScalarType>
inline void
set(std::vector<ScalarType>& v, uint_t i, ScalarType s) {
    v[i] = s;
}


inline void
set(libMesh::NumericVector<real_t>& v, uint_t i, real_t s) {
    v.set(i, s);
}


template <typename ScalarType>
inline void
set(Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& v,
    uint_t i,
    ScalarType s) {
    v(i) = s;
}


template <typename ScalarType>
inline ScalarType
get(const std::vector<ScalarType>& v, uint_t i) {
    return v[i];
}


inline real_t
get(const libMesh::NumericVector<real_t>& v, uint_t i) {
    return v.el(i);
}


template <typename ScalarType>
inline ScalarType
get(const Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& v, uint_t i) {
    return v(i);
}


template <typename ValType>
inline void
finalize(ValType& m) { }


inline void
finalize(libMesh::NumericVector<real_t>& v) { v.close();}


inline void
finalize(libMesh::SparseMatrix<real_t>& m) { m.close();}


template <typename P1, int P2, typename P3>
inline void
finalize(Eigen::SparseMatrix<P1, P2, P3>& m) { m.makeCompressed();}


template <typename ScalarType, typename VecType>
inline void
copy(const VecType& v_from,
     libMesh::DenseVector<ScalarType>& v_to) {
    
    v_to.resize(v_from.size());
    
    for (uint_t i=0; i<v_from.size(); i++)
        v_to(i) = v_from(i);
}

template <typename ScalarType, typename MatType>
inline void
copy(const MatType& m_from,
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
