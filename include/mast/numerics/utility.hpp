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
#include <libmesh/parallel.h>
#include <libmesh/system.h>

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


template <typename VecType>
inline typename
std::enable_if<std::is_same<typename Eigen::internal::traits<VecType>::Scalar,
                            real_t>::value, real_t>::type
real_norm(const VecType& v) {
    return v.norm();
}


template <typename VecType>
inline typename
std::enable_if<std::is_same<typename Eigen::internal::traits<VecType>::Scalar,
                            complex_t>::value, real_t>::type
real_norm(const VecType& v) {
    return v.norm();
}


#if MAST_ENABLE_ADOLC == 1
template <typename VecType>
inline typename
std::enable_if<std::is_same<typename Eigen::internal::traits<VecType>::Scalar,
                            adouble_tl_t>::value, real_t>::type
real_norm(const VecType& v) {
    return v.norm().getValue();
}
#endif

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


template <typename ScalarType>
inline void resize(std::vector<ScalarType>& v, uint_t n) {

    v.resize(n, ScalarType());
}


template <typename ScalarType>
inline void resize(Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>& v, uint_t n) {

    v = Eigen::Matrix<ScalarType, Eigen::Dynamic, 1>::Zero(n);
}


template <typename ValType>
inline std::unique_ptr<ValType>
build(const libMesh::System& sys) {

    std::unique_ptr<ValType> rval(new ValType);
    MAST::Numerics::Utility::resize(*rval, sys.n_dofs());
    
    return rval;
}


template <>
inline std::unique_ptr<libMesh::NumericVector<real_t>>
build(const libMesh::System& sys) {

    return std::unique_ptr<libMesh::NumericVector<real_t>>
    (sys.solution->zero_clone().release());
}



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


inline void
comm_sum(const libMesh::Parallel::Communicator& comm,
         real_t& v) {
    comm.sum(v);
}


inline void
comm_sum(const libMesh::Parallel::Communicator& comm,
         complex_t& v) {
    real_t
    v_re = v.real(),
    v_im = v.imag();
    
    comm.sum(v_re);
    comm.sum(v_im);
    
    v.real(v_re);
    v.imag(v_im);
}


inline void
comm_sum(const libMesh::Parallel::Communicator& comm,
         std::vector<real_t>& v) {
    comm.sum(v);
}


inline void
comm_sum(const libMesh::Parallel::Communicator& comm,
         std::vector<complex_t>& v) {
    
    std::vector<real_t>
    v_re(v.size()),
    v_im(v.size());
    
    for (uint_t i=0; i<v.size(); i++) {
        
        v_re[i] = v[i].real();
        v_im[i] = v[i].imag();
    }
    
    comm.sum(v_re);
    comm.sum(v_im);
    
    for (uint_t i=0; i<v.size(); i++) {
        
        v[i].real(v_re[i]);
        v[i].imag(v_im[i]);
    }
}


} // namespace Utility
} // namespace Numerics
} // namespace MAST

#endif // __mast_numerics_utility_h__
