
#ifndef __mast_libmesh_assembly_utility_h__
#define __mast_libmesh_assembly_utility_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/utility.hpp>

// libMesh includes
#include <libmesh/dof_map.h>
#include <libmesh/dense_vector.h>
#include <libmesh/dense_matrix.h>
#include <libmesh/numeric_vector.h>
#include <libmesh/sparse_matrix.h>

namespace MAST {
namespace Base {
namespace Assembly {
namespace libMeshWrapper {


template <typename ScalarType, typename MatType>
inline void
add_to_matrix(MatType& m, const uint_t i, const uint_t j, const ScalarType& v) {
    
    m(i, j) += v;
}


template <typename ScalarType, int P2, typename P3>
inline void
add_to_matrix(Eigen::SparseMatrix<ScalarType, P2, P3> &m,
              const uint_t i, const uint_t j, const ScalarType& v) {
    
    m.coeffRef(i, j) += v;
}


template <typename ScalarType,
          typename VecType,
          typename MatType,
          typename SubVecType,
          typename SubMatType>
inline
typename std::enable_if<std::is_same<ScalarType, real_t>::value, void>::type
constrain_and_add_matrix_and_vector(VecType                           &v,
                                    MatType                           &m,
                                    const libMesh::DofMap             &dof_map,
                                    std::vector<libMesh::dof_id_type> &dof_indices,
                                    SubVecType                        &v_sub,
                                    SubMatType                        &m_sub) {

    libMesh::DenseVector<real_t> v1;
    libMesh::DenseMatrix<real_t> m1;
    MAST::Numerics::Utility::copy(v_sub, v1);
    MAST::Numerics::Utility::copy(m_sub, m1);

    dof_map.constrain_element_matrix_and_vector(m1, v1, dof_indices);
    
    for (uint_t i=0; i<dof_indices.size(); i++)
        v(dof_indices[i]) += v1(i);
    
    for (uint_t i=0; i<dof_indices.size(); i++)
        for (uint_t j=0; j<dof_indices.size(); j++)
            add_to_matrix(m, dof_indices[i], dof_indices[j], m1(i,j));
}


template <typename ScalarType, typename VecType, typename SubVecType>
inline
typename std::enable_if<std::is_same<ScalarType, real_t>::value, void>::type
constrain_and_add_vector(VecType                           &v,
                         const libMesh::DofMap             &dof_map,
                         std::vector<libMesh::dof_id_type> &dof_indices,
                         SubVecType                        &v_sub) {
    
    libMesh::DenseVector<real_t> v1;
    MAST::Numerics::Utility::copy(v_sub, v1);

    dof_map.constrain_element_vector(v1, dof_indices);

    for (uint_t i=0; i<dof_indices.size(); i++)
        v(dof_indices[i]) += v1(i);
}



template <typename ScalarType, typename MatType, typename SubMatType>
inline
typename std::enable_if<std::is_same<ScalarType, real_t>::value, void>::type
constrain_and_add_matrix(MatType                           &m,
                         const libMesh::DofMap             &dof_map,
                         std::vector<libMesh::dof_id_type> &dof_indices,
                         SubMatType                        &m_sub) {
    
    libMesh::DenseMatrix<real_t> m1;
    MAST::Numerics::Utility::copy(m_sub, m1);

    dof_map.constrain_element_matrix(m1, dof_indices);

    for (uint_t i=0; i<dof_indices.size(); i++)
        for (uint_t j=0; j<dof_indices.size(); j++)
            add_to_matrix(m, dof_indices[i], dof_indices[j], m1(i,j));
}



template <typename ScalarType,
          typename VecType,
          typename MatType,
          typename SubVecType,
          typename SubMatType>
inline
typename std::enable_if<std::is_same<ScalarType, complex_t>::value, void>::type
constrain_and_add_matrix_and_vector(VecType                           &v,
                                    MatType                           &m,
                                    const libMesh::DofMap             &dof_map,
                                    std::vector<libMesh::dof_id_type> &dof_indices,
                                    SubVecType                        &v_sub,
                                    SubMatType                        &m_sub) {

    Assert2(v_sub.size() == dof_indices.size(),
            v_sub.size(), dof_indices.size(),
            "Incompatible vector size");
    Assert2(m_sub.rows() == dof_indices.size(),
            m_sub.rows(), dof_indices.size(),
            "Incompatible matrix rows");
    Assert2(m_sub.cols() == dof_indices.size(),
            m_sub.cols(), dof_indices.size(),
            "Incompatible matrix columns");

    // complex matrix and vector require that the complex and imaginary parts of the matrix
    // and vector be independently constrained
    libMesh::DenseVector<real_t>
    v_re(v_sub.size()),
    v_im(v_sub.size());
    libMesh::DenseMatrix<real_t>
    m_re(v_sub.size(), v_sub.size()),
    m_im(v_sub.size(), v_sub.size());

    std::vector<libMesh::dof_id_type>
    dof_indices_im = dof_indices;
    
    // copy the real and imaginary parts
    for (uint_t i=0; i<v_sub.size(); i++) {
        
        v_re(i) = v_sub(i).real();
        v_im(i) = v_sub(i).imag();
        
        for (uint_t j=0; j<v_sub.size(); j++) {
            
            m_re(i, j) = m_sub(i, j).real();
            m_im(i, j) = m_sub(i, j).imag();
        }
    }

    // now constraint the vector and matrix. The imaginary part is homogenously constrained
    //
    dof_map.constrain_element_matrix_and_vector(m_re, v_re, dof_indices);
    dof_map.constrain_element_matrix_and_vector(m_im, v_im, dof_indices_im);
    
    
    for (uint_t i=0; i<dof_indices.size(); i++) {
        
        v(dof_indices[i]) += complex_t(v_re(i), v_im(i));

        for (uint_t j=0; j<dof_indices.size(); j++)
            add_to_matrix(m, dof_indices[i], dof_indices[j], complex_t(m_re(i,j), m_im(i,j)));
    }
}


template <typename ScalarType, typename VecType, typename SubVecType>
inline
typename std::enable_if<std::is_same<ScalarType, complex_t>::value, void>::type
constrain_and_add_vector(VecType                           &v,
                         const libMesh::DofMap             &dof_map,
                         std::vector<libMesh::dof_id_type> &dof_indices,
                         SubVecType                        &v_sub) {
    
    Assert2(v_sub.size() == dof_indices.size(),
            v_sub.size(), dof_indices.size(),
            "Incompatible vector size");

    // complex matrix and vector require that the complex and imaginary parts of the matrix
    // and vector be independently constrained
    libMesh::DenseVector<real_t>
    v_re(v_sub.size()),
    v_im(v_sub.size());

    std::vector<libMesh::dof_id_type>
    dof_indices_im = dof_indices;
    
    // copy the real and imaginary parts
    for (uint_t i=0; i<v_sub.size(); i++) {
        
        v_re(i) = v_sub(i).real();
        v_im(i) = v_sub(i).imag();
    }

    // now constraint the vector and matrix. The imaginary part is homogenously constrained
    //
    dof_map.constrain_element_vector(v_re, dof_indices);
    dof_map.constrain_element_vector(v_im, dof_indices_im);
    
    
    for (uint_t i=0; i<dof_indices.size(); i++)
        v(dof_indices[i]) += complex_t(v_re(i), v_im(i));
}


template <typename ScalarType, typename MatType, typename SubMatType>
inline
typename std::enable_if<std::is_same<ScalarType, complex_t>::value, void>::type
constrain_and_add_matrix(MatType                           &m,
                         const libMesh::DofMap             &dof_map,
                         std::vector<libMesh::dof_id_type> &dof_indices,
                         SubMatType                        &m_sub) {
    
    Assert2(m_sub.rows() == dof_indices.size(),
            m_sub.rows(), dof_indices.size(),
            "Incompatible matrix rows");
    Assert2(m_sub.cols() == dof_indices.size(),
            m_sub.cols(), dof_indices.size(),
            "Incompatible matrix columns");

    // complex matrix and vector require that the complex and imaginary parts of the matrix
    // and vector be independently constrained
    libMesh::DenseMatrix<real_t>
    m_re(m_sub.rows(), m_sub.cols()),
    m_im(m_sub.rows(), m_sub.cols());

    std::vector<libMesh::dof_id_type>
    dof_indices_im = dof_indices;
    
    // copy the real and imaginary parts
    for (uint_t i=0; i<m_sub.rows(); i++)
        for (uint_t j=0; j<m_sub.cols(); j++) {
            
            m_re(i, j) = m_sub(i, j).real();
            m_im(i, j) = m_sub(i, j).imag();
        }

    // now constraint the vector and matrix. The imaginary part is homogenously constrained
    //
    dof_map.constrain_element_matrix(m_re, dof_indices);
    dof_map.constrain_element_matrix(m_im, dof_indices_im);
    
    
    for (uint_t i=0; i<dof_indices.size(); i++)
        for (uint_t j=0; j<dof_indices.size(); j++)
            add_to_matrix(m, dof_indices[i], dof_indices[j], complex_t(m_re(i,j), m_im(i,j)));
}


inline void
constrain_and_add_matrix_and_vector(libMesh::NumericVector<real_t>    &v,
                                    libMesh::SparseMatrix<real_t>     &m,
                                    const libMesh::DofMap             &dof_map,
                                    std::vector<libMesh::dof_id_type> &dof_indices,
                                    libMesh::DenseVector<real_t>      &v_sub,
                                    libMesh::DenseMatrix<real_t>      &m_sub) {

    dof_map.constrain_element_matrix_and_vector(m_sub, v_sub, dof_indices);
    m.add_matrix(m_sub, dof_indices);
    v.add_vector(v_sub, dof_indices);
}


inline void
constrain_and_add_vector(libMesh::NumericVector<real_t>     &v,
                         const libMesh::DofMap              &dof_map,
                         std::vector<libMesh::dof_id_type>  &dof_indices,
                         libMesh::DenseVector<real_t>       &v_sub) {

    dof_map.constrain_element_vector(v_sub, dof_indices);
    v.add_vector(v_sub, dof_indices);
}


inline void
constrain_and_add_matrix(libMesh::SparseMatrix<real_t>     &m,
                         const libMesh::DofMap             &dof_map,
                         std::vector<libMesh::dof_id_type> &dof_indices,
                         libMesh::DenseMatrix<real_t>      &m_sub) {
    
    dof_map.constrain_element_matrix(m_sub, dof_indices);
    m.add_matrix(m_sub, dof_indices);
}


} // namespace libMesh
} // namespace Assembly
} // namespace Base
} // namespace MAST

#endif // __mast_libmesh_assembly_utility_h__
