
#ifndef __mast_accessor_h__
#ifndef __mast_accessor_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Base {
namespace Assembly {
namespace libMeshWrapper {

template <typename ScalarType, typename VecType>
class Accessor {

public:

    Assessor():
    _vec      (nullptr),
    _mat      (nullptr)
    { }

    inline void set_vec(libMesh::NumericVector<ScalarType>& vec) { _vec = &vec;}
    inline void set_matrix(libMesh::SparseMatrix& mat) { _mat = &mat;}
    inline ScalarType operator() (uint_t i) const { return (*_vec)(i);}

private:
    
    libMesh::System                    *_sys;
    libMesh::NumericVector<ScalarType> *_vec;
    libMesh::SparseMatrix<ScalarType>  *_mat;
    std::vector<libMesh::dof_id_type>   _dof_ids;
};

} // namespace libMesh
} // namespace Assembly
} // namespace Base
} // namespace MAST

#ifndef __mast_accessor_h__

