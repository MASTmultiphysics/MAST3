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

#ifndef __mast__fem_operator_matrix_h__
#define __mast__fem_operator_matrix_h__

// C++ includes
#include <vector>
#include <iomanip>

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>



namespace MAST {
namespace Numerics {

template <typename ScalarType>
class FEMOperatorMatrix
{
public:
    FEMOperatorMatrix();
    
    
    virtual ~FEMOperatorMatrix();
    
    
    /*!
     *   clears the data structures
     */
    void clear();
    
    
    uint_t m() const {return _n_interpolated_vars;}
    
    uint_t n() const {return _n_discrete_vars*_n_dofs_per_var;}
    
    void print(std::ostream& o);
    
    /*!
     *   this initializes the operator for number of rows and variables, assuming
     *   that all variables has the same number of dofs. This is typically the case
     *   for structural strain operator matrices. Note that when this method is used
     *   the user must set the matrix entries by calling set_shape_functions
     */
    void reinit(uint_t n_interpolated_vars,
                uint_t n_discrete_vars,
                uint_t n_discrete_dofs_per_var);
    
    /*!
     *   sets the shape function values for the block corresponding to
     *   \p interpolated_var and \p discrete_var. This means that the row
     *   \p interpolated_var, the value in columns
     *   \p discrete_vars*n_discrete_dofs_per_var - (discrete_vars+1)*n_discrete_dofs_per_var-1)
     *    will be set equal to \p shape_func .
     */
    template <typename VecType>
    void set_shape_function(uint_t interpolated_var,
                            uint_t discrete_var,
                            const VecType& shape_func);
    
    /*!
     *   this initializes all variables to use the same interpolation function.
     *   It is assumed that the number of discrete vars is same as the number of
     *   interpolated vars. This is typically the case for fluid elements and
     *   for structural element inertial matrix calculations
     */
    template <typename VecType>
    void reinit(uint_t n_interpolated_vars,
                const VecType& shape_func);
    
    /*!
     *   res = [this] * v
     */
    template <typename T>
    void vector_mult(T& res, const T& v) const;
    
    
    /*!
     *   res = v^T * [this]
     */
    template <typename T1, typename T2>
    void vector_mult_transpose(T1& res, const T2& v) const;
    
    
    /*!
     *   [R] = [this] * [M]
     */
    template <typename T>
    void right_multiply(T& r, const T& m) const;
    
    
    /*!
     *   [R] = [this]^T * [M]
     */
    template <typename T>
    void right_multiply_transpose(T& r, const T& m) const;
    
    
    /*!
     *   [R] = [this]^T * [M]
     */
    template <typename T>
    void right_multiply_transpose(T& r,
                                  const MAST::Numerics::FEMOperatorMatrix<ScalarType>& m) const;
    
    
    /*!
     *   [R] = [M] * [this]
     */
    template <typename T1, typename T2>
    void left_multiply(T1& r, const T2& m) const;
    
    
    /*!
     *   [R] = [M] * [this]^T
     */
    template <typename T>
    void left_multiply_transpose(T& r, const T& m) const;
    
    
protected:
    
    /*!
     *    number of rows of the operator
     */
    uint_t _n_interpolated_vars;
    
    /*!
     *    number of discrete variables in the system
     */
    uint_t _n_discrete_vars;
    
    /*!
     *    number of dofs for each variable
     */
    uint_t _n_dofs_per_var;
    
    /*!
     *    stores the shape function values that defines the coupling
     *    of i_th interpolated var and j_th discrete var. Stored in
     *    column major format. nullptr, if values are zero, otherwise the
     *    value is set in the vector.
     */
    std::vector<ScalarType*>  _var_shape_functions;
};

} // namespace Numerics
} // namespace MAST

template <typename ScalarType>
MAST::Numerics::FEMOperatorMatrix<ScalarType>::FEMOperatorMatrix():
_n_interpolated_vars(0),
_n_discrete_vars(0),
_n_dofs_per_var(0)
{
    
}


template <typename ScalarType>
MAST::Numerics::FEMOperatorMatrix<ScalarType>::~FEMOperatorMatrix()
{
    this->clear();
}


template <typename ScalarType>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::print(std::ostream& o) {
    
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) {// row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) // check if this is non-nullptr
                for (uint_t k=0; k<_n_dofs_per_var; k++)
                    o << std::setw(15) << _var_shape_functions[index][k];
            else
                for (uint_t k=0; k<_n_dofs_per_var; k++)
                    o << std::setw(15) << 0.;
        }
        o << std::endl;
    }
}


template <typename ScalarType>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::clear() {
    
    _n_interpolated_vars = 0;
    _n_discrete_vars     = 0;
    _n_dofs_per_var      = 0;
    
    // iterate over the shape function entries and delete the non-nullptr values
    typename std::vector<ScalarType*>::iterator
    it  = _var_shape_functions.begin(),
    end = _var_shape_functions.end();
    
    for ( ; it!=end; it++)
        if ( *it != nullptr)
            delete *it;
    
    _var_shape_functions.clear();
}




template <typename ScalarType>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
reinit(uint_t n_interpolated_vars,
       uint_t n_discrete_vars,
       uint_t n_discrete_dofs_per_var) {
    
    this->clear();
    _n_interpolated_vars = n_interpolated_vars;
    _n_discrete_vars = n_discrete_vars;
    _n_dofs_per_var = n_discrete_dofs_per_var;
    _var_shape_functions.resize(_n_interpolated_vars*_n_discrete_vars, nullptr);
}



template <typename ScalarType>
template <typename VecType>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
set_shape_function(uint_t interpolated_var,
                   uint_t discrete_var,
                   const VecType& shape_func) {
    
    // make sure that reinit has been called.
    Assert0(_var_shape_functions.size(), "Object not initialized");
    
    // also make sure that the specified indices are within bounds
    Assert2(interpolated_var < _n_interpolated_vars,
            interpolated_var, _n_interpolated_vars,
            "Invalid interpolation variable index");
    Assert2(discrete_var < _n_discrete_vars,
            discrete_var, _n_discrete_vars,
            "Invalid discrete variable index");
    Assert2(shape_func.size() == _n_dofs_per_var,
            shape_func.size(), _n_dofs_per_var,
            "Invalid basis function vector size.");
    
    ScalarType* vec =
    _var_shape_functions[discrete_var*_n_interpolated_vars+interpolated_var];
    
    if (!vec) {
        
        vec = new ScalarType[shape_func.size()];
        _var_shape_functions[discrete_var*_n_interpolated_vars+interpolated_var] = vec;
    }
    
    for (uint_t i=0; i<_n_dofs_per_var; i++)
        vec[i] = shape_func(i);
}



//template <typename ScalarType>
//template <typename VecType>
//inline
//void
//MAST::Numerics::FEMOperatorMatrix<ScalarType>::
//set_shape_function(uint_t interpolated_var,
//                   uint_t discrete_var,
//                   const ScalarType v,
//                   const VecType& shape_func) {
//    
//    // make sure that reinit has been called.
//    Assert0(_var_shape_functions.size(), "Object not initialized");
//    
//    // also make sure that the specified indices are within bounds
//    Assert2(interpolated_var < _n_interpolated_vars,
//            interpolated_var, _n_interpolated_vars,
//            "Invalid interpolation variable index");
//    Assert2(discrete_var < _n_discrete_vars,
//            discrete_var, _n_discrete_vars,
//            "Invalid discrete variable index");
//    Assert2(shape_func.size() == _n_dofs_per_var,
//            shape_func.size(), _n_dofs_per_var,
//            "Invalid basis function vector size.");
//    
//    ScalarType* vec =
//    _var_shape_functions[discrete_var*_n_interpolated_vars+interpolated_var];
//    
//    if (!vec) {
//        
//        vec = new ScalarType[shape_func.size()];
//        _var_shape_functions[discrete_var*_n_interpolated_vars+interpolated_var] = vec;
//    }
//    
//    for (uint_t i=0; i<_n_dofs_per_var; i++)
//        vec[i] = v*shape_func(i);
//}




template <typename ScalarType>
template <typename VecType>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
reinit(uint_t n_vars,
       const VecType& shape_func) {
    
    this->clear();
    
    _n_interpolated_vars = n_vars;
    _n_discrete_vars = n_vars;
    _n_dofs_per_var = (uint_t)shape_func.size();
    _var_shape_functions.resize(n_vars*n_vars, nullptr);
    
    for (uint_t i=0; i<n_vars; i++)
    {
        ScalarType *vec = new ScalarType[_n_dofs_per_var];
        for (uint_t i=0; i<_n_dofs_per_var; i++)
            vec[i] = shape_func(i);
        _var_shape_functions[i*n_vars+i] = vec;
    }
}



template <typename ScalarType>
template <typename T>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
vector_mult(T& res, const T& v) const {
    
    Assert2(res.size() == _n_interpolated_vars,
            res.size(), _n_interpolated_vars,
            "Incompatible vector size.");
    Assert2(v.size() == n(),
            v.size(), n(),
            "Incompatible vector size");
    
    res.setZero();
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) // row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) // check if this is non-nullptr
                for (uint_t k=0; k<_n_dofs_per_var; k++)
                    res(i) +=
                    _var_shape_functions[index][k] * v(j*_n_dofs_per_var+k);
        }
}

template <typename ScalarType>
template <typename T1, typename T2>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
vector_mult_transpose(T1& res, const T2& v) const {
    
    Assert2(res.size() == n(),
            res.size(), n(),
            "Incompatible vector size");
    Assert2(v.size() == _n_interpolated_vars,
            v.size(), _n_interpolated_vars,
            "Incompatible vector size");
    
    res.setZero(res.size());
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) // row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) // check if this is non-nullptr
                for (uint_t k=0; k<_n_dofs_per_var; k++)
                    res(j*_n_dofs_per_var+k) +=
                    _var_shape_functions[index][k] * v(i);
        }
}



template <typename ScalarType>
template <typename T>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
right_multiply(T& r, const T& m) const {
    
    Assert2(r.rows() == _n_interpolated_vars,
            r.rows(), _n_interpolated_vars,
            "Incompatible matrix row dimension");
    Assert2(r.cols() == m.cols(),
            r.cols(), m.cols(),
            "Incompatible matrix column dimension");
    Assert2(m.rows() == n(),
            m.rows(), n(),
            "Incompatible matrix row dimension");

    r.setZero();
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) // row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column of operator
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) { // check if this is non-nullptr
                for (uint_t l=0; l<m.cols(); l++) // column of matrix
                    for (uint_t k=0; k<_n_dofs_per_var; k++)
                        r(i,l) +=
                        _var_shape_functions[index][k] * m(j*_n_dofs_per_var+k,l);
            }
        }
}




template <typename ScalarType>
template <typename T>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
right_multiply_transpose(T& r, const T& m) const {
    
    Assert2(r.rows() == n(),
            r.rows(), n(),
            "Incompatible matrix row dimension");
    Assert2(r.cols() == m.cols(),
            r.cols(), m.cols(),
            "Incompatible matrix column dimension");
    Assert2(m.rows() == _n_interpolated_vars,
            m.rows(), _n_interpolated_vars,
            "Incompatible matrix row dimension");
    
    r.setZero(r.rows(), r.cols());
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) // row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column of operator
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) { // check if this is non-nullptr
                for (uint_t l=0; l<m.cols(); l++) // column of matrix
                    for (uint_t k=0; k<_n_dofs_per_var; k++)
                        r(j*_n_dofs_per_var+k,l) +=
                        _var_shape_functions[index][k] * m(i,l);
            }
        }
}



template <typename ScalarType>
template <typename T>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
right_multiply_transpose(T& r, const MAST::Numerics::FEMOperatorMatrix<ScalarType>& m) const {
    
    Assert2(r.rows() == n(),
            r.rows(), n(),
            "Incompatible matrix row dimension");
    Assert2(r.cols() == m.n(),
            r.cols(), m.n(),
            "Incompatible matrix column dimension");
    Assert2(_n_interpolated_vars == m._n_interpolated_vars,
            _n_interpolated_vars, m._n_interpolated_vars,
            "Incompatible number of variables");
    
    r.setZero();
    uint_t index_i, index_j = 0;
    
    for (uint_t i=0; i<_n_discrete_vars; i++) // row of result
        for (uint_t j=0; j<m._n_discrete_vars; j++) // column of result
            for (uint_t k=0; k<_n_interpolated_vars; k++) {
                index_i = i*_n_interpolated_vars+k;
                index_j = j*m._n_interpolated_vars+k;
                if (_var_shape_functions[index_i] &&
                    m._var_shape_functions[index_j]) { // if shape function exists for both
                    const ScalarType
                    *n1 = _var_shape_functions[index_i],
                    *n2 = m._var_shape_functions[index_j];
                    for (uint_t i_n1=0; i_n1<_n_interpolated_vars; i_n1++)
                        for (uint_t i_n2=0; i_n2<m._n_interpolated_vars; i_n2++)
                            r (i*_n_dofs_per_var+i_n1,
                               j*m._n_dofs_per_var+i_n2) += n1[i_n1] * n2[i_n2];
                }
            }
}




template <typename ScalarType>
template <typename T1, typename T2>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
left_multiply(T1& r, const T2& m) const {
    
    Assert2(r.rows() == m.rows(),
            r.rows(), m.rows(),
            "Incompatible matrix rows");
    Assert2(r.cols() == n(),
            r.cols(), n(),
            "Incompatible matrix columns");
    Assert2(m.cols() == _n_interpolated_vars,
            m.cols(), _n_interpolated_vars,
            "Incompatible matrix columns");
    
    r.setZero(r.rows(), r.cols());
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) // row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column of operator
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) { // check if this is non-nullptr
                for (uint_t l=0; l<m.rows(); l++) // rows of matrix
                    for (uint_t k=0; k<_n_dofs_per_var; k++)
                        r(l,j*_n_dofs_per_var+k) +=
                        _var_shape_functions[index][k] * m(l,i);
            }
        }
}



template <typename ScalarType>
template <typename T>
inline
void
MAST::Numerics::FEMOperatorMatrix<ScalarType>::
left_multiply_transpose(T& r, const T& m) const {
    
    Assert2(r.rows() == m.rows(),
            r.rows(), m.rows(),
            "Incompatible matrix rows");
    Assert2(r.cols() == _n_interpolated_vars,
            r.cols(), _n_interpolated_vars,
            "Incompatible matrix columns");
    Assert2(m.cols() == n(),
            m.cols(), n(),
            "Incompatible matrix columns");
    
    r.setZero();
    uint_t index = 0;
    
    for (uint_t i=0; i<_n_interpolated_vars; i++) // row
        for (uint_t j=0; j<_n_discrete_vars; j++) { // column of operator
            index = j*_n_interpolated_vars+i;
            if (_var_shape_functions[index]) { // check if this is non-nullptr
                for (uint_t l=0; l<m.rows(); l++) // column of matrix
                    for (uint_t k=0; k<_n_dofs_per_var; k++)
                        r(l,i) +=
                        _var_shape_functions[index][k] * m(l,j*_n_dofs_per_var+k);
            }
        }
}



#endif // __mast__fem_operator_matrix_h__

