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

#ifndef __mast_slepc_constrained_generalized_hermitian_eigen_solver_h__
#define __mast_slepc_constrained_generalized_hermitian_eigen_solver_h__

// C++ includes
#include <iomanip>

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// Eigen includes
#include <Eigen/Eigenvalues>


namespace MAST {
namespace Solvers {
namespace EigenWrapper {

template <typename ScalarType,
          typename EigenSolverType,
          typename MatType,
          typename VectorType>
class ConstrainedGeneralizedHermitianEigenSolver {
    
public:

    using scalar_t = ScalarType;
    
    /*!
     * \p dofs is the vector of unconstrained degrees of freedom on the local rank.
     */
    ConstrainedGeneralizedHermitianEigenSolver(const std::vector<PetscInt> &dofs):
    _dofs         (dofs),
    _initialized  (false),
    _n            (0),
    _type         (type),
    _A_sub        (nullptr),
    _B_sub        (nullptr) { }
    
    virtual ~ConstrainedGeneralizedHermitianEigenSolver() { }
        
    /// this method returns the eigen value
    inline ScalarType eig(uint_t i) {
        
        Assert0(_initialized, "solver not initialized");
        Assert2(i < _n,
                i, _n,
                "Eigenvalue index must be less than matrix size");
        
        return _solver.eigenvalues()(i);
    }
    
    
    /*!
     * this method returns the eigen pair. The \p i th eigenvalue is returned in \p eig and the
     * corresponding eigenvector of the origin problem size (including both constrained and
     * unconstrained dofs) is returned in \p x.
     * The index \p i must be less than the total number of converged eigenvalues.
     */
    inline void getEigenVector(uint_t        i,
                               VectorType   &x) {
        
        Assert0(_initialized, "solver not initialized");
        Assert2(i < _n,
                i, _n,
                "Eigenvalue index must be less than matrix size");

        Assert2(x.size() == _n,
                x.size(), _n,
                "Vector must have dimension of original matrix");
        

        x.setZero();
        
        for (uint_t j=0; j<_dofs.size(); j++) {
            x(_dofs(j)) = solver.eigenvectors().col(i)(j);
        }
    }
    
    /*!
     *  method for eigenvalue problems  \f$ A x = \lambda B x \f$
     */
    inline void solve(MatType          &A_mat,
                      MatType          &B_mat,
                      bool              computeEigenvectors) {
        
        Assert0(!_initialized, "solver already initialized");
        
        Assert2(A_mat.rows() == A_mat.cols(),
                A_mat.rows(), A_mat.cols(),
                "Matrix must be square");
        Assert2(B_mat.rows() == B_mat.cols(),
                B_mat.rows(), B_mat.cols(),
                "Matrix must be square");
        Assert2(A_mat.rows() == B_mat.rows(),
                A_mat.rows(), B_mat.rows(),
                "Matrices must have same dimensions");

        _n = A_mat.rows();
        
        // initialize the matrices
        _init_sub_matrices(A_mat, B_mat);
        
        // create the solver context
        if (computeEigenvectors)
            _solver.compute(_A_sub, _B_sub, Eigen::ComputeEigenvectors|Eigen::Ax_lBx);
        else
            _solver.compute(_A_sub, _B_sub, Eigen::EigenvaluesOnly);
        
        _initialized = true;
    }
    

    /*!
     *  compute the sensitivity of \p i th eigenvalue
     *  \f[  \frac{d \lambda_i}{d p} = \frac{ x^T \left(\frac{\partial A}{\partial p} -
     *   \lambda_i \frac{\partial B}{\partial p}\right) x}{x_i^T B x_i } \f]
     *   Dimension of the matrices \p B, \p A_sens and \p B_sens is equal to
     *   the original problem including both the constrained and unconstrained degrees-of-freedom.
     */
    inline scalar_t sensitivity_solve(MatType      &B,
                                      MatType      &A_sens,
                                      MatType      &B_sens,
                                      uint_t       i) {

        VecType
        v1 = VecType::Zeros(_n);
        
        this->get_eigenvector(i, v1);
        
        scalar_t
        eig   = this->get_eigenvalue(i);

        eig =
        v1.dot( (A_sens * v1) - (eig * B_sens * v1) )/ // numerator
        v1.dot(B*v1); // denominator
        
        return eig;
    }
    

protected:

    inline void
    _init_sub_matrices(const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> &A,
                       const Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic> &B,
                       Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>       &A_sub,
                       Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>       &B_sub) {
        
        Assert0(!_initialized, "solver already initialized");
        
        A_sub.setZero(_dofs.size(), _dofs.size());
        B_sub.setZero(_dofs.size(), _dofs.size());
        
        for (uint_t i=0; i<_dofs.size(); i++) {
            for (uint_t j=0; j<_dofs.size(); j++) {
                A_sub(i, j) = A(_dofs[i], _dof[j]);
                B_sub(i, j) = B(_dofs[i], _dof[j]);
            }
        }
    }
    
    
    const std::vector<PetscInt>  &_dofs;
    bool                          _initialized;
    int                           _n;
    MatType                       _A_sub;
    MatType                       _B_sub;
    EigenSolverType               _solver;
};

}  // namespace EigenWrapper
}  // namespace Solvers
}  // namespace MAST

#endif // __mast_slepc_constrained_generalized_hermitian_eigen_solver_h__

