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

#ifndef __mast_slepc_eigen_solver_h__
#define __mast_slepc_eigen_solver_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// SLEPc includes
#include <slepc.h>


namespace MAST {
namespace Solvers {
namespace SLEPcWrapper {

class GeneralizedHermitianEigenSolver {
    
public:
    
    GeneralizedHermitianEigenSolver(EigenproblemType type):
    _initialized(false),
    _n(0),
    _n_converged(0),
    _type(type) { }
    
    virtual ~GeneralizedHermitianEigenSolver() {
        
        if (_initialized)
            EPSDestroy(&_eps);
    }
    
    /// @returns the number of converged eigen pairs
    inline unsigned int n_converged() {
        
        Assert0(_initialized, "solver not initialized");
        
        return _n_converged;
    }
    
    
    /// this method returns the eigen value
    inline real_t eig(uint_t i) {
        
        Assert0(_initialized, "solver not initialized");
        Assert2(eigen_index < _n_converged,
                eigen_index, _n_converged,
                "Eigenvalue index index must be less than n_converged");
        
        real_t
        re = 0.,
        im = 0.;
        EPSGetEigenvalue(_eps, eigen_index, &re, &im);
        
        // assuming that im == 0 for Hermitian problem
        return r;
    }
    
    
    /// this method returns the eigen pair
    inline void getEigenPair(uint_t   i,
                             real_t  &eig,
                             Vec      x) {
        
        Assert0(_initialized);
        Assert2(eigen_index < _n_converged,
                eigen_index, _n_converged,
                "Eigenvalue index index must be less than n_converged");

        
        VecZeroEntries(x);

        Vec vi;
        VecCopy(x, &vi, PETSC_NULL);
        
        eig = 0.;
        
        real_t
        im = 0.;
        
        EPSGetEigenpair(_eps, i, &eig, &im, x, vi);

        // assuming that the imaginary componet of vector will be zero for
        // generalized Hermitian problem
        VecDestroy(&vi);
    }
    
    /// method for eigenvalue problems  \f$ A x = \lambda B x \f$
    inline void solve(Mat    A_mat,
                      Mat    B_mat,
                      uint_t nev,
                      EigenSpectrumType spectrum,
                      bool computeEigenvectors) {
        
        assert(!_initialized);
        
        assert(A_mat.rows() == A_mat.cols());
        if (B_mat) {
            assert(_type == GENERALIZED_HERMITIAN ||
                   _type == GENERALIZED_NON_HERMITIAN);
            assert(A_mat.rows() == B_mat->rows());
            assert(B_mat->rows() == B_mat->cols());
        }
        
        _n = (int) A_mat.rows();
        _init_mat(A_mat, _A);
        if (B_mat)
            _init_mat(*B_mat, _B);
        
        EPSCreate(PETSC_COMM_SELF, &_eps);
        if (!B_mat) {
            EPSSetOperators(_eps, _A, PETSC_NULL);
            if (_type == HERMITIAN)
                EPSSetProblemType(_eps, EPS_HEP);
            else if (_type == NON_HERMITIAN)
                EPSSetProblemType(_eps, EPS_NHEP);
            else
                assert(false);
        }
        else {
            EPSSetOperators(_eps, _A, _B);
            if (_type == GENERALIZED_HERMITIAN)
                EPSSetProblemType(_eps, EPS_GHEP);
            else if (_type == GENERALIZED_NON_HERMITIAN)
                EPSSetProblemType(_eps, EPS_GNHEP);
            else
                assert(false);
        }
        
        if (spectrum == LARGEST_MAGNITUDE)
            EPSSetWhichEigenpairs(_eps, EPS_LARGEST_MAGNITUDE);
        else if (spectrum == SMALLEST_MAGNITUDE)
            EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_MAGNITUDE);
        else if (spectrum == LARGEST_IMAGINARY)
            EPSSetWhichEigenpairs(_eps, EPS_LARGEST_IMAGINARY);
        else if (spectrum == SMALLEST_IMAGINARY)
            EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_IMAGINARY);
        else if (spectrum == LARGEST_REAL)
            EPSSetWhichEigenpairs(_eps, EPS_LARGEST_REAL);
        else if (spectrum == SMALLEST_REAL)
            EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_REAL);
        else
            assert (false);
        
        EPSSetDimensions(_eps, nev, PETSC_DEFAULT, PETSC_DEFAULT);
        EPSSetFromOptions(_eps);
        EPSSolve(_eps);
        EPSGetConverged(_eps, &_n_converged);
        
        _initialized = true;
    }
    
    inline void printResidualForEigenPairs() {
        
        real_t error;
        
        for (uint_t i=0; i<_n_converged; i++) {
            
            EPSComputeError(_eps, i, EPS_ERROR_RELATIVE, &error);
            std::cout
            << std::setw(10) << i
            << std::setw(30) << this->getEigenValue(i)
            << std::setw(30) << error << std::endl;
        }
        
    }

protected:
    
    void _init_mat(const RealMatrixX& A, Mat& mat);
    bool _initialized;
    int  _n, _n_converged;
    EigenproblemType _type;
    Mat _A, _B;
    EPS _eps;
};

}  // namespace SLEPcWrapper
}  // namespace Solvers
}  // namespace MAST

#endif // __mast__slepc_eigen_solver__

