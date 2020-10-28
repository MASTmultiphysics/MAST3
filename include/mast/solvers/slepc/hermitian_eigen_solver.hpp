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

class HermitianEigenSolver {
    
public:
    
    HermitianEigenSolver(EPSProblemType type):
    _initialized(false),
    _n(0),
    _n_converged(0),
    _type(type) { }
    
    virtual ~HermitianEigenSolver() {
        
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
        Assert2(i < _n_converged,
                i, _n_converged,
                "Eigenvalue index must be less than n_converged");
        
        real_t
        re = 0.,
        im = 0.;
        EPSGetEigenvalue(_eps, i, &re, &im);
        
        // assuming that im == 0 for Hermitian problem
        return re;
    }
    
    
    /// this method returns the eigen pair
    inline void getEigenPair(uint_t   i,
                             real_t  &eig,
                             Vec      x) {
        
        Assert0(_initialized, "solver not initialized");
        Assert2(i < _n_converged,
                i, _n_converged,
                "Eigenvalue index must be less than n_converged");

        
        VecZeroEntries(x);

        Vec vi;
        VecCopy(x, vi);
        
        eig = 0.;
        
        real_t
        im = 0.;
        
        EPSGetEigenpair(_eps, i, &eig, &im, x, vi);

        // assuming that the imaginary componet of vector will be zero for
        // generalized Hermitian problem
        VecDestroy(&vi);
    }
    
    /// method for eigenvalue problems  \f$ A x = \lambda B x \f$
    inline void solve(Mat               A_mat,
                      Mat              *B_mat,
                      uint_t            nev,
                      EPSWhich          spectrum,
                      bool              computeEigenvectors) {
        
        Assert0(_initialized, "solver not initialized");

        PetscInt
        m = 0,
        n = 0;
        
        MatGetSize(A_mat, &m, &n);
        
        Assert2(m == n, m, n, "Matrix must be square");
        
        if (B_mat) {
            
            PetscInt
            m2 = 0,
            n2 = 0;
            
            MatGetSize(*B_mat, &m, &n);

            Assert0(_type == EPS_GHEP,
                    "Eigensolver type must be Generalized Hermitian");
            Assert2(m==m2 && n==n2, m2, n2, "A and B must be same size");
        }
        
        EPSCreate(PETSC_COMM_SELF, &_eps);
        
        if (!B_mat) {

            EPSSetOperators(_eps, A_mat, PETSC_NULL);
            
            if (_type == EPS_HEP)
                EPSSetProblemType(_eps, EPS_HEP);
            else if (_type == EPS_NHEP)
                EPSSetProblemType(_eps, EPS_NHEP);
            else
                Assert0(false, "Invalid EPS type");
        }
        else {

            EPSSetOperators(_eps, A_mat, *B_mat);
            
            if (_type == EPS_GHEP)
                EPSSetProblemType(_eps, EPS_GHEP);
            else if (_type == EPS_GNHEP)
                EPSSetProblemType(_eps, EPS_GNHEP);
            else
                Assert0(false, "Invalid EPS type");
        }
        
        if (spectrum == EPS_LARGEST_MAGNITUDE)
            EPSSetWhichEigenpairs(_eps, EPS_LARGEST_MAGNITUDE);
        else if (spectrum == EPS_SMALLEST_MAGNITUDE)
            EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_MAGNITUDE);
        else if (spectrum == EPS_LARGEST_IMAGINARY)
            EPSSetWhichEigenpairs(_eps, EPS_LARGEST_IMAGINARY);
        else if (spectrum == EPS_SMALLEST_IMAGINARY)
            EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_IMAGINARY);
        else if (spectrum == EPS_LARGEST_REAL)
            EPSSetWhichEigenpairs(_eps, EPS_LARGEST_REAL);
        else if (spectrum == EPS_SMALLEST_REAL)
            EPSSetWhichEigenpairs(_eps, EPS_SMALLEST_REAL);
        else
            Assert0(false, "Invalid spectrum type");

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
            << std::setw(30) << this->eig(i)
            << std::setw(30) << error << std::endl;
        }
        
    }

protected:
    
    bool              _initialized;
    int               _n, _n_converged;
    EPSProblemType    _type;
    EPS               _eps;
};

}  // namespace SLEPcWrapper
}  // namespace Solvers
}  // namespace MAST

#endif // __mast__slepc_eigen_solver__

