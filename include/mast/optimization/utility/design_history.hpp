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

#ifndef __mast_optimization_design_history_h__
#define __mast_optimization_design_history_h__

// C++ includes
#include <sys/stat.h>
#include <string>
#include <fstream>
#include <iomanip>
#include <boost/algorithm/string.hpp>


// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Utility {

/*!
 * Prints the design iteration data to stream \p out. The template parameter \p FuncEvalType is the class that
 * provides the method
 *    - \p FuncEvalType::n_vars() : returns number of design variables
 *    - \p FuncEvalType::n_eq() : returns number of equality constraints
 *    - \p FuncEvalType::n_ineq() : returns number of inequality constraints
 *    - \p FuncEvalType::tol() : returns tolerance for identifying constraint as active
 *
 *  function arguments are
 *    - \p feval : object of type \p FuncEvalType that provides the relevant functions noted above
 *    - \p out : Stream to which the output will be written
 *    - \p iter : current iteration number
 *    - \p x : current design variable vector
 *    - \p obj : current objective function values.
 *    - \p fval : current constraint function values. It is assumed that the equality constraint values are included before
 *    the inequality constraint values.
 */
template <typename FuncEvalType>
inline void
write_dhistory_to_screen(const FuncEvalType          &feval,
                         std::ostream                &out,
                         const uint_t                 iter,
                         const std::vector<real_t>   &x,
                         const real_t                &obj,
                         const std::vector<real_t>   &fval) {
    
    out
    << " *************************** " << std::endl
    << " *** Optimization Output *** " << std::endl
    << " *************************** " << std::endl
    << std::endl
    << "Iter:            "  << std::setw(10) << iter << std::endl
    << "Nvars:           " << std::setw(10) << x.size() << std::endl
    << "Ncons-Equality:  " << std::setw(10) << feval.n_eq() << std::endl
    << "Ncons-Inquality: " << std::setw(10) << feval.n_ineq() << std::endl
    << std::endl
    << "Obj =                  " << std::setw(20) << obj << std::endl
    << std::endl
    << "Vars:            " << std::endl;
    
    for (unsigned int i=0; i<feval.n_vars(); i++)
        out
        << "x     [ " << std::setw(10) << i << " ] = "
        << std::setw(20) << x[i] << std::endl;
    
    if (feval.n_eq()) {
        
        out << std::endl
        << "Equality Constraints: " << std::endl;
        
        for (unsigned int i=0; i<feval.n_eq(); i++)
            out
            << "feq [ " << std::setw(10) << i << " ] = "
            << std::setw(20) << fval[i] << std::endl;
    }
    
    if (feval.n_ineq()) {
        
        out << std::endl
        << "Inequality Constraints: " << std::endl;
        unsigned int
        n_active      = 0,
        n_violated    = 0,
        max_constr_id = 0;
        real_t
        max_constr  = -1.e20;
        
        for (unsigned int i=0; i<feval.n_ineq(); i++) {
            out
            << "fineq [ " << std::setw(10) << i << " ] = "
            << std::setw(20) << fval[i+feval.n_eq()];
            if (fabs(fval[i+feval.n_eq()]) <= feval.tol()) {
                n_active++;
                out << "  ***";
            }
            else if (fval[i+feval.n_eq()] > feval.tol()) {
                n_violated++;
                out << "  +++";
            }
            out  << std::endl;
            
            if (max_constr < fval[i+feval.n_eq()]) {
                max_constr_id = i;
                max_constr    = fval[i+feval.n_eq()];
            }
        }
        
        out << std::endl
        << std::setw(35) << " N Active Constraints: "
        << std::setw(20) << n_active << std::endl
        << std::setw(35) << " N Violated Constraints: "
        << std::setw(20) << n_violated << std::endl
        << std::setw(35) << " Most critical constraint: "
        << std::setw(20) << max_constr << std::endl;
    }
    
    out << std::endl
    << " *************************** " << std::endl;

}


/*!
 * Prints the objective function and constraint values from design iterations stream \p output. Note that design variables are not
 * included. If the design variables are needed then write_dhistory_to_file() must be used instead.
 * The template parameter \p FuncEvalType is the class that provides the method
 *    - \p FuncEvalType::n_vars() : returns number of design variables
 *    - \p FuncEvalType::n_eq() : returns number of equality constraints
 *    - \p FuncEvalType::n_ineq() : returns number of inequality constraints
 *
 *  function arguments are
 *    - \p feval : object of type \p FuncEvalType that provides the relevant functions noted above
 *    - \p output : Stream to which the output will be written
 *    - \p iter : current iteration number
 *    - \p obj : current objective function values.
 *    - \p fval : current constraint function values. It is assumed that the equality constraint values are included before
 *    the inequality constraint values.
 */
template <typename FuncEvalType>
inline void
write_obj_constr_history_to_file(const FuncEvalType          &feval,
                                 std::ostream                &output,
                                 const uint_t                 iter,
                                 const real_t                &obj,
                                 const std::vector<real_t>   &fval) {
    
    // write header for the first iteration
    if (iter == 0) {

        // number of desing variables
        output
        << std::setw(10) << "n_dv" << std::setw(10) << feval.n_vars() << std::endl;
        output
        << std::setw(10) << "n_eq" << std::setw(10) << feval.n_eq() << std::endl;
        output
        << std::setw(10) << "n_ineq" << std::setw(10) << feval.n_ineq() << std::endl;

        output << std::setw(10) << "Iter";
        output << std::setw(20) << "Obj";
        for (unsigned int i=0; i<fval.size(); i++) {
            std::stringstream f; f << "f_" << i;
            output << std::setw(20) << f.str();
        }
        output << std::endl;
    }
    
    output << std::setw(10) << iter;
    output << std::setw(20) << obj;
    for (unsigned int i=0; i < fval.size(); i++)
        output << std::setw(20) << fval[i];
    output << std::endl;
}



/*!
 * Prints the design iteration data to stream \p output in a format that can be read back to initialize a problem from a design history.
 * The template parameter \p FuncEvalType is the class that provides the method
 *    - \p FuncEvalType::n_vars() : returns number of design variables
 *    - \p FuncEvalType::n_eq() : returns number of equality constraints
 *    - \p FuncEvalType::n_ineq() : returns number of inequality constraints
 *  function arguments are
 *    - \p feval : object of type \p FuncEvalType that provides the relevant functions noted above
 *    - \p output : Stream to which the output will be written
 *    - \p iter : current iteration number
 *    - \p x : current design variable vector
 *    - \p obj : current objective function values.
 *    - \p fval : current constraint function values. It is assumed that the equality constraint values are included before
 *    the inequality constraint values.
 *
 *  Following the initial header information, the data is written in a tabular format where each row corresponds to a design
 *  iteration and the columns are arranged with:
 *     -  all design variables,
 *     - all equality constraints,
 *     - objective function,
 *     - all inequality constraints.
 */
template <typename FuncEvalType>
inline void
write_dhistory_to_file(const FuncEvalType          &feval,
                       std::ostream                &output,
                       const uint_t                 iter,
                       const std::vector<real_t>   &x,
                       const real_t                &obj,
                       const std::vector<real_t>   &fval) {
    
    // write header for the first iteration
    if (iter == 0) {

        // number of desing variables
        output
        << std::setw(10) << "n_dv" << std::setw(10) << feval.n_vars() << std::endl;
        output
        << std::setw(10) << "n_eq" << std::setw(10) << feval.n_eq() << std::endl;
        output
        << std::setw(10) << "n_ineq" << std::setw(10) << feval.n_ineq() << std::endl;

        output << std::setw(10) << "Iter";
        for (unsigned int i=0; i < x.size(); i++) {
            std::stringstream x; x << "x_" << i;
            output << std::setw(20) << x.str();
        }
        output << std::setw(20) << "Obj";
        for (unsigned int i=0; i<fval.size(); i++) {
            std::stringstream f; f << "f_" << i;
            output << std::setw(20) << f.str();
        }
        output << std::endl;
    }
    
    output << std::setw(10) << iter;
    for (unsigned int i=0; i < x.size(); i++)
        output << std::setw(20) << x[i];
    output << std::setw(20) << obj;
    for (unsigned int i=0; i < fval.size(); i++)
        output << std::setw(20) << fval[i];
    output << std::endl;
}


/*!
 * Initializes the design variable vector \p dv from \p iter iteration stored in \p file . This assumes that the file was written
 * by the function write_dhistory_to_file(). The template parameter \p FuncEvalType is the class that provides the method
 *    - \p FuncEvalType::n_vars() : returns number of design variables
 *    - \p FuncEvalType::n_eq() : returns number of equality constraints
 *    - \p FuncEvalType::n_ineq() : returns number of inequality constraints
 *
 *  function arguments are
 *    - \p feval : object of type \p FuncEvalType that provides the relevant functions noted above
 *    - \p file : Name of file from which the data will be read
 *    - \p iter : iteration number that will be read from \p file
 *    - \p dv : vector into which the design variables will be initialized
 */
template <typename FuncEvalType>
inline void initialize_dv_from_output_file(const FuncEvalType      &f_eval,
                                           const std::string       &file,
                                           const uint_t             iter,
                                           std::vector<real_t>     &dv) {
    
    
    struct stat stat_info;
    int stat_result = stat(file.c_str(), &stat_info);
    
    if (stat_result != 0)
        Error(false, "File does not exist: " + file);
    
    if (!std::ifstream(file))
        Error(false, "File missing: " + file);
    
    std::ifstream input;
    input.open(file, std::ofstream::in);
    
    
    std::string
    line;
    uint_t
    ndv        = 0,
    nineq      = 0,
    neq        = 0,
    it_num     = 0;
    
    std::vector<std::string> results;
    
    // number of desing variables
    std::getline(input, line);
    boost::trim(line);
    boost::split(results, line, boost::is_any_of(" \t"), boost::token_compress_on);
    Assert0(results[0].compare("n_dv") != 0,
            "Invalid file format: Expected n_dv, found: " + results[0]);
    ndv = stod(results[1]);
    Assert2(ndv == dv.size(), ndv, dv.size(),
            "Design variable vector size incompatible with number of DVs in file");
    
    
    // number of equality constraint
    std::getline(input, line);
    boost::trim(line);
    boost::split(results, line, boost::is_any_of(" \t"), boost::token_compress_on);
    Assert0(results[0].compare("n_eq") != 0,
            "Invalid file format: Expected n_eq, found: " + results[0]);
    neq = stod(results[1]);
    Assert2(neq == f_eval.n_eq(), neq, f_eval.n_eq(),
            "Incompatible number of equality constraints");
    
    
    // number of inequality constriants
    std::getline(input, line);
    boost::trim(line);
    boost::split(results, line, boost::is_any_of(" \t"), boost::token_compress_on);
    Assert0(results[0].compare("n_ineq") != 0,
            "Invalid file format: Expected n_ineq, found: " + results[0]);
    nineq = stod(results[1]);
    Assert2(nineq == f_eval.n_ineq(), nineq, f_eval.n_ineq(),
            "Incompatible number of inequality constraints");

    
    // skip all lines before iter.
    while (!input.eof() && it_num < iter+1) {
        std::getline(input, line);
        it_num++;
    }
    
    // make sure that the iteration number is what we are looking for
    std::getline(input, line);
    boost::trim(line);
    boost::split(results, line, boost::is_any_of(" \t"), boost::token_compress_on);
    
    Assert2(results.size() > ndv+1,
            results.size(), ndv+1,
            "Invalid file format or incomplete data in in put file.");
    
    it_num = stoi(results[0]);
    Assert2(it_num == iter, it_num, iter,
            "Requested iteration number not found");
    
    // make sure that the file has data
    for (unsigned int i=0; i<ndv; i++)
        dv[i] = real_t(stod(results[i+1]));
}



template <typename FuncEvalType>
inline void write_dv_to_file(const FuncEvalType            &feval,
                             const std::string             &file,
                             const uint_t                   iter,
                             const std::vector<complex_t>  &x,
                             const complex_t               &obj,
                             const std::vector<complex_t>  &fval) {
    
    Error(false, "Not implemented for complex type");
}


template <typename FuncEvalType>
inline void initialize_dv_from_output_file(const FuncEvalType      &f_eval,
                                           const std::string       &file,
                                           const uint_t             iter,
                                           std::vector<complex_t>  &dv) {

    Error(false, "Not implemented for complex type");
}


#if MAST_ENABLE_ADOLC == 1
template <typename FuncEvalType>
inline void write_dv_to_file(const FuncEvalType               &feval,
                             const std::string                &file,
                             const uint_t                      iter,
                             const std::vector<adouble_tl_t>  &x,
                             const adouble_tl_t               &obj,
                             const std::vector<adouble_tl_t>  &fval) {
    
    Error(false, "Not implemented for adouble type");
}


template <typename FuncEvalType>
inline void initialize_dv_from_output_file(const FuncEvalType         &f_eval,
                                           const std::string          &file,
                                           const uint_t                iter,
                                           std::vector<adouble_tl_t>  &dv) {
    
    Error(false, "Not implemented for adouble type");
}
#endif

} // namespace Utility
} // namespace Optimization
} // namespace MAST


#endif // __mast_optimization_design_history_h__
