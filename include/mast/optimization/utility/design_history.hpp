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
#include <boost/algorithm/string.hpp>


// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Utility {

template <typename FuncEvalType>
inline void write_dv_to_file(const FuncEvalType          &feval,
                             const std::string           &file,
                             const uint_t                 iter,
                             const std::vector<real_t>   &x,
                             const real_t                &obj,
                             const std::vector<real_t>   &fval) {
    
    std::ofstream output;

    // create a new file for first iteration. Otherwise append
    if (iter == 0)
        output.open(file.c_str(), std::ofstream::out);
    else
        output.open(file.c_str(), std::ofstream::app);
    
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
    
    
    output.close();
}


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
