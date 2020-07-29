
#ifndef __mast_optimization_design_history_h__
#define __mast_optimization_design_history_h__

// C++ includes
#include <sys/stat.h>
#include <string>
#include <boost/algorithm/string.hpp>


// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Optimization {
namespace Utility {

inline void write_dv_to_file(const std::string     &file,
                             const uint_t           iter,
                             std::vector<real_t>  &dv) {
    
}


template <typename FuncEvalType, typename ScalarType>
inline void initialize_dv_from_output_file(const FuncEvalType      &f_eval,
                                           const std::string       &file,
                                           const uint_t             iter,
                                           std::vector<ScalarType> &dv) {
    
    
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
        dv[i] = ScalarType(stod(results[i+1]));
}


} // namespace Utility
} // namespace Optimization
} // namespace MAST


#endif // __mast_optimization_design_history_h__
