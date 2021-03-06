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

#ifndef __mast_optimization_discrete_aggregation_h__
#define __mast_optimization_discrete_aggregation_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/numerics/utility.hpp>

// libMesh includes
#include <libmesh/parallel.h>

namespace MAST {
namespace Optimization {
namespace Aggregation {

/*!
 * computes aggregated minimum of values specified in vector \p vec. The aggregation constant is \p p.
 * The aggregation expression used is
 * \f[ v_{agg} = v_{min} - \frac{1}{p} \log \left( \sum_i \exp (-p (v_i - v_{min}))  \right) \f],
 * where, \f$ v_{min} \f$ is the minimum value out of all values in \p vec.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_minimum(const libMesh::Parallel::Communicator *comm,
                  const std::vector<ScalarType>         &vec,
                  const real_t                           p) {
    
    ScalarType
    v      = 0.,
    v_min  = 0.;
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);
    
    if (comm) MAST::Numerics::Utility::comm_min(*comm, v_min);
    
    for (uint_t i=0; i<vec.size(); i++) {
        
        v += exp(-p * (vec[i] - v_min));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, v);

    v = v_min - log(v) / p;
    
    return v;
}


/*!
 * computes sensitivity of aggregated minimum of values specified in vector \p vec with respect to \p i th value.
 * The aggregation constant is \p p.
 * The aggregation expression used is
 * \f[ \frac{d v_{agg}}{d v_j} =  \frac{ \exp (-p (v_j - v_{min})) }{  \sum_i \exp (-p (v_i - v_{min})) }  \f],
 * where, \f$ v_{min} \f$ is the minimum value out of all values in \p vec.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_minimum_sensitivity(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const uint_t                           i,
                              const real_t                           p) {
    
    ScalarType
    v      = 0.,
    v_min  = 0.;
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);

    if (comm) MAST::Numerics::Utility::comm_min(*comm, v_min);

    for (uint_t i=0; i<vec.size(); i++) {
        
        v  += exp(-p * (vec[i] - v_min));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, v);

    v = exp(-p * (vec[i] - v_min)) / v;
    
    return v;
}



/*!
 * computes sensitivity of aggregated minimum of values specified in vector \p vec with respect to
 * parameter \f$ \alpha \f$.  The aggregation constant is \p p. The sensitivity of values with respect to parameter is
 * provided in \p dvec.
 * The sensitivity expression used is
 * \f[ \frac{d v_{agg}}{d p} =  \frac{ \sum_j \exp (-p (v_j - v_{min})) \frac{dv_j}{d\alpha} }{  \sum_i \exp (-p (v_i - v_{min})) }  \f],
 * where, \f$ v_{min} \f$ is the minimum value out of all values in \p vec.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_minimum_sensitivity(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const std::vector<ScalarType>         &dvec,
                              const real_t                           p) {
    
    ScalarType
    dv     = 0.,
    v      = 0.,
    v_min  = 0.;
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);

    if (comm) MAST::Numerics::Utility::comm_min(*comm, v_min);

    for (uint_t i=0; i<vec.size(); i++) {
        
        dv += exp(-p * (vec[i] - v_min)) * dvec[i];
        v  += exp(-p * (vec[i] - v_min));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, dv);
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, v);

    v = dv / v;
    
    return v;
}



/*!
 * Computes the denominator of the sensitivity of aggregated minimum function for use in later sensitivity computations.
 * The value is  \f$  \sum_i \exp (-p (v_i - v_{min}))   \f$,
 * where, \f$ v_{min} \f$ is the minimum value out of all values in \p vec. The denomimator is returned in
 * \p denom and the minimum value is returned in \p vmin.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
void
aggregate_minimum_denominator(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const real_t                           p,
                              ScalarType                            &denom,
                              ScalarType                            &v_min) {
    
    denom  = 0.;
    
    v_min = MAST::Numerics::Utility::real_minimum(vec);

    if (comm) MAST::Numerics::Utility::comm_min(*comm, v_min);

    for (uint_t i=0; i<vec.size(); i++) {
        
        denom  += exp(-p * (vec[i] - v_min));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, denom);
}



/*!
 * computes sensitivity of aggregated minimum of values specified in vector \p vec with respect to \p i th value.
 * The aggregation constant is \p p.
 * The aggregation expression used is
 * \f[ \frac{d v_{agg}}{d v_j} =  \frac{ \exp (-p (v_j - v_{min})) }{  \sum_i \exp (-p (v_i - v_{min})) }  \f],
 * where, \f$ v_{min} \f$ is the minimum value out of all values in \p vec. This method differs from the other
 * in that the user provides the cached denominator of the sensitivity  and the minimum value in \p denom and
 * \p v_min.
 */
template <typename ScalarType>
ScalarType
aggregate_minimum_sensitivity(const std::vector<ScalarType>   &vec,
                              const uint_t                     i,
                              const real_t                     p,
                              const ScalarType                &denom,
                              const ScalarType                &v_min) {
    
    return exp(-p * (vec[i] - v_min)) / denom;
}


/*!
 * computes sensitivity of aggregated minimum of values specified in vector \p vec with respect to
 * parameter \f$ \alpha \f$.  The aggregation constant is \p p. The sensitivity of values with respect to parameter is
 * provided in \p dvec.
 * The sensitivity expression used is
 * \f[ \frac{d v_{agg}}{d p} =  \frac{ \sum_j \exp (-p (v_j - v_{min})) \frac{dv_j}{d\alpha} }{  \sum_i \exp (-p (v_i - v_{min})) }  \f],
 * where, \f$ v_{min} \f$ is the minimum value out of all values in \p vec.
 * This method differs from the other
 * in that the user provides the cached denominator of the sensitivity  and the minimum value in \p denom and
 * \p v_min.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the
 * summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_minimum_sensitivity(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const std::vector<ScalarType>         &dvec,
                              const real_t                           p,
                              const ScalarType                      &denom,
                              const ScalarType                      &v_min) {
    
    ScalarType
    dv     = 0.;
    
    for (uint_t i=0; i<vec.size(); i++) {
        
        dv += exp(-p * (vec[i] - v_min)) * dvec[i];
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, dv);

   return dv / denom;
}


/*!
 * computes aggregated maximum of values specified in vector \p vec. The aggregation constant is \p p.
 * The aggregation expression used is
 * \f[ v_{agg} = v_{max} + \frac{1}{p} \log \left( \sum_i \exp (p (v_i - v_{max}))  \right) \f],
 * where, \f$ v_{max} \f$ is the maximum value out of all values in \p vec.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_max\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_maximum(const libMesh::Parallel::Communicator *comm,
                  const std::vector<ScalarType>         &vec,
                  const real_t                           p) {
    
    ScalarType
    v      = 0.,
    v_max  = 0.;

    v_max = MAST::Numerics::Utility::real_maximum(vec);

    if (comm) MAST::Numerics::Utility::comm_max(*comm, v_max);

    for (uint_t i=0; i<vec.size(); i++) {
        
        v += exp(p * (vec[i] - v_max));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, v);

    v = v_max + log(v) / p;
    
    return v;
}


/*!
 * computes sensitivity of aggregated maximum of values specified in vector \p vec with respect to \p i th value.
 * The aggregation constant is \p p.
 * The aggregation expression used is
 * \f[ \frac{d v_{agg}}{d v_j} =  \frac{ \exp ( p (v_j - v_{max})) }{  \sum_i \exp ( p (v_i - v_{max})) } \f],
 * where, \f$ v_{max} \f$ is the maximum value out of all values in \p vec.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_maximum_sensitivity(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const uint_t                           i,
                              const real_t                           p) {
    
    ScalarType
    v      = 0.,
    v_max  = 0.;
    
    v_max = MAST::Numerics::Utility::real_maximum(vec);

    if (comm) MAST::Numerics::Utility::comm_max(*comm, v_max);
    
    for (uint_t i=0; i<vec.size(); i++) {
        
        v += exp(p * (vec[i] - v_max));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, v);

    v = exp(p * (vec[i] - v_max)) / v;
    
    return v;
}



/*!
 * computes sensitivity of aggregated maximum of values specified in vector \p vec with respect to
 * parameter \f$ \alpha \f$.  The aggregation constant is \p p. The sensitivity of values with respect to parameter is
 * provided in \p dvec.
 * The sensitivity expression used is
 * \f[ \frac{d v_{agg}}{d p} =  \frac{ \sum_j \exp ( p (v_j - v_{max})) \frac{dv_j}{d\alpha} }{  \sum_i \exp ( p (v_i - v_{max})) } \f],
 * where, \f$ v_{max} \f$ is the maximum value out of all values in \p vec.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_maximum_sensitivity(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const std::vector<ScalarType>         &dvec,
                              const real_t                           p) {
    
    ScalarType
    dv     = 0.,
    v      = 0.,
    v_max  = 0.;
    
    v_max = MAST::Numerics::Utility::real_maximum(vec);

    if (comm) MAST::Numerics::Utility::comm_max(*comm, v_max);

    for (uint_t i=0; i<vec.size(); i++) {
        
        dv += exp(p * (vec[i] - v_max)) * dvec[i];
        v  += exp(p * (vec[i] - v_max));
    }
    
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, dv);
    if (comm) MAST::Numerics::Utility::comm_sum(*comm, v);

    v = dv / v;
    
    return v;
}


/*!
 * Computes the denominator of the sensitivity of aggregated maximum function for use in later sensitivity computations.
 * The value is  \f$  \sum_i \exp (p (v_i - v_{max}))   \f$,
 * where, \f$ v_{max} \f$ is the maximum value out of all values in \p vec. The denominator is returned in
 * \p denom and the minimum value is returned in \p v_max.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
void
aggregate_maximum_denominator(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const real_t                           p,
                              ScalarType                            &denom,
                              ScalarType                            &v_max) {
    
    denom  = 0.;
    
    v_max = MAST::Numerics::Utility::real_maximum(vec);

    if (comm) MAST::Numerics::Utility::comm_max(*comm, v_max);
    
    for (uint_t i=0; i<vec.size(); i++) {
        
        denom  += exp(p * (vec[i] - v_max));
    }

    if (comm) MAST::Numerics::Utility::comm_sum(*comm, denom);
}



/*!
 * computes sensitivity of aggregated maximum of values specified in vector \p vec with respect to \p i th value.
 * The aggregation constant is \p p.
 * The aggregation expression used is
 * \f[ \frac{d v_{agg}}{d v_j} =  \frac{ \exp (p (v_j - v_{max})) }{  \sum_i \exp (p (v_i - v_{max})) }  \f],
 * where, \f$ v_{max} \f$ is the maximum value out of all values in \p vec. This method differs from the other
 * in that the user provides the cached denominator of the sensitivity  and the maximum value in \p denom and
 * \p v_max.
 */
template <typename ScalarType>
ScalarType
aggregate_maximum_sensitivity(const std::vector<ScalarType>   &vec,
                              const uint_t                     i,
                              const real_t                     p,
                              const ScalarType                &denom,
                              const ScalarType                &v_max) {
    
    return exp(p * (vec[i] - v_max)) / denom;
}


/*!
 * computes sensitivity of aggregated maximum of values specified in vector \p vec with respect to
 * parameter \f$ \alpha \f$.  The aggregation constant is \p p. The sensitivity of values with respect to parameter is
 * provided in \p dvec.
 * The sensitivity expression used is
 * \f[ \frac{d v_{agg}}{d p} =  \frac{ \sum_j \exp (p (v_j - v_{max})) \frac{dv_j}{d\alpha} }{  \sum_i \exp (p (v_i - v_{max})) }  \f],
 * where, \f$ v_{max} \f$ is the maximum value out of all values in \p vec.
 * This method differs from the other
 * in that the user provides the cached denominator of the sensitivity  and the maximum value in \p denom and
 * \p v_max.
 * If \p comm is a non-null pointer then the computation is synchronized across all ranks with the \f$ v_min\f$ and
 * the summation obtained from a collective.
 */
template <typename ScalarType>
ScalarType
aggregate_maximum_sensitivity(const libMesh::Parallel::Communicator *comm,
                              const std::vector<ScalarType>         &vec,
                              const std::vector<ScalarType>         &dvec,
                              const real_t                           p,
                              const ScalarType                      &denom,
                              const ScalarType                      &v_max) {
    
    ScalarType
    dv     = 0.;
    
    for (uint_t i=0; i<vec.size(); i++) {
        
        dv += exp(p * (vec[i] - v_max)) * dvec[i];
    }

    if (comm) MAST::Numerics::Utility::comm_sum(*comm, dv);

    return dv / denom;
}


} // Aggregation
} // Optimization
} // MAST

#endif // __mast_optimization_discrete_aggregation_h__
