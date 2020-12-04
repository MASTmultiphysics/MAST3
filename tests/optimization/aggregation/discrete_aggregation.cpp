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

// Catch includes
#include "catch.hpp"

// C++ includes
#include <algorithm>

// MAST includes
#include <mast/optimization/aggregation/discrete_aggregation.hpp>

// Test includes
#include <test_helpers.h>

namespace MAST {
namespace Test {
namespace Optimization {
namespace Aggregation {


void run_checks(const real_t p, bool check_agg_min) {
    
    uint_t
    n  = 10;
    
    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    vals  = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Random(n),
    dvals = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Random(n);
    
    std::vector<real_t>
    vec  (vals.data(), vals.data()+n),
    dvec (dvals.data(), dvals.data()+n);

    real_t
    min_val  = *std::min_element(vec.begin(), vec.end()),
    min_agg  = MAST::Optimization::Aggregation::aggregate_minimum(vec, p),
    dmin_val = MAST::Optimization::Aggregation::aggregate_minimum_sensitivity(vec, dvec, p),
    max_val  = *std::max_element(vec.begin(), vec.end()),
    max_agg  = MAST::Optimization::Aggregation::aggregate_maximum(vec, p),
    dmax_val = MAST::Optimization::Aggregation::aggregate_maximum_sensitivity(vec, dvec, p);

    // check aggregated minimum if asked
    if (check_agg_min) {
        
        CHECK(min_val == Catch::Detail::Approx(min_agg).margin(1.e-2));
        CHECK(max_val == Catch::Detail::Approx(max_agg).margin(1.e-2));
    }

    // complex step sensitivity of each component
    std::vector<complex_t>
    vec_cs(n);
    
    complex_t
    val;

    real_t
    v            = 0.,
    dval         = 0.,
    dval_cs      = 0.,
    dval_agg_min = 0.,
    dval_agg_max = 0.;
    
    for (uint_t i=0; i<n; i++) {
        
        // add perturbation to the ith component
        for (uint_t j=0; j<n; j++) vec_cs[j] = vec[j];
        vec_cs[i] += complex_t(0., 1.e-12);

        //////////////////////////////////////////////////////////
        // sensitivity of minimum aggregate
        //////////////////////////////////////////////////////////
        // analytical sensitivity wrt ith var
        dval = MAST::Optimization::Aggregation::aggregate_minimum_sensitivity(vec, i, p);
        
        // complex-step sensitivity wrt ith var
        val     = MAST::Optimization::Aggregation::aggregate_minimum(vec_cs, p);
        dval_cs = val.imag()/1.e-12;
        
        dval_agg_min += dval_cs * dvec[i];
        
        CHECK(dval_cs == Catch::Detail::Approx(dval));

        //////////////////////////////////////////////////////////
        // sensitivity of maximum aggregate
        //////////////////////////////////////////////////////////
        // analytical sensitivity wrt ith var
        dval = MAST::Optimization::Aggregation::aggregate_maximum_sensitivity(vec, i, p);
        
        // complex-step sensitivity wrt ith var
        val     = MAST::Optimization::Aggregation::aggregate_maximum(vec_cs, p);
        dval_cs = val.imag()/1.e-12;

        dval_agg_max += dval_cs * dvec[i];
        
        CHECK(dval_cs == Catch::Detail::Approx(dval));
    }

    // check aggregated sensitivity
    CHECK(dval_agg_min == Catch::Detail::Approx(dmin_val));
    CHECK(dval_agg_max == Catch::Detail::Approx(dmax_val));
    
#if MAST_ENABLE_ADOLC == 1
    adtl::setNumDir(1);
    
    std::vector<adouble_tl_t>
    vec_ad(n, 0.);
    
    for (uint_t i=0; i<n; i++) {
        
        for (uint_t j=0; j<n; j++) {
            vec_ad[j] = vec[j];
            vec_ad[j].setADValue(&dvec[j]);
        }
    }

    //////////////////////////////////////////////////////////
    // sensitivity of minimum aggregate
    //////////////////////////////////////////////////////////
    // analytical sensitivity wrt ith var
    {
        adouble_tl_t
        val_ad = MAST::Optimization::Aggregation::aggregate_minimum(vec_ad, p);
        
        CHECK(val_ad.getValue() == Catch::Detail::Approx(min_agg));
        CHECK(*val_ad.getADValue() == Catch::Detail::Approx(dmin_val));
    }

    //////////////////////////////////////////////////////////
    // sensitivity of maximum aggregate
    //////////////////////////////////////////////////////////
    // analytical sensitivity wrt ith var
    {
        adouble_tl_t
        val_ad = MAST::Optimization::Aggregation::aggregate_maximum(vec_ad, p);
        
        CHECK(val_ad.getValue() == Catch::Detail::Approx(max_agg));
        CHECK(*val_ad.getADValue() == Catch::Detail::Approx(dmax_val));
    }
#endif
}


TEST_CASE("discrete_aggregation",
          "[Optimization][Aggregation]") {

    // check only sensitivity for low value of aggregation constant
    run_checks(10, false);
    
    // check both aggregated value and sensitivity for high value of aggregation constant
    run_checks(100, true);
}

} // namespace Aggregation
} // namespace Optimization
} // namespace Test
} // namespace MAST


