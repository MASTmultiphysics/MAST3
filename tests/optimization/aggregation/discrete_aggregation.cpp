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




TEST_CASE("discrete_aggregation",
          "[Optimization][Aggregation]") {

    uint_t
    n  = 10;
    
    Eigen::Matrix<real_t, Eigen::Dynamic, 1>
    vals = Eigen::Matrix<real_t, Eigen::Dynamic, 1>::Random(n);
    
    std::vector<real_t>
    vec (vals.data(), vals.data()+n);

    real_t
    min_val = *std::min_element(vec.begin(), vec.end()),
    min_agg = MAST::Optimization::Aggregation::aggregate_minimum(vec, 100),
    max_val = *std::max_element(vec.begin(), vec.end()),
    max_agg = MAST::Optimization::Aggregation::aggregate_maximum(vec, 100);

    CHECK(min_val == Catch::Detail::Approx(min_agg).margin(1.e-2));
    CHECK(max_val == Catch::Detail::Approx(max_agg).margin(1.e-2));

}

} // namespace Aggregation
} // namespace Optimization
} // namespace Test
} // namespace MAST


