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

// MAST includes
#ifndef MAST_TESTING
#define MAST_TESTING 1
#endif

#include <structural/example_6/example_6.cpp>

// Test includes
#include <test_helpers.h>

extern libMesh::LibMeshInit* p_global_init;

namespace MAST {
namespace Test {
namespace Mesh {
namespace libMesh {
namespace GeometricFilter {




inline void test_filter_transpose_operation()  {

    std::vector<std::string> arguments = {"filter_radius=0.05"};

    std::vector<char*> argv;
    for (const auto& arg : arguments)
        argv.push_back((char*)arg.data());
    argv.push_back(nullptr);

    MAST::Utility::GetPotWrapper input(argv.size() - 1, argv.data());

    using traits_t    = MAST::Examples::Structural::Example6::Traits<real_t, real_t, real_t, MAST::Mesh::Generation::Bracket2D>;
    
    typename traits_t::ex_init_t ex_init(p_global_init->comm(), input);

    MAST::Optimization::DesignParameterVector<traits_t::scalar_t> dvs(p_global_init->comm());
    ex_init.model->init_simp_dvs(ex_init, dvs);

    const uint_t
    n_rho_vals      = ex_init.rho_sys->n_dofs(),
    first_local_rho = ex_init.rho_sys->get_dof_map().first_dof(ex_init.rho_sys->comm().rank()),
    last_local_rho  = ex_init.rho_sys->get_dof_map().end_dof(ex_init.rho_sys->comm().rank()),
    qoi_dof         = last_local_rho-1;
    
    std::unique_ptr<typename traits_t::assembled_vector_t>
    rho_base(ex_init.rho_sys->solution->zero_clone().release()),
    vec1(ex_init.rho_sys->solution->zero_clone().release()),
    rho_sens_filtered(ex_init.rho_sys->solution->zero_clone().release());

    /*for (uint_t qoi_dof=0; qoi_dof<n_rho_vals; qoi_dof++)*/ {
        
        
        for (uint_t i=0; i<n_rho_vals; i++) {
            
            rho_base->zero();
            rho_base->set(i, 1.);
            rho_base->close();
            
            ex_init.filter->template compute_filtered_values
            <traits_t::scalar_t,
            typename traits_t::assembled_vector_t,
            typename traits_t::assembled_vector_t>
            (dvs, *rho_base, *vec1);
            
            vec1->close();
            
            rho_sens_filtered->set(i, vec1->el(qoi_dof));
        }
        
        rho_sens_filtered->close();
        
        rho_base->zero();
        rho_base->set(qoi_dof, 1.);
        rho_base->close();
        
        ex_init.filter->template compute_reverse_filtered_values
        <traits_t::scalar_t,
        typename traits_t::assembled_vector_t,
        typename traits_t::assembled_vector_t>
        (dvs, *rho_base, *vec1);
        
        vec1->close();
        
        Eigen::Matrix<traits_t::scalar_t, Eigen::Dynamic, 1>
        v1  = Eigen::Matrix<traits_t::scalar_t, Eigen::Dynamic, 1>::Zero(n_rho_vals),
        v2  = Eigen::Matrix<traits_t::scalar_t, Eigen::Dynamic, 1>::Zero(n_rho_vals);
        
        for (uint_t i=0; i<n_rho_vals; i++) {
            v1[i] = rho_sens_filtered->el(i);
            v2[i] = vec1->el(i);
        }
        
        CHECK_THAT(MAST::Test::eigen_matrix_to_std_vector(v1),
                   Catch::Approx(MAST::Test::eigen_matrix_to_std_vector(v2)));
    }
}



TEST_CASE("geometric_filter_transpose",
          "[Optimization][Topology][Filter]") {
    
    test_filter_transpose_operation();
}

} // namespace SIMP
} // namespace Topology
} // namespace Optimization
} // namespace Test
} // namespace MAST


