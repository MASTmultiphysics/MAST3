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

#ifndef __test__test_helpers__
#define __test__test_helpers__

// C++ includes
#include <vector>

// MAST includes
#include <mast/base/mast_data_types.h>
//#include "elasticity/structural_element_base.h"
//#include "base/physics_discipline_base.h"
//#include "libmesh/point.h"
//#include "libmesh/face_quad4.h"

namespace MAST {
namespace Test {

    /**
     * Converts an Eigen Matrix object to a std::vector. Useful for performing
     * elementwise comparisons in Catch2 tests.
     */
template<typename ValType>
std::vector<real_t> eigen_matrix_to_std_vector(const ValType& M) {

    static_assert(std::is_same<real_t, typename ValType::Scalar>::value,
                  "Scalar type of matrix shoudl be real_t");
    
    const real_t *v = M.data();
    std::vector<real_t> vec(v, v+M.rows()*M.cols());
    return vec;
}


//    /**
//     * Calcualtes the area of a 2D polygon using the shoelace formula.
//     */
//    real_tget_shoelace_area(RealMatrixX X);
//
//    /**
//     * Approximates the internal Jacobian of an element using a 6th order
//     * accurate central finite difference scheme.
//     */
//    void approximate_internal_jacobian_with_finite_difference(
//            MAST::StructuralElementBase& elem,
//            const RealVectorX& initial_elem_solution,
//            RealMatrixX& jacobian);
//
//
//    /**
//     * Approximates the side external jacobian using a 4th order accurate central
//     * finite difference scheme.
//     */
//    void approximate_side_external_jacobian_with_finite_difference(
//            MAST::StructuralElementBase& elem,
//            MAST::PhysicsDisciplineBase& discipline,
//            const RealVectorX& initial_elem_solution,
//            RealMatrixX& jacobian);
//
//    /**
//     * Approximates the volume external jacobian using a 4th order accurate central
//     * finite difference scheme.
//     */
//    void approximate_volume_external_jacobian_with_finite_difference(
//            MAST::StructuralElementBase& elem,
//            MAST::PhysicsDisciplineBase& discipline,
//            const RealVectorX& initial_elem_solution,
//            RealMatrixX& jacobian);
//
//    /**
//     * Approximates the inertial jacobian using a 6th order accurate central
//     * finite difference scheme.
//     */
//    void approximate_inertial_jacobian_with_finite_difference(
//            MAST::StructuralElementBase& elem,
//            const RealVectorX& initial_elem_solution,
//            RealMatrixX& jacobian);
//
//
//    /**
//     * Approximates the thermal jacobian using a 6th order accurate central
//     * finite difference scheme.
//     */
//    void approximate_thermal_jacobian_with_finite_difference(
//                                        MAST::StructuralElementBase& elem,
//                                        const RealVectorX& initial_elem_solution,
//                                        RealMatrixX& jacobian,
//                                        MAST::BoundaryConditionBase& thermal_bc);
//
//
//    /**
//     * Transform an element by applying any combination of: shifts, scales,
//     * rotations, and shears. Useful for testing elements of different geometries
//     * and in different orientations.
//     */
//    void transform_element(libMesh::MeshBase& mesh, const RealMatrixX X0,
//                           real_tshift_x, real_tshift_y, real_tshift_z,
//                           real_tscale_x, real_tscale_y,
//                           real_trotation_x, real_trotation_y, real_trotation_z,
//                           real_tshear_x = 0, real_tshear_y = 0);
} // namespace Test
} // namespace MAST
#endif // __test__test_helpers__
