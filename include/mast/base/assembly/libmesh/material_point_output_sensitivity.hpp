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

#ifndef __mast_libmesh_material_point_output_sensitivity_h__
#define __mast_libmesh_material_point_output_sensitivity_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/base/assembly/libmesh/utility.hpp>
#include <mast/base/assembly/libmesh/accessor.hpp>
#include <mast/numerics/utility.hpp>
#include <mast/optimization/design_parameter_vector.hpp>
#include <mast/mesh/libmesh/utility.hpp>

// libMesh includes
#include <libmesh/parallel.h>


namespace MAST {
namespace Base {
namespace Assembly {
namespace libMeshWrapper {

/*!
 * This class computes the adjoint sensitivity of scalar output quantity that is defined based on
 * material point data, for example an aggregated stress quantity based on quadrature point stress
 * values.
 */

template <typename ScalarType,
          typename ResidualElemOpsType,
          typename OutputElemOpsType>
class MaterialPointOutputSensitivity {
    
public:

    static_assert(std::is_same<ScalarType,
                  typename ResidualElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");
    static_assert(std::is_same<ScalarType,
                  typename OutputElemOpsType::scalar_t>::value,
                  "Scalar type of assembly and element operations must be same");

    
    MaterialPointOutputSensitivity(libMesh::Communicator &comm):
    _comm         (comm),
    _e_ops        (nullptr),
    _output_e_ops (nullptr)
    { }
    
    virtual ~MaterialPointOutputSensitivity() {}
    
    inline void set_elem_ops(ResidualElemOpsType &e_ops,
                             OutputElemOpsType   &output_ops) {
        
        _e_ops        = &e_ops;
        _output_e_ops = &output_ops;
    }

    /*!
     *  output derivative is defined as a
     * \f[ \frac{dQ}{d\alpha} = \frac{\partial Q}{\partial \alpha} + \lambda^T \frac{\partial R}{\partial \alpha} \f]
     */
    template <typename Vec1Type,
              typename Vec2Type,
              typename IndexingType,
              typename StorageType,
              typename ContextType,
              typename ScalarFieldType>
    inline ScalarType
    assemble(ContextType               &c,
             ScalarFieldType           &f,
             const Vec1Type            &X,
             const Vec1Type            &X_adj,
             const IndexingType        &index,
             const StorageType         &stress) {
                
        Assert0(_e_ops && _output_e_ops, "Elem Operation objects not initialized");

        ScalarType
        val = 0.;
        
        // iterate over each element, initialize it and get the relevant
        // analysis quantities
        typename MAST::Base::Assembly::libMeshWrapper::Accessor<ScalarType, Vec1Type>
        sol_accessor     (*c.sys, X),
        dsol_accessor    (*c.sys, X),
        adj_accessor     (*c.sys, X_adj);
        
        using elem_vector_t = typename ResidualElemOpsType::vector_t;
        using elem_matrix_t = typename ResidualElemOpsType::matrix_t;
        
        elem_vector_t
        dres_e;

        uint_t
        idx = 0;

        libMesh::MeshBase::const_element_iterator
        el     = c.mesh->active_local_elements_begin(),
        end_el = c.mesh->active_local_elements_end();
        
        // first compute the sensitivity information assuming unfiltered
        // density variables.
        for ( ; el != end_el; ++el) {
            
            // set element in the context, which will be used for
            // the initialization routines
            c.elem = *el;
            
            sol_accessor.init  (*c.elem);
            dsol_accessor.init (*c.elem);
            adj_accessor.init  (*c.elem);
            
            dres_e.setZero(sol_accessor.n_dofs());
            
            // first we compute the partial derivative of the
            // residual wrt the parameter.
            _e_ops->derivative(c,
                               f,
                               sol_accessor,
                               dres_e,
                               nullptr);
            
            
            val +=
            _output_e_ops->derivative(c, f,
                                      sol_accessor,
                                      dsol_accessor,
                                      index,
                                      stress) // partial derivative of output
            + adj_accessor.dot(dres_e)); // the adjoint vector combined w/ res sens
        }
        
        MAST::Numerics::Utility::comm_sum(_comm, val);
        
        return val;
    }

private:
  
    libMesh::Comm        &_comm;
    ResidualElemOpsType  *_e_ops;
    OutputElemOpsType    *_output_e_ops;
};

} // namespace libMeshWrapper
} // namespace Assembly
} // namespace Base
} // namespace MAST


#endif // __mast_libmesh_material_point_output_sensitivity_h__
