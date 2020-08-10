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

#ifndef __mast_mesh_generation_bracket_3d_h__
#define __mast_mesh_generation_bracket_3d_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/parameter_data.hpp>
#include <mast/util/getpot_wrapper.hpp>
#include <mast/optimization/design_parameter_vector.hpp>

// libMesh includes
#include <libmesh/system.h>
#include <libmesh/unstructured_mesh.h>
#include <libmesh/fe_type.h>
#include <libmesh/string_to_enum.h>
#include <libmesh/mesh_generation.h>
#include <libmesh/elem.h>
#include <libmesh/node.h>
#include <libmesh/boundary_info.h>
#include <libmesh/dirichlet_boundaries.h>
#include <libmesh/zero_function.h>



namespace MAST {
namespace Mesh {
namespace Generation {


struct Bracket3D {
    
    template <typename ScalarType>
    class Pressure {
    public:
        Pressure(real_t p,
                 real_t l1,
                 real_t frac):
        _p(p), _l1(l1), _frac(frac)
        {}
        virtual ~Pressure() {}
        
        template <typename ContextType>
        inline ScalarType value(ContextType& c) const {
            ScalarType v=(c.qp_location(0)>=_l1*(1.-_frac))?_p:0.;
            return v;
        }
        
        template <typename ContextType, typename ScalarFieldType>
        inline ScalarType derivative(ContextType& c,
                                     const ScalarFieldType& f) const {
            return 0.;
        }
        
    private:
        real_t _p, _l1, _frac;
    };
    
    static const uint_t dim = 3;
    template <typename ScalarType>
    using pressure_t        =  MAST::Mesh::Generation::Bracket3D::Pressure<ScalarType>;
    
    template <typename Context>
    inline real_t
    reference_volume(Context& c) {
        
        real_t
        length  = c.input("length", "length of domain along x-axis", 0.3),
        height  = c.input("height", "length of domain along y-axis", 0.3),
        width   = c.input("width",  "length of domain along z-axis", 0.3);
        
        return length * height * width;
    }
    
    
    
    template <typename Context>
    inline void
    init_analysis_mesh(Context& c,
                       libMesh::UnstructuredMesh& mesh) {
        
        real_t
        length  = c.input("length", "length of domain along x-axis", 0.3),
        height  = c.input("height", "length of domain along y-axis", 0.3),
        width   = c.input("width",  "length of domain along z-axis", 0.3);
        
        uint_t
        nx_divs = c.input("nx_divs", "number of elements along x-axis", 20),
        ny_divs = c.input("ny_divs", "number of elements along y-axis", 20),
        nz_divs = c.input("nz_divs", "number of elements along z-axis", 20);
        
        if (nx_divs%10 != 0 || ny_divs%10 != 0) libmesh_error();
        
        std::string
        t = c.input("elem_type", "type of geometric element in the mesh", "hex8");
        
        libMesh::ElemType
        e_type = libMesh::Utility::string_to_enum<libMesh::ElemType>(t);
        
        //
        // if high order FE is used, libMesh requires atleast a second order
        // geometric element.
        //
        if (c.fe_order > 1 && e_type == libMesh::HEX8)
            e_type = libMesh::HEX27;
        else if (c.fe_order > 1 && e_type == libMesh::TET4)
            e_type = libMesh::TET10;
        
        //
        // initialize the mesh with one element
        //
        libMesh::MeshTools::Generation::build_cube(mesh,
                                                   nx_divs, ny_divs, nz_divs,
                                                   0, length,
                                                   0, height,
                                                   0, width,
                                                   e_type);
        
        _delete_elems_from_bracket_mesh(c, mesh);
    }
    
        
    
    template <typename Context>
    inline void
    init_analysis_dirichlet_conditions(Context& c) {
        
        c.sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({1}, {0, 1, 2}, libMesh::ZeroFunction<real_t>()));
    }
    
    
    
    template <typename ScalarType, typename InitType>
    std::unique_ptr<pressure_t<ScalarType>>
    build_pressure_load(InitType& c) {
        
        real_t
        length      = c.input("length", "length of domain along x-axis", 0.3),
        frac        = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.125),
        p_val       = c.input("pressure", "pressure on side of domain",   5.e7);
        c.p_side_id = 7;
        
        std::unique_ptr<pressure_t<ScalarType>>
        press(new pressure_t<ScalarType>(p_val, length, frac));
        
        return press;
    }
    
        
    
    template <typename ScalarType, typename Context>
    inline void
    init_simp_dvs
    (Context                                               &c,
     MAST::Optimization::DesignParameterVector<ScalarType> &dvs) {
        
        //
        // this assumes that density variable has a constant value per element
        //
        Assert2(c.fe_family == libMesh::LAGRANGE,
                c.fe_family, libMesh::LAGRANGE,
                "Method assumes Lagrange interpolation function for density");
        
        real_t
        tol           = 1.e-12,
        l_frac        = 0.4,
        h_frac        = 0.4,
        length        = c.input("length", "length of domain along x-axis", 0.3),
        height        = c.input("height", "length of domain along y-axis", 0.3),
        x_lim         = length * l_frac,
        y_lim         = height * (1.-h_frac),
        frac          = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.125),
        filter_radius = c.input("filter_radius", "radius of geometric filter for level set field", 0.015),
        rho_min       = c.input("rho_min", "lower limit on density variable", 0.),
        vf            = c.input("volume_fraction", "upper limit for the volume fraction", 0.2);
        
        uint_t
        sys_num = c.rho_sys->number(),
        dof_id  = 0;
        
        real_t
        val     = 0.;
        
        //
        // all ranks will have DVs defined for all variables. So, we should be
        // operating on a replicated mesh
        //
        Assert0(c.mesh->is_replicated(),
                "Function currently assumes replicated mesh");
                
        // iterate over all the element values
        libMesh::MeshBase::const_node_iterator
        it  = c.mesh->nodes_begin(),
        end = c.mesh->nodes_end();
        
        //
        // maximum number of dvs is the number of nodes on the level set function
        // mesh. We will evaluate the actual number of dvs
        //
        //dvs.reserve(c.mesh->n_elem());

        for ( ; it!=end; it++) {
            
            const libMesh::Node& n = **it;
            
            dof_id                     = n.dof_number(sys_num, 0, 0);
            
            if ((n(1)-filter_radius) <= y_lim &&
                (n(0)+filter_radius) >= length*(1.-frac)) {
                
                //
                // set value at the constrained points to be solid material
                //
                if (dof_id >= c.rho_sys->solution->first_local_index() &&
                    dof_id <  c.rho_sys->solution->last_local_index())
                    c.rho_sys->solution->set(dof_id, 1.e0);
            }
            else {
                
                MAST::Optimization::DesignParameter<ScalarType>
                *dv = new MAST::Optimization::DesignParameter<ScalarType>(vf);
                dv->set_point(n(0), n(1), n(2));

                MAST::Base::ParameterData
                &data = dvs.add_topology_parameter(*dv, dof_id);
            }
        }
        
        c.rho_sys->solution->close();
    }
    
    
    
    template <typename Context>
    inline void
    _delete_elems_from_bracket_mesh(Context& c,
                                    libMesh::MeshBase &mesh) {
        
        real_t
        tol     = 1.e-12,
        x       = -1.,
        y       = -1.,
        l_frac  = 0.4,
        w_frac  = 0.4,
        length  = c.input("length", "length of domain along x-axis", 0.3),
        height  = c.input("height", "length of domain along y-axis", 0.3),
        x_lim   = length * l_frac,
        y_lim   = height * (1.-w_frac);
        
        //
        // now, remove elements that are outside of the L-bracket domain
        //
        libMesh::MeshBase::element_iterator
        e_it   = mesh.elements_begin(),
        e_end  = mesh.elements_end();
        
        for ( ; e_it!=e_end; e_it++) {
            
            libMesh::Elem* elem = *e_it;
            x = length;
            y = 0.;
            for (uint_t i=0; i<elem->n_nodes(); i++) {
                const libMesh::Node& n = elem->node_ref(i);
                if (x > n(0)) x = n(0);
                if (y < n(1)) y = n(1);
            }
            
            //
            // delete element if the lowest x,y locations are outside of the bracket
            // domain
            //
            if (x >= x_lim && y<= y_lim)
                mesh.delete_elem(elem);
        }
        
        mesh.prepare_for_use();
        
        //
        // add the two additional boundaries to the boundary info so that
        // we can apply loads on them
        //
        bool
        facing_right = false,
        facing_down  = false;
        
        e_it   = mesh.elements_begin();
        e_end  = mesh.elements_end();
        
        for ( ; e_it != e_end; e_it++) {
            
            libMesh::Elem* elem = *e_it;
            
            if (!elem->on_boundary()) continue;
            
            for (uint_t i=0; i<elem->n_sides(); i++) {
                
                if (elem->neighbor_ptr(i)) continue;
                
                std::unique_ptr<libMesh::Elem> s(elem->side_ptr(i).release());
                
                const libMesh::Point p = s->centroid();
                
                facing_right = true;
                facing_down  = true;
                for (uint_t j=0; j<s->n_nodes(); j++) {
                    const libMesh::Node& n = s->node_ref(j);
                    
                    if (n(0) < x_lim ||  n(1) > y_lim) {
                        facing_right = false;
                        facing_down  = false;
                    }
                    else if (std::fabs(n(0) - p(0)) > tol)
                        facing_right = false;
                    else if (std::fabs(n(1) - p(1)) > tol)
                        facing_down = false;
                }
                
                if (facing_right) mesh.boundary_info->add_side(elem, i, 6);
                if (facing_down) mesh.boundary_info->add_side(elem, i, 7);
            }
        }
        
        mesh.boundary_info->sideset_name(6) = "facing_right";
        mesh.boundary_info->sideset_name(7) = "facing_down";
    }
};

}  // namespace Generation
}  // namespace Mesh
}  // namespace MAST


#endif // __mast_mesh_generation_bracket_3d_h__
