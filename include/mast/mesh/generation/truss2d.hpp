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

#ifndef __mast_mesh_generation_truss_2d_h__
#define __mast_mesh_generation_truss_2d_h__

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


struct Truss2D {
    
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
            ScalarType v=(fabs(c.qp_location(0)-_l1*0.5) <= 0.5*_frac*_l1)?_p:0.;
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
    
    static const uint_t dim = 2;
    template <typename ScalarType>
    using pressure_t        =  MAST::Mesh::Generation::Truss2D::Pressure<ScalarType>;
    
    template <typename Context>
    inline real_t
    reference_volume(Context& c) {
        
        real_t
        length  = c.input("length", "length of domain along x-axis", 0.24),
        height  = c.input("height", "length of domain along y-axis", 0.04);
        
        return length * height;
    }
    
    
    inline uint_t idx(const libMesh::ElemType type,
                      const uint_t nx,
                      const uint_t i,
                      const uint_t j)
    {
      switch(type)
        {
            case libMesh::QUAD4:
            {
                return i + (nx+1)*j;
            }
                
            case libMesh::QUAD9:
            {
                return i + (2*nx+1)*j;
            }
                
            default:
                Error(false, "Invalid element type");
        }

      return libMesh::invalid_uint;
    }

    
    
    inline void build_mesh(libMesh::UnstructuredMesh & mesh,
                           const uint_t nx,
                           const uint_t ny,
                           const real_t length,
                           const real_t height,
                           const real_t dirichlet_length_fraction,
                           const libMesh::ElemType type) {
        
        Assert0(type == libMesh::QUAD4 || type == libMesh::QUAD9,
                "Method only implemented for Quad4/Quad9");
        
        // Clear the mesh and start from scratch
        mesh.clear();
        
        libMesh::BoundaryInfo & boundary_info = mesh.get_boundary_info();
        
        mesh.set_mesh_dimension(3);
        mesh.set_spatial_dimension(3);
        mesh.reserve_elem(nx*ny);

        if (type == libMesh::QUAD4)
            mesh.reserve_nodes( (nx+1)*(ny+1));
        else if (type == libMesh::QUAD9)
            mesh.reserve_nodes( (2*nx+1)*(2*ny+1));

        real_t
        xmax    = length,
        ymax    = height;


        std::map<uint_t, libMesh::Node*> nodes;
        
        // Build the nodes.
        uint_t
        node_id = 0,
        n       = 0;
        switch (type)
        {
            case libMesh::QUAD4: {

                for (uint_t j=0; j<=ny; j++)
                    for (uint_t i=0; i<=nx; i++) {
                        nodes[node_id] =
                        mesh.add_point(libMesh::Point(static_cast<real_t>(i)/static_cast<real_t>(nx)*length,
                                                      static_cast<real_t>(j)/static_cast<real_t>(ny)*height,
                                                      0.),
                                       n++);
                        node_id++;
                    }
                
                
                break;
            }

            case libMesh::QUAD9: {

                for (uint_t j=0; j<=(2*ny); j++)
                    for (uint_t i=0; i<=(2*nx); i++) {
                        nodes[node_id] =
                        mesh.add_point(libMesh::Point(static_cast<real_t>(i)/static_cast<real_t>(2*nx)*length,
                                                      static_cast<real_t>(j)/static_cast<real_t>(2*ny)*height,
                                                      0.),
                                       n++);
                        node_id++;
                    }
                
                break;
            }
                
                
            default:
                Assert0(false, "ERROR: Unrecognized 2D element type.");
        }

        // Build the elements.
        uint_t
        elem_id = 0;
        switch (type) {
                
            case libMesh::QUAD4: {
                
                for (uint_t j=0; j<ny; j++)
                for (uint_t i=0; i<nx; i++) {
                    
                    libMesh::Elem
                    *elem = libMesh::Elem::build(libMesh::QUAD4).release();
                    elem->set_id(elem_id++);
                    mesh.add_elem(elem);
                    
                    elem->set_node(0) = nodes[idx(type,nx,i,j)      ];
                    elem->set_node(1) = nodes[idx(type,nx,i+1,j)    ];
                    elem->set_node(2) = nodes[idx(type,nx,i+1,j+1)  ];
                    elem->set_node(3) = nodes[idx(type,nx,i,j+1)    ];
                    
                    if (j == 0)
                        boundary_info.add_side(elem, 0, 0);
                    
                    if (j == (ny-1))
                        boundary_info.add_side(elem, 2, 2);
                    
                    if (i == 0)
                        boundary_info.add_side(elem, 3, 3);
                    
                    if (i == (nx-1))
                        boundary_info.add_side(elem, 1, 1);
                    
                    if (j == 0 && i <= dirichlet_length_fraction * nx)
                        boundary_info.add_side(elem, 0, 6);

                    if (j == 0 && i >= (1.-dirichlet_length_fraction)* nx)
                        boundary_info.add_side(elem, 0, 7);

                    if (j == 0 && i == 0)
                        boundary_info.add_side(elem, 0, 8);
                    }
                break;
            }
                
                
            case libMesh::QUAD9: {
                
                for (uint_t j=0; j<(2*ny); j += 2)
                    for (uint_t i=0; i<(2*nx); i += 2) {
                        
                        libMesh::Elem
                        *elem = libMesh::Elem::build(libMesh::QUAD9).release();
                        elem->set_id(elem_id++);
                        mesh.add_elem(elem);
                        
                        elem->set_node(0)  = nodes[idx(type,nx,i,  j)  ];
                        elem->set_node(1)  = nodes[idx(type,nx,i+2,j)  ];
                        elem->set_node(2)  = nodes[idx(type,nx,i+2,j+2)  ];
                        elem->set_node(3)  = nodes[idx(type,nx,i,  j+2)  ];
                        elem->set_node(4)  = nodes[idx(type,nx,i+1,j)  ];
                        elem->set_node(5)  = nodes[idx(type,nx,i+2,j+1)  ];
                        elem->set_node(6)  = nodes[idx(type,nx,i+1,j+2)  ];
                        elem->set_node(7)  = nodes[idx(type,nx,i,  j+1)  ];
                        elem->set_node(8)  = nodes[idx(type,nx,i+1,j+1)  ];
                        
                        if (j == 0)
                            boundary_info.add_side(elem, 0, 0);
                        
                        if (j == 2*(ny-1))
                            boundary_info.add_side(elem, 2, 2);
                        
                        if (i == 0)
                            boundary_info.add_side(elem, 3, 3);
                        
                        if (i == 2*(nx-1))
                            boundary_info.add_side(elem, 1, 1);
                        
                        if (j == 0 && i <= dirichlet_length_fraction * 2*nx)
                            boundary_info.add_side(elem, 0, 6);

                        if (j == 0 && i >= (1.-dirichlet_length_fraction)* 2*nx)
                            boundary_info.add_side(elem, 0, 7);

                        if (j == 0 && i == 0)
                            boundary_info.add_side(elem, 0, 8);
                    }
                break;
            }
                
            default:
                Assert0(false, "ERROR: Unrecognized 2D element type.");
        }
        
        // Add sideset names to boundary info (Z axis out of the screen)
        boundary_info.sideset_name(0) = "bottom";
        boundary_info.sideset_name(1) = "right";
        boundary_info.sideset_name(2) = "top";
        boundary_info.sideset_name(3) = "left";
        boundary_info.sideset_name(6) = "left_dirichlet";
        boundary_info.sideset_name(7) = "right_dirichlet";

        // Add nodeset names to boundary info
        boundary_info.nodeset_name(0) = "bottom";
        boundary_info.nodeset_name(1) = "right";
        boundary_info.nodeset_name(2) = "top";
        boundary_info.nodeset_name(3) = "left";

        // Done building the mesh.  Now prepare it for use.
        mesh.prepare_for_use ();
    }

    
    template <typename Context>
    inline void
    init_analysis_mesh(Context& c,
                       libMesh::UnstructuredMesh& mesh) {
        
        real_t
        length  = c.input("length", "length of domain along x-axis", 0.24),
        height  = c.input("height", "length of domain along y-axis", 0.04),
        dirichlet_length_fraction = c.input
        ("dirichlet_length_fraction",
         "length fraction of the truss boundary where dirichlet condition is applied",
         0.02);

        uint_t
        nx_divs = c.input("nx_divs", "number of elements along x-axis", 30),
        ny_divs = c.input("ny_divs", "number of elements along y-axis", 5);
        
        std::string
        t = c.input("elem_type", "type of geometric element in the mesh", "quad4");
        
        libMesh::ElemType
        e_type = libMesh::Utility::string_to_enum<libMesh::ElemType>(t);
        
        //
        // if high order FE is used, libMesh requires atleast a second order
        // geometric element.
        //
        if (c.fe_order > 1 && e_type == libMesh::QUAD4)
            e_type = libMesh::QUAD9;
        
        //
        // initialize the mesh with one element
        //
        build_mesh(mesh,
                   nx_divs, ny_divs,
                   length,
                   height,
                   dirichlet_length_fraction,
                   e_type);
    }
    
    
    
    template <typename Context>
    inline void
    init_analysis_dirichlet_conditions(Context& c) {
        
        c.sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({6, 7}, {0, 1}, libMesh::ZeroFunction<real_t>()));
    }
    
    
    
    template <typename ScalarType, typename InitType>
    std::unique_ptr<pressure_t<ScalarType>>
    build_pressure_load(InitType& c) {
        
        real_t
        length      = c.input("length", "length of domain along x-axis", 0.24),
        frac        = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.2),
        p_val       = c.input("pressure", "pressure on side of domain",   2.e4);
        c.p_side_id = 2;
        
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
        length        = c.input("length", "length of domain along x-axis", 0.24),
        height        = c.input("height", "length of domain along y-axis", 0.04),
        frac          = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.2),
        filter_radius = c.input("filter_radius", "radius of geometric filter for level set field", 0.008),
        vf            = c.input("volume_fraction", "upper limit for the volume fraction", 0.2);
        
        uint_t
        sys_num   = c.rho_sys->number(),
        first_dof = c.rho_sys->get_dof_map().first_dof(c.rho_sys->comm().rank()),
        end_dof   = c.rho_sys->get_dof_map().end_dof(c.rho_sys->comm().rank()),
        dof_id    = 0;
        
        real_t
        val     = 0.;
        
        libMesh::MeshBase::const_element_iterator
        e_it  = c.mesh->local_elements_begin(),
        e_end = c.mesh->local_elements_end();
        
        std::set<const libMesh::Node*> nodes;
        
        for ( ; e_it != e_end; e_it++) {
            
            const libMesh::Elem* e = *e_it;
            
            for (uint_t i=0; i<e->n_nodes(); i++) {
                
                const libMesh::Node& n = *e->node_ptr(i);
                
                // if we have alredy operated on this node, then
                // we skip it
                if (nodes.count(&n))
                    continue;
                
                // otherwise, we add it to the set of operated nodes and
                // check if a design parameter should be computed for this
                nodes.insert(&n);

                dof_id = n.dof_number(sys_num, 0, 0);

                /*if ((n(1)-filter_radius) <= 0. &&
                    (n(0)-filter_radius) <= length*0.5*(1.+frac) &&
                    (n(0)+filter_radius) >= length*0.5*(1.-frac)) {
                    
                    //
                    // set value at the constrained points to be solid material
                    //
                    if (dof_id >= first_dof &&
                        dof_id <  end_dof)
                        c.rho_sys->solution->set(dof_id, 1.e0);
                }
                else*/ {
                    
                    MAST::Optimization::DesignParameter<ScalarType>
                    *dv = new MAST::Optimization::DesignParameter<ScalarType>(vf);
                    dv->set_point(n(0), n(1), n(2));
                    
                    if (dof_id >= first_dof &&
                        dof_id <  end_dof)
                        dvs.add_topology_parameter(*dv, dof_id);
                    else
                        dvs.add_ghosted_topology_parameter(*dv, dof_id);
                }
            }
        }

        dvs.synchronize(c.rho_sys->get_dof_map());
        c.rho_sys->solution->close();
    }
};

}  // namespace Generation
}  // namespace Mesh
}  // namespace MAST


#endif // __mast_mesh_generation_truss_2d_h__
