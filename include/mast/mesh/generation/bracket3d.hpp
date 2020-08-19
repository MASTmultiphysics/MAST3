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
    
    
    inline uint_t idx(const libMesh::ElemType type,
                      const uint_t nx,
                      const uint_t ny,
                      const uint_t i,
                      const uint_t j,
                      const uint_t k)
    {
      switch(type)
        {
            case libMesh::HEX8:
            {
                return i + (nx+1)*(j + k*(ny+1));
            }
                
            case libMesh::HEX27:
            {
                return i + (2*nx+1)*(j + k*(2*ny+1));
            }
                
            default:
                Error(false, "Invalid element type");
        }

      return libMesh::invalid_uint;
    }

    
    inline void build_cube(libMesh::UnstructuredMesh & mesh,
                           const uint_t nx,
                           const uint_t ny,
                           const uint_t nz,
                           const real_t length,
                           const real_t height,
                           const real_t width,
                           const libMesh::ElemType type) {
        
        Assert0(type == libMesh::HEX8 || type == libMesh::HEX27,
                "Method only implemented for Hex8/Hex27");
        
        // Clear the mesh and start from scratch
        mesh.clear();
        
        libMesh::BoundaryInfo & boundary_info = mesh.get_boundary_info();
        
        mesh.set_mesh_dimension(3);
        mesh.set_spatial_dimension(3);
        mesh.reserve_elem(nx*ny*nz);

        if (type == libMesh::HEX8)
            mesh.reserve_nodes( (nx+1)*(ny+1)*(nz+1) );
        else if (type == libMesh::HEX27)
            mesh.reserve_nodes( (2*nx+1)*(2*ny+1)*(2*nz+1) );

        real_t
        xmax    = length,
        ymax    = height,
        zmax    = width;
        
                

        std::map<uint_t, libMesh::Node*> nodes;
        
        // Build the nodes.
        uint_t
        node_id = 0,
        n       = 0;
        switch (type)
        {
            case libMesh::HEX8: {

                for (uint_t k=0; k<=nz; k++)
                    for (uint_t j=0; j<=ny; j++)
                        for (uint_t i=0; i<=nx; i++) {
                            if ( i<=nx/10*4 || j>=ny/10*6) {
                                nodes[node_id] =
                                mesh.add_point(libMesh::Point(static_cast<real_t>(i)/static_cast<real_t>(nx)*length,
                                                              static_cast<real_t>(j)/static_cast<real_t>(ny)*height,
                                                              static_cast<real_t>(k)/static_cast<real_t>(nz)*width),
                                               n++);
                            }
                            node_id++;
                        }
                
                
                break;
            }

            case libMesh::HEX27: {

                for (uint_t k=0; k<=(2*nz); k++)
                    for (uint_t j=0; j<=(2*ny); j++)
                        for (uint_t i=0; i<=(2*nx); i++) {
                            if ( i<=2*nx/10*4 || j>=2*ny/10*6) {
                                nodes[node_id] =
                                mesh.add_point(libMesh::Point(static_cast<real_t>(i)/static_cast<real_t>(2*nx)*length,
                                                              static_cast<real_t>(j)/static_cast<real_t>(2*ny)*height,
                                                              static_cast<real_t>(k)/static_cast<real_t>(2*nz)*width),
                                               n++);
                            }
                            node_id++;
                        }
                
                break;
            }
                
                
            default:
                Assert0(false, "ERROR: Unrecognized 3D element type.");
        }

        // Build the elements.
        uint_t
        elem_id = 0;
        switch (type)
        {
            case libMesh::HEX8:
            {
                for (uint_t k=0; k<nz; k++)
                    for (uint_t j=0; j<ny; j++)
                        for (uint_t i=0; i<nx; i++) {
                            if (i < nx*4/10 || j>=ny*6/10) {
                                
                                libMesh::Elem
                                *elem = mesh.add_elem(libMesh::Elem::build_with_id(libMesh::HEX8, elem_id++).release());
                                elem->set_node(0) = nodes[idx(type,nx,ny,i,j,k)      ];
                                elem->set_node(1) = nodes[idx(type,nx,ny,i+1,j,k)    ];
                                elem->set_node(2) = nodes[idx(type,nx,ny,i+1,j+1,k)  ];
                                elem->set_node(3) = nodes[idx(type,nx,ny,i,j+1,k)    ];
                                elem->set_node(4) = nodes[idx(type,nx,ny,i,j,k+1)    ];
                                elem->set_node(5) = nodes[idx(type,nx,ny,i+1,j,k+1)  ];
                                elem->set_node(6) = nodes[idx(type,nx,ny,i+1,j+1,k+1)];
                                elem->set_node(7) = nodes[idx(type,nx,ny,i,j+1,k+1)  ];
                                
                                if (k == 0)
                                    boundary_info.add_side(elem, 0, 0);
                                
                                if (k == (nz-1))
                                    boundary_info.add_side(elem, 5, 5);
                                
                                if (j == 0)
                                    boundary_info.add_side(elem, 1, 1);
                                
                                if (j == (ny-1))
                                    boundary_info.add_side(elem, 3, 3);
                                
                                if (i == 0)
                                    boundary_info.add_side(elem, 4, 4);
                                
                                if (i == (nx-1))
                                    boundary_info.add_side(elem, 2, 2);
                                
                                if (i==nx*3/10 && j<ny*6/10)
                                    boundary_info.add_side(elem, 2, 6);
                                
                                if (j==ny*6/10 && i>nx*3/10)
                                    boundary_info.add_side(elem, 1, 7);
                            }
                        }
                break;
            }
                
                
            case libMesh::HEX27: {
                
                for (uint_t k=0; k<(2*nz); k += 2)
                    for (uint_t j=0; j<(2*ny); j += 2)
                        for (uint_t i=0; i<(2*nx); i += 2)
                        {
                            libMesh::Elem
                            *elem = mesh.add_elem(libMesh::Elem::build_with_id(libMesh::HEX27, elem_id++).release());
                            
                            elem->set_node(0)  = nodes[idx(type,nx,ny,i,  j,  k)  ];
                            elem->set_node(1)  = nodes[idx(type,nx,ny,i+2,j,  k)  ];
                            elem->set_node(2)  = nodes[idx(type,nx,ny,i+2,j+2,k)  ];
                            elem->set_node(3)  = nodes[idx(type,nx,ny,i,  j+2,k)  ];
                            elem->set_node(4)  = nodes[idx(type,nx,ny,i,  j,  k+2)];
                            elem->set_node(5)  = nodes[idx(type,nx,ny,i+2,j,  k+2)];
                            elem->set_node(6)  = nodes[idx(type,nx,ny,i+2,j+2,k+2)];
                            elem->set_node(7)  = nodes[idx(type,nx,ny,i,  j+2,k+2)];
                            elem->set_node(8)  = nodes[idx(type,nx,ny,i+1,j,  k)  ];
                            elem->set_node(9)  = nodes[idx(type,nx,ny,i+2,j+1,k)  ];
                            elem->set_node(10) = nodes[idx(type,nx,ny,i+1,j+2,k)  ];
                            elem->set_node(11) = nodes[idx(type,nx,ny,i,  j+1,k)  ];
                            elem->set_node(12) = nodes[idx(type,nx,ny,i,  j,  k+1)];
                            elem->set_node(13) = nodes[idx(type,nx,ny,i+2,j,  k+1)];
                            elem->set_node(14) = nodes[idx(type,nx,ny,i+2,j+2,k+1)];
                            elem->set_node(15) = nodes[idx(type,nx,ny,i,  j+2,k+1)];
                            elem->set_node(16) = nodes[idx(type,nx,ny,i+1,j,  k+2)];
                            elem->set_node(17) = nodes[idx(type,nx,ny,i+2,j+1,k+2)];
                            elem->set_node(18) = nodes[idx(type,nx,ny,i+1,j+2,k+2)];
                            elem->set_node(19) = nodes[idx(type,nx,ny,i,  j+1,k+2)];
                            
                            elem->set_node(20) = nodes[idx(type,nx,ny,i+1,j+1,k)  ];
                            elem->set_node(21) = nodes[idx(type,nx,ny,i+1,j,  k+1)];
                            elem->set_node(22) = nodes[idx(type,nx,ny,i+2,j+1,k+1)];
                            elem->set_node(23) = nodes[idx(type,nx,ny,i+1,j+2,k+1)];
                            elem->set_node(24) = nodes[idx(type,nx,ny,i,  j+1,k+1)];
                            elem->set_node(25) = nodes[idx(type,nx,ny,i+1,j+1,k+2)];
                            elem->set_node(26) = nodes[idx(type,nx,ny,i+1,j+1,k+1)];
                            
                            if (k == 0)
                                boundary_info.add_side(elem, 0, 0);
                            
                            if (k == 2*(nz-1))
                                boundary_info.add_side(elem, 5, 5);
                            
                            if (j == 0)
                                boundary_info.add_side(elem, 1, 1);
                            
                            if (j == 2*(ny-1))
                                boundary_info.add_side(elem, 3, 3);
                            
                            if (i == 0)
                                boundary_info.add_side(elem, 4, 4);
                            
                            if (i == 2*(nx-1))
                                boundary_info.add_side(elem, 2, 2);

                            if (i==nx*4/10)
                                boundary_info.add_side(elem, 2, 6);
                            
                            if (j==ny*6/10)
                                boundary_info.add_side(elem, 1, 7);
                        }
                break;
            }
                
            default:
                Assert0(false, "ERROR: Unrecognized 3D element type.");
        }
        
        // Add sideset names to boundary info (Z axis out of the screen)
        boundary_info.sideset_name(0) = "back";
        boundary_info.sideset_name(1) = "bottom";
        boundary_info.sideset_name(2) = "right";
        boundary_info.sideset_name(3) = "top";
        boundary_info.sideset_name(4) = "left";
        boundary_info.sideset_name(5) = "front";
        boundary_info.sideset_name(6) = "facing_right";
        boundary_info.sideset_name(7) = "facing_down";

        // Add nodeset names to boundary info
        boundary_info.nodeset_name(0) = "back";
        boundary_info.nodeset_name(1) = "bottom";
        boundary_info.nodeset_name(2) = "right";
        boundary_info.nodeset_name(3) = "top";
        boundary_info.nodeset_name(4) = "left";
        boundary_info.nodeset_name(5) = "front";

        // Done building the mesh.  Now prepare it for use.
        mesh.prepare_for_use ();
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
        nx_divs = c.input("nx_divs", "number of elements along x-axis", 10),
        ny_divs = c.input("ny_divs", "number of elements along y-axis", 10),
        nz_divs = c.input("nz_divs", "number of elements along z-axis", 10);
        
        if (nx_divs%10 != 0 || ny_divs%10 != 0)
            Error(false, "number of divisions in x and y must be multiples of 10");
        
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
        build_cube(mesh,
                   nx_divs, ny_divs, nz_divs,
                   length,
                   height,
                   width,
                   e_type);
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
        
        // iterate over all the element values
        libMesh::MeshBase::const_node_iterator
        it  = c.mesh->local_nodes_begin(),
        end = c.mesh->local_nodes_end();
        
        //
        // maximum number of dvs is the number of nodes on the level set function
        // mesh. We will evaluate the actual number of dvs
        //

        for ( ; it!=end; it++) {
            
            const libMesh::Node& n = **it;
            
            dof_id = n.dof_number(sys_num, 0, 0);
            
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
        
        dvs.synchronize();
        c.rho_sys->solution->close();
    }
};

}  // namespace Generation
}  // namespace Mesh
}  // namespace MAST


#endif // __mast_mesh_generation_bracket_3d_h__
