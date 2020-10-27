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

#ifndef __mast_mesh_generation_bracket_2d_h__
#define __mast_mesh_generation_bracket_2d_h__

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


struct Bracket2D {
    
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
    
    static const uint_t dim = 2;
    template <typename ScalarType>
    using pressure_t        =  MAST::Mesh::Generation::Bracket2D::Pressure<ScalarType>;
    
    template <typename Context>
    inline real_t
    reference_volume(Context& c) {
        
        real_t
        length  = c.input("length", "length of domain along x-axis", 0.3),
        height  = c.input("height", "length of domain along y-axis", 0.3);
        
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
                        if ( i<=nx/10*4 || j>=ny/10*6) {
                            nodes[node_id] =
                            mesh.add_point(libMesh::Point(static_cast<real_t>(i)/static_cast<real_t>(nx)*length,
                                                          static_cast<real_t>(j)/static_cast<real_t>(ny)*height,
                                                          0.),
                                           n++);
                        }
                        node_id++;
                    }
                
                
                break;
            }

            case libMesh::QUAD9: {

                for (uint_t j=0; j<=(2*ny); j++)
                    for (uint_t i=0; i<=(2*nx); i++) {
                        if ( i<=2*nx/10*4 || j>=2*ny/10*6) {
                            nodes[node_id] =
                            mesh.add_point(libMesh::Point(static_cast<real_t>(i)/static_cast<real_t>(2*nx)*length,
                                                          static_cast<real_t>(j)/static_cast<real_t>(2*ny)*height,
                                                          0.),
                                           n++);
                        }
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
                        if (i < nx*4/10 || j>=ny*6/10) {
                            
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
                            
                            if (i==nx*3/10 && j<ny*6/10)
                                boundary_info.add_side(elem, 1, 4);
                            
                            if (j==ny*6/10 && i>nx*3/10)
                                boundary_info.add_side(elem, 0, 5);
                        }
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
                        
                        if (i==nx*4/10)
                            boundary_info.add_side(elem, 1, 4);
                        
                        if (j==ny*6/10)
                            boundary_info.add_side(elem, 0, 5);
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
        boundary_info.sideset_name(4) = "facing_right";
        boundary_info.sideset_name(5) = "facing_down";

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
        length  = c.input("length", "length of domain along x-axis", 0.3),
        height  = c.input("height", "length of domain along y-axis", 0.3);
        
        uint_t
        nx_divs = c.input("nx_divs", "number of elements along x-axis", 20),
        ny_divs = c.input("ny_divs", "number of elements along y-axis", 20);
        
        if (nx_divs%10 != 0 || ny_divs%10 != 0) libmesh_error();
        
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
                   e_type);
    }
    
    
    /*template <typename Context>
     inline void
     init_level_set_mesh(Context& c,
     libMesh::UnstructuredMesh& mesh) {
     
     real_t
     length  = c.input("length", "length of domain along x-axis", 0.3),
     height  = c.input("height", "length of domain along y-axis", 0.3);
     
     uint_t
     nx_divs = c.input("level_set_nx_divs", "number of elements of level-set mesh along x-axis", 10),
     ny_divs = c.input("level_set_ny_divs", "number of elements of level-set mesh along y-axis", 10);
     
     if (nx_divs%10 != 0 || ny_divs%10 != 0) libmesh_error();
     
     libMesh::ElemType
     e_type  = libMesh::QUAD4;
     
     // initialize the mesh with one element
     libMesh::MeshTools::Generation::build_square(mesh,
     nx_divs, ny_divs,
     0, length,
     0, height,
     e_type);
     
     _delete_elems_from_bracket_mesh(c, mesh);
     }*/
    
    
    
    template <typename Context>
    inline void
    init_analysis_dirichlet_conditions(Context& c) {
        
        c.sys->get_dof_map().add_dirichlet_boundary
        (libMesh::DirichletBoundary({0}, {0, 1}, libMesh::ZeroFunction<real_t>()));
    }
    
    
    
    template <typename ScalarType, typename InitType>
    std::unique_ptr<pressure_t<ScalarType>>
    build_pressure_load(InitType& c) {
        
        real_t
        length      = c.input("length", "length of domain along x-axis", 0.3),
        frac        = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.125),
        p_val       = c.input("pressure", "pressure on side of domain",   5.e7);
        c.p_side_id = 5;
        
        std::unique_ptr<pressure_t<ScalarType>>
        press(new pressure_t<ScalarType>(p_val, length, frac));
        
        return press;
    }
    
    
    /*template <typename Context>
     inline void
     init_level_set_dvs(Context& c) {
     
     libmesh_assert(c._initialized);
     //
     // this assumes that level set is defined using lagrange shape functions
     //
     libmesh_assert_equal_to(c._level_set_fetype.family, libMesh::LAGRANGE);
     
     real_t
     tol           = 1.e-12,
     l_frac        = 0.4,//_input("length_fraction", "fraction of length along x-axis that is in the bracket", 0.4),
     h_frac        = 0.4,//_input( "height_fraction", "fraction of length along y-axis that is in the bracket", 0.4),
     length        = c.input("length", "length of domain along x-axis", 0.3),
     height        = c.input("height", "length of domain along y-axis", 0.3),
     x_lim         = length * l_frac,
     y_lim         = height * (1.-h_frac),
     frac          = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.125),
     filter_radius = c.input("filter_radius", "radius of geometric filter for level set field", 0.015);
     
     uint_t
     dof_id  = 0,
     n_vars  = 0;
     
     real_t
     val     = 0.;
     
     //
     // all ranks will have DVs defined for all variables. So, we should be
     // operating on a replicated mesh
     //
     libmesh_assert(c._level_set_mesh->is_replicated());
     
     std::vector<real_t> local_phi(c._level_set_sys->solution->size());
     c._level_set_sys->solution->localize(local_phi);
     
     //
     // iterate over all the node values
     //
     libMesh::MeshBase::const_node_iterator
     it  = c._level_set_mesh->nodes_begin(),
     end = c._level_set_mesh->nodes_end();
     
     //
     // maximum number of dvs is the number of nodes on the level set function
     // mesh. We will evaluate the actual number of dvs
     //
     c._dv_params.reserve(c._level_set_mesh->n_nodes());
     n_vars = 0;
     
     for ( ; it!=end; it++) {
     
     const libMesh::Node& n = **it;
     
     dof_id                     = n.dof_number(0, 0, 0);
     
     if ((n(1)-filter_radius) <= y_lim &&
     (n(0)+filter_radius) >= length*(1.-frac)) {
     
     //
     // set value at the constrained points to a small positive number
     // material here
     //
     if (dof_id >= c._level_set_sys->solution->first_local_index() &&
     dof_id <  c._level_set_sys->solution->last_local_index())
     c._level_set_sys->solution->set(dof_id, 1.e0);
     }
     else {
     
     std::ostringstream oss;
     oss << "dv_" << n_vars;
     val = local_phi[dof_id];
     
     //
     // on the boundary, set everything to be zero, so that there
     // is always a boundary there that the optimizer can move
     //
     if (n(0) < tol                     ||  // left boundary
     std::fabs(n(0) - length) < tol ||  // right boundary
     std::fabs(n(1) - height) < tol ||  // top boundary
     (n(0) >= x_lim && n(1) <= y_lim)) {
     
     if (dof_id >= c._level_set_sys->solution->first_local_index() &&
     dof_id <  c._level_set_sys->solution->last_local_index())
     c._level_set_sys->solution->set(dof_id, -1.0);
     val = -1.0;
     }
     
     c.dv_params.push_back(std::pair<uint_t, MAST::Parameter*>());
     c.dv_params[n_vars].first  = dof_id;
     c.dv_params[n_vars].second = new MAST::LevelSetParameter(oss.str(), val, &n);
     c.dv_params[n_vars].second->set_as_topology_parameter(true);
     c.dv_dof_ids.insert(dof_id);
     
     n_vars++;
     }
     }
     
     c.set_n_vars(n_vars);
     
     c.level_set_sys->solution->close();
     }*/
    
    
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
        vf            = c.input("volume_fraction",
                                "upper limit for the volume fraction", 0.2);
        
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

        dvs.synchronize(c.rho_sys->get_dof_map());
        c.rho_sys->solution->close();
    }

    
    
    /*template <typename Context>
     inline void
     initialize_level_set_solution(Context& c) {
     
     real_t
     length  = c.input("length", "length of domain along x-axis", 0.3),
     height  = c.input("height", "length of domain along y-axis", 0.3);
     
     uint_t
     nx_h    = c.input("initial_level_set_n_holes_in_x",
     "number of holes along x-direction for initial level-set field", 6),
     ny_h    = c.input("initial_level_set_n_holes_in_y",
     "number of holes along y-direction for initial level-set field", 6),
     nx_m    = c.input("level_set_nx_divs", "number of elements of level-set mesh along x-axis", 10),
     ny_m    = c.input("level_set_ny_divs", "number of elements of level-set mesh along y-axis", 10);
     
     MAST::Examples::LevelSetNucleationFunction
     phi(0., 0., length, height, nx_m, ny_m, nx_h, ny_h);
     
     c._level_set_sys_init->initialize_solution(phi);
     }
     
     class BracketLoad:
     public MAST::FieldFunction<real_t> {
     public:
     BracketLoad(const std::string& nm, real_t p, real_t l1, real_t fraction):
     MAST::FieldFunction<real_t>(nm), _p(p), _l1(l1), _frac(fraction) { }
     ~BracketLoad() {}
     void operator() (const libMesh::Point& p, const real_t t, real_t& v) const {
     if (fabs(p(0) >= _l1*(1.-_frac))) v = _p;
     else v = 0.;
     }
     void derivative(const MAST::FunctionBase& f, const libMesh::Point& p, const real_t t, real_t& v) const {
     v = 0.;
     }
     protected:
     real_t _p, _l1, _frac;
     };*/
    
    
    
};

}  // namespace Generation
}  // namespace Mesh
}  // namespace MAST


#endif // __mast_mesh_generation_bracket_2d_h__
