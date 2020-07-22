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
        else if (c.fe_order > 1 && e_type == libMesh::TRI3)
            e_type = libMesh::TRI6;
        
        //
        // initialize the mesh with one element
        //
        libMesh::MeshTools::Generation::build_square(mesh,
                                                     nx_divs, ny_divs,
                                                     0, length,
                                                     0, height,
                                                     e_type);
        
        _delete_elems_from_bracket_mesh(c, mesh);
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
        l_frac        = 0.4,//_input("length_fraction", "fraction of length along x-axis that is in the bracket", 0.4),
        h_frac        = 0.4,//_input( "height_fraction", "fraction of length along y-axis that is in the bracket", 0.4),
        length        = c.input("length", "length of domain along x-axis", 0.3),
        height        = c.input("height", "length of domain along y-axis", 0.3),
        x_lim         = length * l_frac,
        y_lim         = height * (1.-h_frac),
        frac          = c.input("loadlength_fraction", "fraction of boundary length on which pressure will act", 0.125),
        filter_radius = c.input("filter_radius", "radius of geometric filter for level set field", 0.015),
        rho_min       = c.input("rho_min", "lower limit on density variable", 0.);
        
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
        
        std::vector<real_t> local_phi(c.rho_sys->solution->size());
        c.rho_sys->solution->localize(local_phi);
        
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
                // set value at the constrained points to a small positive number
                // material here
                //
                if (dof_id >= c.rho_sys->solution->first_local_index() &&
                    dof_id <  c.rho_sys->solution->last_local_index())
                    c.rho_sys->solution->set(dof_id, 1.e0);
            }
            else {
                
                val = local_phi[dof_id];
                
                //
                // on the boundary, set everything to be zero, so that there
                // is always a boundary there that the optimizer can move
                //
                if (n(0) < tol                     ||  // left boundary
                    std::fabs(n(0) - length) < tol ||  // right boundary
                    std::fabs(n(1) - height) < tol ||  // top boundary
                    (n(0) >= x_lim && n(1) <= y_lim)) {
                    
                    if (dof_id >= c.rho_sys->solution->first_local_index() &&
                        dof_id <  c.rho_sys->solution->last_local_index())
                        c.rho_sys->solution->set(dof_id, rho_min);
                    val = rho_min;
                }
                
                MAST::Optimization::DesignParameter<ScalarType>
                *dv = new MAST::Optimization::DesignParameter<ScalarType>(val);
                dv->set_point(n(0), n(1), n(2));

                MAST::Base::ParameterData
                &data = dvs.add_parameter(*dv);
                
                data.add<uint>("dof_id") = dof_id;
                //data.add<bool>("topology", true);
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
                
                if (facing_right) mesh.boundary_info->add_side(elem, i, 4);
                if (facing_down) mesh.boundary_info->add_side(elem, i, 5);
            }
        }
        
        mesh.boundary_info->sideset_name(4) = "facing_right";
        mesh.boundary_info->sideset_name(5) = "facing_down";
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
