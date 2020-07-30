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


#ifndef __mast__libmesh_geometric_filter_h__
#define __mast__libmesh_geometric_filter_h__


// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/numerics/utility.hpp>
#include <mast/optimization/design_parameter_vector.hpp>
#include <mast/mesh/libmesh/geometric_filter_augment_send_list.hpp>

// libMesh includes
#include "libmesh/system.h"
#include "libmesh/node.h"
#include "libmesh/elem.h"
#include "libmesh/mesh_base.h"
#include "libmesh/numeric_vector.h"
#ifdef LIBMESH_HAVE_NANOFLANN
#include "libmesh/nanoflann.hpp"
#endif



namespace MAST {
namespace Mesh {
namespace libMeshWrapper {


/*!
 *   Creates a geometric filter for the location-based design variables, for example density and
 *   level-set function parmaters in topology optimization.
 */
class GeometricFilter {
    
public:
    
    /*!
     *   \param sys
     *   \param radius geometric filter radius
     */
    GeometricFilter(libMesh::System         &sys,
                    const real_t            radius):
    _system            (sys),
    _radius            (radius),
    _fe_size           (0.),
    _augment_send_list (nullptr) {
        
        Assert1(radius > 0., radius,
                "geometric filter radius must be greater than 0.");
        
#ifdef LIBMESH_HAVE_NANOFLANN
        _init2();  // KD-tree search using NanoFlann
#else
        _init(); // linear filter search
#endif
        
        // now initialize and attach sendlist to the dofmap
        _augment_send_list =
        new MAST::Mesh::libMeshWrapper::GeometricFilterAugmentSendList(_forward_send_list);
        
        _system.get_dof_map().attach_extra_send_list_object(*_augment_send_list);
    }
    
    
    virtual ~GeometricFilter() {
        
        if (_augment_send_list) delete _augment_send_list;
    }
    
    /*!
     *   computes the filtered output from the provided input.
     */
    inline void
    compute_filtered_values
    (const MAST::Optimization::DesignParameterVector<real_t> &dvs,
     const libMesh::NumericVector<real_t>            &input,
     libMesh::NumericVector<real_t>                  &output,
     bool                                            close_vec) const {
        
        Assert2(input.size() == _system.n_dofs(),
                input.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        Assert2(output.size() == _system.n_dofs(),
                output.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        
        output.zero();
        
        std::vector<real_t> input_vals(input.size(), 0.);
        input.localize(input_vals);

        const uint_t
        first_local_dof = _system.get_dof_map().first_dof(_system.comm().rank()),
        last_local_dof  = _system.get_dof_map().end_dof(_system.comm().rank());

        std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>::const_iterator
        map_it   = _filter_map.begin(),
        map_end  = _filter_map.end();
        
        for ( ; map_it != map_end; map_it++) {
            
            if (map_it->first >= first_local_dof &&
                map_it->first <  last_local_dof) {
                
                std::vector<std::pair<uint_t, real_t>>::const_iterator
                vec_it  = map_it->second.begin(),
                vec_end = map_it->second.end();
                
                for ( ; vec_it != vec_end; vec_it++) {
                    if (map_it->first >= input.first_local_index() &&
                        map_it->first <  input.last_local_index()) {
                        
                        if (dvs.is_design_parameter_index(map_it->first))
                            output.add(map_it->first, input_vals[vec_it->first] * vec_it->second);
                        else
                            output.set(map_it->first, input_vals[map_it->first]);
                    }
                }
            }
        }
        
        if (close_vec)
            output.close();
    }
    
    /*!
     *  for large problems it is more efficient to specify only the non-zero entries in the input vector in
     *  \p nonzero_vals. Here, \p output is expected to be of type SERIAL vector. All ranks in the
     *  communicator will perform the same operaitons and provide an identical \p output vector.
     *  If \p close_vector is \p true then \p output.close() will be called in this
     *  routines, otherwise not.
     */
    template <typename ScalarType, typename VecType>
    inline void
    compute_filtered_values
    (const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
     const std::map<uint_t, ScalarType>              &nonzero_vals,
     VecType                                         &output) const {
        
        Assert2(output.size() == _system.n_dofs(),
                output.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        Assert2(output.type() == libMesh::SERIAL,
                output.type(), libMesh::SERIAL,
                "Incompatible vector");
        
        MAST::Numerics::Utility::setZero(output);
        
        const uint_t
        first_local_dof = _system.get_dof_map().first_dof(_system.comm().rank()),
        last_local_dof  = _system.get_dof_map().end_dof(_system.comm().rank());

        std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>::const_iterator
        map_it   = _filter_map.begin(),
        map_end  = _filter_map.end();
        
        for ( ; map_it != map_end; map_it++) {
            
            if (map_it->first >= first_local_dof &&
                map_it->first <  last_local_dof) {
                
                std::vector<std::pair<uint_t, real_t>>::const_iterator
                vec_it  = map_it->second.begin(),
                vec_end = map_it->second.end();
                
                for ( ; vec_it != vec_end; vec_it++) {
                    if (nonzero_vals.count(vec_it->first)) {
                        
                        if (dvs.is_design_parameter(map_it->first))
                            MAST::Numerics::Utility::add
                            (output, map_it->first, nonzero_vals[vec_it->first] * vec_it->second);
                        else
                            MAST::Numerics::Utility::set
                            (output, map_it->first, nonzero_vals[map_it->first]);
                    }
                }
            }
        }
    }
    
    
    template <typename ScalarType,
              typename Vec1Type,
              typename Vec2Type>
    inline void
    compute_filtered_values
    (const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
     const Vec1Type       &input,
     Vec2Type             &output) const {
        
        Assert2(input.size() == _system.n_dofs(),
                input.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        Assert2(output.size() == _system.n_dofs(),
                output.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        
        MAST::Numerics::Utility::setZero(output);
        
        const uint_t
        first_local_dof = _system.get_dof_map().first_dof(_system.comm().rank()),
        last_local_dof  = _system.get_dof_map().end_dof(_system.comm().rank());

        std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>::const_iterator
        map_it   = _filter_map.begin(),
        map_end  = _filter_map.end();
        
        for ( ; map_it != map_end; map_it++) {
            
            // The forward map only processes the local dofs.
            if (map_it->first >= first_local_dof &&
                map_it->first <  last_local_dof) {
                
                std::vector<std::pair<uint_t, real_t>>::const_iterator
                vec_it  = map_it->second.begin(),
                vec_end = map_it->second.end();
                
                for ( ; vec_it != vec_end; vec_it++) {
                    if (dvs.is_design_parameter_index(map_it->first))
                        MAST::Numerics::Utility::add
                        (output, map_it->first,
                         MAST::Numerics::Utility::get(input, vec_it->first) * vec_it->second);
                    else
                        MAST::Numerics::Utility::set
                        (output, map_it->first,
                         MAST::Numerics::Utility::get(input, vec_it->first));
                }
            }
        }
    }
    
    /*!
     *  Applies the reverse map, which is used for sensitivity analysis by first computing the sensitivty wrt filtered coefficients
     *  and then using the columns of the filter coefficient mattix to compute the sensitivity of unfiltered coefficients.
     */
    template <typename ScalarType,
              typename Vec1Type,
              typename Vec2Type>
    inline void
    compute_reverse_filtered_values
    (const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
     const Vec1Type       &input,
     Vec2Type             &output) const {
        
        Assert2(input.size() == _system.n_dofs(),
                input.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        Assert2(output.size() == _system.n_dofs(),
                output.size(), _system.n_dofs(),
                "Incompatible vector sizes");
        
        MAST::Numerics::Utility::setZero(output);
        
        const libMesh::DofMap
        &dof_map = _system.get_dof_map();

        const uint_t
        first_local_dof = _system.get_dof_map().first_dof(_system.comm().rank()),
        last_local_dof  = _system.get_dof_map().end_dof(_system.comm().rank());
        
        std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>::const_iterator
        map_it   = _reverse_map.begin(),
        map_end  = _reverse_map.end();
        
        for ( ; map_it != map_end; map_it++) {
            
            // The reverse map requires processing of dofs that are local and in the send
            // list. Hence, we use dof_map to check if the dof falls in this category.
            if (dof_map.semilocal_index(map_it->first)) {
                
                std::vector<std::pair<uint_t, real_t>>::const_iterator
                vec_it  = map_it->second.begin(),
                vec_end = map_it->second.end();
                
                for ( ; vec_it != vec_end; vec_it++) {
                    if (dvs.is_design_parameter_index(map_it->first))
                        MAST::Numerics::Utility::add
                        (output, map_it->first,
                         MAST::Numerics::Utility::get(input, vec_it->first) * vec_it->second);
                    else
                        MAST::Numerics::Utility::set
                        (output, map_it->first,
                         MAST::Numerics::Utility::get(input, vec_it->first));
                }
            }
        }
    }

    
    /*!
     *   function identifies if the given element is within the domain of
     *   influence of this specified level set design variable. Currently,
     *   this is identified based on the filter radius, the distance of
     *   element nodes from the specified level set design variable location
     *   and the element sizes.
     */
    inline bool if_elem_in_domain_of_influence(const libMesh::Elem& elem,
                                               const libMesh::Node& node) const {
        
        real_t
        d    = 1.e12; // arbitrarily large value to initialize the search
        
        libMesh::Point
        pt;
        
        // first get the smallest distance from the node to the element nodes
        for (uint_t i=0; i<elem.n_nodes(); i++) {
            pt  = elem.point(i);
            pt -= node;
            
            if (pt.norm() < d)
                d = pt.norm();
        }
        
        // if this distance is outside the domain of influence, then this
        // element is not influenced by the design variable
        return (d>_radius+_fe_size);
    }
    
    
    
    /*!
     *  prints the filter data.
     */
    template <typename ScalarType>
    inline void
    print(const MAST::Optimization::DesignParameterVector<ScalarType> &dvs,
          std::ostream         &o) const {
        
        o << "Filter radius: " << _radius << std::endl;
        
        o
        << std::setw(20) << "Filtered ID"
        << std::setw(20) << "Dependent Vars" << std::endl;
        
        std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>::const_iterator
        map_it   = _filter_map.begin(),
        map_end  = _filter_map.end();
        
        for ( ; map_it != map_end; map_it++) {
            
            o
            << std::setw(20) << map_it->first;
            
            std::vector<std::pair<uint_t, real_t>>::const_iterator
            vec_it  = map_it->second.begin(),
            vec_end = map_it->second.end();
            
            for ( ; vec_it != vec_end; vec_it++) {
                
                if (dvs.is_design_parameter(map_it->first))
                    o
                    << " : " << std::setw(8) << vec_it->first
                    << " (" << std::setw(8) << vec_it->second << " )";
                else
                    std::cout << " : " << map_it->first;
            }
            std::cout << std::endl;
        }
    }
    
    
private:
    
#ifdef LIBMESH_HAVE_NANOFLANN
    // Nanoflann uses "duck typing" to allow users to define their own adaptors...
    template <uint_t Dim>
    class NanoflannMeshAdaptor
    {
    private:
        // Constant reference to the Mesh we are adapting for use in Nanoflann
        const libMesh::MeshBase & _mesh;
        
    public:
        NanoflannMeshAdaptor (const libMesh::MeshBase & mesh) :
        _mesh(mesh)
        {}
        
        /**
         * libMesh \p Point coordinate type
         */
        typedef real_t coord_t;
        
        /**
         * Must return the number of data points
         */
        inline size_t
        kdtree_get_point_count() const { return _mesh.n_nodes(); }
        
        /**
         * Returns the distance between the vector "p1[0:size-1]"
         * and the data point with index "idx_p2" stored in _mesh
         */
        inline coord_t
        kdtree_distance(const coord_t * p1,
                        const size_t idx_p2,
                        size_t size) const {
            
            Assert2(size == Dim, size, Dim, "Incompatible dimension");
            
            // Construct a libmesh Point object from the input coord_t.  This
            // assumes LIBMESH_DIM==3.
            libMesh::Point point1(p1[0],
                                  size > 1 ? p1[1] : 0.,
                                  size > 2 ? p1[2] : 0.);
            
            // Get the referred-to point from the Mesh
            const libMesh::Point & point2 = _mesh.point(idx_p2);
            
            // Compute Euclidean distance
            return (point1 - point2).norm_sq();
        }
        
        /**
         * Returns the dim'th component of the idx'th point in the class:
         * Since this is inlined and the "dim" argument is typically an immediate value, the
         *  "if's" are actually solved at compile time.
         */
        inline coord_t
        kdtree_get_pt(const size_t idx, int dim) const
        {
            Assert2(dim < (int) Dim, dim, (int) Dim,
                    "Incompatible dimension");
            Assert2(idx < _mesh.n_nodes(), idx, _mesh.n_nodes(),
                    "Invalid node index");
            Assert1(dim < 3, dim, "Invalid dimension");
            
            return _mesh.point(idx)(dim);
        }
        
        /**
         * Optional bounding-box computation: return false to default to a standard bbox computation loop.
         * Return true if the BBOX was already computed by the class and returned in "bb" so it can be
         * avoided to redo it again. Look at bb.size() to find out the expected dimensionality
         * (e.g. 2 or 3 for point clouds)
         */
        template <class BBOX>
        inline bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
    };
    
    inline void _init2() {
        
        libMesh::MeshBase& mesh = _system.get_mesh();
        
        // currently implemented for replicated mesh
        Assert0(mesh.is_replicated(), "Method implemented only for replicated mesh");
        
        // Loop over nodes to try and detect duplicates.  We use nanoflann
        // for this, inspired by
        // https://gist.github.com/jwpeterson/7a36f9f794df67d51126#file-detect_slit-cc-L65
        // which was inspired by nanoflann example in libMesh source:
        // contrib/nanoflann/examples/pointcloud_adaptor_example.cpp
        
        // Declare a type templated on NanoflannMeshAdaptor
        typedef nanoflann::L2_Simple_Adaptor<real_t, NanoflannMeshAdaptor<3> > adatper_t;
        
        // Declare a KDTree type based on NanoflannMeshAdaptor
        typedef nanoflann::KDTreeSingleIndexAdaptor<adatper_t, NanoflannMeshAdaptor<3>, 3> kd_tree_t;
        
        // Build adaptor and tree objects
        NanoflannMeshAdaptor<3> mesh_adaptor(mesh);
        kd_tree_t kd_tree(3, mesh_adaptor, nanoflann::KDTreeSingleIndexAdaptorParams(/*max leaf=*/10));
        
        // Construct the tree
        kd_tree.buildIndex();
        
        real_t
        d_12 = 0.,
        sum  = 0.;

        std::set<uint_t> send_list;
        
        const libMesh::DofMap
        &dof_map = _system.get_dof_map();

        const uint_t
        first_local_dof = _system.get_dof_map().first_dof(_system.comm().rank()),
        last_local_dof  = _system.get_dof_map().end_dof(_system.comm().rank());

        uint_t
        dof_1,
        dof_2;
        
        libMesh::MeshBase::const_node_iterator
        node_it      =  mesh.nodes_begin(),
        node_end     =  mesh.nodes_end();
        
        // For every node in the mesh, search the KDtree and find any
        // nodes at _radius distance from the current
        // node being searched... this will be added to the .
        for (; node_it != node_end; node_it++) {
            
            const libMesh::Node* node = *node_it;
            
            dof_1 = node->dof_number(_system.number(), 0, 0);

            // only local dofs are processed.
            if (/*dof_1 >= first_local_dof &&
                dof_1 <  last_local_dof*/
                dof_map.semilocal_index(dof_1)) {
                
                real_t query_pt[3] = {(*node)(0), (*node)(1), (*node)(2)};
                
                std::vector<std::pair<size_t, real_t>>
                indices_dists;
                nanoflann::RadiusResultSet<real_t, size_t>
                resultSet(_radius*_radius, indices_dists);
                
                kd_tree.findNeighbors(resultSet, query_pt, nanoflann::SearchParams());
                
                sum       = 0.;
                
                for (unsigned r=0; r<indices_dists.size(); ++r) {
                    
                    d_12 = std::sqrt(indices_dists[r].second);
                    
                    // the distance of this node should be less than or equal to the
                    // specified search radius
                    Assert2(d_12 <= _radius, d_12, _radius,
                            "Node distance must be <= search radius");
                    
                    sum  += _radius - d_12;
                    dof_2 = mesh.node_ptr(indices_dists[r].first)->dof_number(_system.number(), 0, 0);
                    
                    _filter_map[dof_1].push_back(std::pair<uint_t, real_t>(dof_2, _radius - d_12));

                    // add this dof to the local send list
                    if (dof_2 < first_local_dof ||
                        dof_2 >= last_local_dof)
                        send_list.insert(dof_2);
                }
                
                Assert1(sum > 0., sum, "Weight must be > 0.");
                
                // with the coefficients computed for dof_1, divide each coefficient
                // with the sum
                std::vector<std::pair<uint_t, real_t>>& vec = _filter_map[dof_1];
                for (uint_t i=0; i<vec.size(); i++) {
                    
                    vec[i].second /= sum;
                    Assert1(vec[i].second <= 1., vec[i].second,
                            "Normalized weight must be <= 1.");
                }
            }
        }

        // now prepare the reverse map. The send list is sorted for later use.
        std::set<uint_t>::const_iterator
        s_it  = send_list.begin(),
        s_end = send_list.end();
        
        _forward_send_list.reserve(send_list.size());
        for ( ; s_it != s_end; s_it++) _forward_send_list.push_back(*s_it);
        
        _init_reverse_map(_filter_map, _reverse_map);
        
        // compute the largest element size
        libMesh::MeshBase::const_element_iterator
        e_it          = mesh.elements_begin(),
        e_end         = mesh.elements_end();
        
        for ( ; e_it != e_end; e_it++) {
            const libMesh::Elem* e = *e_it;
            d_12 = e->hmax();
            
            if (_fe_size < d_12)
                _fe_size = d_12;
        }
    }
#endif
    
    
    /*!
     *   initializes the algebraic data structures
     */
    inline void _init() {
        
        Assert0(!_filter_map.size(), "Filter already initialized");
        
        libMesh::MeshBase& mesh = _system.get_mesh();
        
        // currently implemented for replicated mesh
        Assert0(mesh.is_replicated(), "Function implemented only for replicated mesh");
        
        // iterate over all nodes to compute the
        libMesh::MeshBase::const_node_iterator
        node_it_1    =  mesh.nodes_begin(),
        node_it_2    =  mesh.nodes_begin(),
        node_end     =  mesh.nodes_end();
        
        libMesh::Point
        d;
        
        real_t
        d_12 = 0.,
        sum  = 0.;
        
        std::set<uint_t> send_list;
        
        const libMesh::DofMap
        &dof_map = _system.get_dof_map();

        const uint_t
        first_local_dof = _system.get_dof_map().first_dof(_system.comm().rank()),
        last_local_dof  = _system.get_dof_map().end_dof(_system.comm().rank());

        uint_t
        dof_1,
        dof_2;
        
        for ( ; node_it_1 != node_end; node_it_1++) {
            
            dof_1 = (*node_it_1)->dof_number(_system.number(), 0, 0);

            // only local dofs are processed.
            if (/*dof_1 >= first_local_dof &&
                dof_1 <  last_local_dof*/
                dof_map.semilocal_index(dof_1)) {
                
                node_it_2 = mesh.nodes_begin();
                sum       = 0.;
                
                for ( ; node_it_2 != node_end; node_it_2++) {
                    
                    // compute the distance between the two nodes
                    d    = (**node_it_1) - (**node_it_2);
                    d_12 = d.norm();
                    
                    // if the nodes is within the filter radius, add it to the map
                    if (d_12 <= _radius) {
                        
                        sum  += _radius - d_12;
                        dof_2 = (*node_it_2)->dof_number(_system.number(), 0, 0);
                        
                        _filter_map[dof_1].push_back(std::pair<uint_t, real_t>(dof_2, _radius - d_12));
                        
                        // add this dof to the local send list if it is not a local dof
                        if (dof_2 < first_local_dof ||
                            dof_2 >= last_local_dof)
                            send_list.insert(dof_2);
                    }
                }
                
                Assert1(sum > 0., sum, "Weight must be > 0.");
                
                // with the coefficients computed for dof_1, divide each coefficient
                // with the sum
                std::vector<std::pair<uint_t, real_t>>& vec = _filter_map[dof_1];
                for (uint_t i=0; i<vec.size(); i++) {
                    
                    vec[i].second /= sum;
                    Assert1(vec[i].second <= 1., vec[i].second,
                            "Normalized weight must be <= 1.");
                }
            }
        }
        
        // now prepare the reverse map. The send list is sorted for later use.
        std::set<uint_t>::const_iterator
        s_it  = send_list.begin(),
        s_end = send_list.end();
        
        _forward_send_list.reserve(send_list.size());
        for ( ; s_it != s_end; s_it++) _forward_send_list.push_back(*s_it);
        
        _init_reverse_map(_filter_map, _reverse_map);

        // compute the largest element size
        libMesh::MeshBase::const_element_iterator
        e_it          = mesh.elements_begin(),
        e_end         = mesh.elements_end();
        
        for ( ; e_it != e_end; e_it++) {
            const libMesh::Elem* e = *e_it;
            d_12 = e->hmax();
            
            if (_fe_size < d_12)
                _fe_size = d_12;
        }
        
    }
    
    
    inline void
    _init_reverse_map(const std::map<uint_t, std::vector<std::pair<uint_t, real_t>>> &forward_map,
                      std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>       &reverse_map) {
        
        // now prepare the reverse map
        std::map<uint_t, std::vector<std::pair<uint_t, real_t>>>::const_iterator
        it   =  forward_map.begin(),
        end  =  forward_map.end();
        
        for ( ; it!=end; it++) {
            
            const std::vector<std::pair<uint_t, real_t>>
            &vec = it->second;
            
            for (uint_t i=0; i<vec.size(); i++)
                reverse_map[vec[i].first].push_back(std::pair<uint_t, real_t>(it->first, vec[i].second));
        }
    }
    
    /*!
     *   system on which the level set discrete function is defined
     */
    libMesh::System& _system;
    
    /*!
     *   radius of the filter.
     */
    real_t _radius;
    
    
    /*!
     *   largest element size in the level set mesh
     */
    real_t _fe_size;
    
    /*!
     *  pointer to an object that appends the sendlist for dofmap to localize the dofs needed for local
     *  computations.
     */
    MAST::Mesh::libMeshWrapper::GeometricFilterAugmentSendList *_augment_send_list;

    /*!
     *   Algebraic relation between filtered level set values and the
     *   design variables \f$ \tilde{\phi}_i = B_{ij} \phi_j \f$
     */
    std::map<uint_t, std::vector<std::pair<uint_t, real_t>>> _filter_map;
    
    /*!
     * this map stores the columns of the matrix, which is required for sensitivity analysis
     */
    std::map<uint_t, std::vector<std::pair<uint_t, real_t>>> _reverse_map;
    
    /*!
     *   vector of dof ids that the current processor depends on.
     */
    std::vector<uint_t> _forward_send_list;
    
};


} // namespace libMeshWrapper
} // namespace Mesh
} // namespace MAST


#endif // __mast__libmesh_geometric_filter_h__