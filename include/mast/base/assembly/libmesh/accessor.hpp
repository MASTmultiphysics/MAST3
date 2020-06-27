
#ifndef __mast_libmesh_accessor_h__
#define __mast_libmesh_accessor_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

// libMesh includes
#include <libmesh/system.h>
#include <libmesh/dof_map.h>


namespace MAST {
namespace Base {
namespace Assembly {
namespace libMeshWrapper {

template <typename ScalarType, typename VecType>
class Accessor {

public:

    Accessor(const libMesh::System& sys, const VecType& vec):
    _sys      (&sys),
    _vec      (&vec)
    { }

    Accessor():
    _sys      (nullptr),
    _vec      (nullptr)
    { }

    inline void set_system(libMesh::System& sys) { _sys = &sys;}
    inline void set_vec(const VecType& vec) { _vec = &vec;}
    inline const std::vector<libMesh::dof_id_type>& dof_indices() const {return _dof_ids;}
    inline std::vector<libMesh::dof_id_type>& dof_indices() {return _dof_ids;}
    inline uint_t n_dofs() const { return _dof_ids.size();}
    
    inline ScalarType operator() (uint_t i) const {
        
        Assert2(i < _dof_ids.size(),
                i, _dof_ids.size(),
                "Invalid element degree-of-freedom index");
        
        return (*_vec)(_dof_ids[i]);
    }

    inline void init(const libMesh::Elem& e) {
        
        _dof_ids.clear();
        _sys->get_dof_map().dof_indices (&e, _dof_ids);
    }
    
private:
    
    const libMesh::System              *_sys;
    const VecType                      *_vec;
    std::vector<libMesh::dof_id_type>   _dof_ids;
};

} // namespace libMesh
} // namespace Assembly
} // namespace Base
} // namespace MAST

#endif // __mast_libmesh_accessor_h__

