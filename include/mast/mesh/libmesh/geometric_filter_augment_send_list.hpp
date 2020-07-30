
#ifndef __mast_geometric_filter_augment_send_list_h__
#define __mast_geometric_filter_augment_send_list_h__


// libMesh includes
#include <libmesh/dof_map.h>


namespace MAST {
namespace Mesh {
namespace libMeshWrapper {


class GeometricFilterAugmentSendList:
public libMesh::DofMap::AugmentSendList {

public:
    
    GeometricFilterAugmentSendList(const std::vector<uint_t>& v):
    _list  (v)
    { }

    virtual ~GeometricFilterAugmentSendList() { }

    virtual void
    augment_send_list(std::vector<libMesh::dof_id_type>& send_list) override {
        
        for (uint_t i=0; i<_list.size(); i++)
            send_list.push_back(_list[i]);
    }

private:
    
    const std::vector<uint_t>& _list;
};

} // namespace libMeshWrapper
} // namespace Mesh
} // namespace MAST


#endif // __mast_geometric_filter_augment_send_list_h__
