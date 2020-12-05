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

#ifndef __mast_material_point_storage_h__
#define __mast_material_point_storage_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>


namespace MAST {
namespace Base {

template <typename ScalarType, uint_t NComponents>
class MaterialPointStorage {
    
public:
  
    using scalar_t = ScalarType;
    using view_t   = Eigen::Map<const typename Eigen::Matrix<scalar_t, NComponents, 1>>;

    
    MaterialPointStorage(MPI_Comm comm):
    _initialized  (false),
    _n_points     (0)
    _data         (nullptr),
    _comm         (comm)
    { }
    
    virtual ~MaterialPointStorage() {

        if (_data) delete _data;
    }

    /*!
     * initializes the storage for \p n_points with \p NComponents data on each point
     */
    inline void init(uint_t n_points) {
        
        Assert0(!_initialized, "Object already initialized");
        
        _n_points = n_points;
        
        _data     = scalar_t[n_comp*NComponents];
        
        _initialized = true;
        
        this->zero();
    }

    
    inline scalar_t* data() {
        
        Assert0(_initialized, "Object must be initialized");
        return _data;
    }

    
    inline const scalar_t* data() const {
        
        Assert0(_initialized, "Object must be initialized");
        return _data;
    }


    inline void zero() {
        
        Assert0(_initialized, "Object must be initialized");
        std::fill(_data, _data+_n_points*NComponents, 0.);
    }
    
    /*!
     * @returns a \p view_t object for the data on point \p pt
     */
    inline view_t data(uint_t pt) {
        
        Assert0(_initialized, "Object must be initialized");
        return view_t(_data + NComponents*pt, NComponents, 1);
    }

    
private:
    
    bool         _initialized;
    uint_t       _n_points;
    scalar_t    *_data;
    MPI_Comm     _comm;
};

}  // namespace Base
}  // namespace MAST

#endif // __mast_material_point_storage_h__
