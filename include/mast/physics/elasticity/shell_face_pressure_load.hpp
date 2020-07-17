
#ifndef __mast_shell_face_pressure_load_h__
#define __mast_shell_face_pressure_load_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>

namespace MAST {
namespace Physics {
namespace Elasticity {

template <typename FEVarType,
          typename PressureFieldType,
          typename ContextType>
class ShellFacePressureLoad {

public:

    using scalar_t         = typename FEVarType::scalar_t;
    using basis_scalar_t   = typename FEVarType::fe_shape_deriv_t::scalar_t;
    using vector_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, 1>;
    using matrix_t         = typename Eigen::Matrix<scalar_t, Eigen::Dynamic, Eigen::Dynamic>;

    ShellFacePressureLoad():
    _pressure      (nullptr),
    _fe_var_data   (nullptr),
    _displ_index   (-1)
    { }
    
    virtual ~ShellFacePressureLoad() { }
    
    inline void set_pressure(const PressureFieldType& p) { _pressure = &p;}
    
    /*!
     *  This assumes that the face is oriented along the \a x-axis (+ve or -ve) for a 1D element
     *  and in the \a xy-plane for a 2D element.
     *
     *  For a 1D element the surface normal is assumed to be along the +ve \a y-axis so that a
     *  positive pressure results in a force along this direction.
     *
     *  For a 2D element the surface normal is assumed to be along the +ve \a z-axis so that a
     *  positive pressure results in a force along this direction.
     *
     *  \p displ_index is the component of variable in \p fe that serves as the transverse displacement
     *   used to compute work done. 
     */
    inline void set_fe_var_data(const FEVarType& fe,
                                const uint_t     displ_index) {
        
        _fe_var_data = &fe;
        _displ_index = displ_index;
    }

    inline uint_t n_dofs() const {

        Assert0(_fe_var_data, "FE data not initialized.");

        return _fe_var_data->get_fe_shape_data().n_basis();
    }

    inline void compute(ContextType& c,
                        vector_t& res,
                        matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_pressure, "Pressure not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            scalar_t p = _pressure->value(c);
            
            for (uint_t k=0; k<fe.n_basis(); k++)
                res(k) -= fe.detJxW(i) * fe.phi(i, k) * p;
        }
    }
    
    
    template <typename ScalarFieldType>
    inline void derivative(ContextType& c,
                           const ScalarFieldType& f,
                           vector_t& res,
                           matrix_t* jac = nullptr) const {
        
        Assert0(_fe_var_data, "FE data not initialized.");
        Assert0(_pressure, "Pressure not initialized");
        
        const typename FEVarType::fe_shape_deriv_t
        &fe = _fe_var_data->get_fe_shape_data();
        
        for (uint_t i=0; i<fe.n_q_points(); i++) {
            
            c.qp       = i;
            scalar_t p = _pressure->derivative(c, f);
            
            for (uint_t k=0; k<fe.n_basis(); k++)
                res(k) -= fe.detJxW(i) * fe.phi(i, k) * p;
        }
    }
    
private:

    const PressureFieldType    *_pressure;
    const FEVarType            *_fe_var_data;
    uint_t                      _displ_index;
};


} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_shell_face_pressure_load_h__
