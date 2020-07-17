
#ifndef __mast_plate_bending_section_property_h__
#define __mast_plate_bending_section_property_h__

// MAST includes
#include <mast/base/mast_data_types.h>


namespace MAST {
namespace Physics {
namespace Elasticity {

template <typename ScalarType,
          typename MaterialType,
          typename ThicknessType,
          typename ContextType>
class PlateBendingSectionProperty {
public:
    
    using material_scalar_t  = typename MaterialType::scalar_t;
    using thickness_scalar_t = typename ThicknessType::scalar_t;
    using scalar_t    = typename MAST::DeducedScalarType<typename MAST::DeducedScalarType<material_scalar_t, thickness_scalar_t>::type, ScalarType>::type;
    using inplane_value_t    = typename Eigen::Matrix<scalar_t, 3, 3>;
    using shear_value_t      = typename Eigen::Matrix<scalar_t, 2, 2>;

    PlateBendingSectionProperty():
    _material     (nullptr),
    _th           (nullptr) { }
    
    virtual ~PlateBendingSectionProperty() {}
    
    inline void set_material_and_thickness(const MaterialType& material,
                                           const ThicknessType& th) {
        
        _material  = &material;
        _th        = &th;
    }
    
    
    inline const MaterialType& get_material() const {return *_material;}
    
    inline const ThicknessType& get_thickness() const {return *_th;}
    
    inline void inplane_value(ContextType     &c,
                              inplane_value_t &m) const {

        Assert0(_material && _th, "Material and thickness not provided");

        _material->value(c, m);
        m *= std::pow(_th->value(c), 3)/12.;
     }
    
    
    template <typename ScalarFieldType>
    inline void inplane_derivative(ContextType           &c,
                                   const ScalarFieldType &f,
                                   inplane_value_t       &m) const {

        Assert0(_material && _th, "Material and thickness not provided");

        typename MaterialType::value_t
        dm;
        
        _material->value(c, m);
        _material->derivative(c, f, dm);
        
        dm *= std::pow(_th->value(c), 3)/12.;
        m  *= std::pow(_th->value(c), 2)/4.*_th->derivative(c, f);
        m  += dm;
    }

    inline void shear_value(ContextType& c,
                            shear_value_t &m) const {

        Assert0(_material && _th, "Material and thickness not provided");

        m.setZero();
        m(0, 0) = m(1, 1) = _material->G(c) * _th->value(c);
     }
    
    
    template <typename ScalarFieldType>
    inline void shear_derivative(ContextType           &c,
                                 const ScalarFieldType &f,
                                 shear_value_t         &m) const {

        Assert0(_material && _th, "Material and thickness not provided");

        m.setZero();
        m(0, 0) = m(1, 1) = (_material->G_derivative(c, f) * _th->value(c) +
                             _material->G(c) * _th->derivative(c, f));
    }

    
private:
    
    const MaterialType*  _material;
    const ThicknessType* _th;
};

} // namespace Elasticity
} // namespace Physics
} // namespace MAST


#endif // __mast_plate_bending_section_property_h__
