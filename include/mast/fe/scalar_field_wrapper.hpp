
#ifndef __mast_scalar_field_wrapper_h__
#define __mast_scalar_field_wrapper_h__


namespace MAST {
namespace FEBasis {

template <typename ScalarType, typename FEVarType, typename ContextType>
class ScalarFieldWrapper {

public:
    
    ScalarFieldWrapper():
    _fe              (nullptr),
    _fe_derivative   (nullptr),
    _comp            (0),
    _comp_derivative (0)
    {}
    
    virtual ~ScalarFieldWrapper() { }
    
    inline void set_fe_object_and_component(const FEVarType& fe, uint_t comp) {
        
        _fe   = &fe;
        _comp = comp;
    }

    inline void set_derivative_fe_object_and_component(const FEVarType& fe, uint_t comp) {
        
        _fe_derivative   = &fe;
        _comp_derivative = comp;
    }

    inline ScalarType value(const ContextType& c) {
        
        Assert0(_fe, "Object not initialized");
        _fe->u(c.qp, _comp);
    }

    inline ScalarType value(const ContextType& c) {
        
        Assert0(_fe_derivative, "Object not initialized");
        _fe_derivative->u(c.qp, _comp_derivative);
    }

    
private:
    
    const FEVarType *   _fe;
    const FEVarType *   _fe_derivative;

    uint_t              _comp;
    uint_t              _comp_derivative;

};

} // namespace FEBasis
} // namespace MAST

#endif  // __mast_scalar_field_wrapper_h__
