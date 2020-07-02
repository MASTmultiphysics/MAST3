
#ifndef __mast_scalar_constant_h__
#define __mast_scalar_constant_h__

namespace MAST {
namespace Base {

template <typename ScalarType>
class ScalarConstant {

public:
    
    using scalar_t = ScalarType;
    
    ScalarConstant(ScalarType v = 0.):
    _v  (v)
    { }
    
    virtual ~ScalarConstant() {}
    
    inline ScalarType& operator= (const ScalarType& v) {
        _v = v;
        return _v;
    }

    inline ScalarType operator() () const {
        return _v;
    }

    template <typename ContextType>
    inline ScalarType value(ContextType& c) const {
        return _v;
    }

    template <typename ContextType, typename ScalarFieldType>
    inline ScalarType derivative(ContextType& c,
                                 const ScalarFieldType& f) {
        return &f==this?1.:0.;
    }

private:
    
    ScalarType _v;
};
}
}

#endif // __mast_scalar_constant_h__
