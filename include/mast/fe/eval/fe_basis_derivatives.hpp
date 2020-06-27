
#ifndef __mast_fe_shape_derivative_h__
#define __mast_fe_shape_derivative_h__

// MAST includes
#include <mast/base/mast_data_types.h>
#include <mast/base/exceptions.hpp>
#include <mast/fe/eval/fe_derivative_evaluation.hpp>


namespace MAST {
namespace FEBasis {
namespace Evaluation {

template <typename BasisScalarType,
          typename NodalScalarType,
          uint_t ElemDim,
          uint_t SpatialDim,
          typename FEBasisType>
class FEShapeDerivative {
    
public:
    
    static const uint_t ref_dim     = ElemDim;
    static const uint_t spatial_dim = SpatialDim;
    using basis_scalar_t = BasisScalarType;
    using nodal_scalar_t = NodalScalarType;
    using scalar_t       = typename MAST::DeducedScalarType<BasisScalarType, NodalScalarType>::type;
    using fe_basis_t     = FEBasisType;
    using dxi_dx_mat_t   = typename Eigen::Map<const typename Eigen::Matrix<NodalScalarType, ElemDim, SpatialDim>>;
    using dx_dxi_mat_t   = typename Eigen::Map<const typename Eigen::Matrix<NodalScalarType, SpatialDim, ElemDim>>;
    using dphi_dx_mat_t  = typename Eigen::Map<const typename Eigen::Matrix<NodalScalarType, Eigen::Dynamic, SpatialDim>>;
    using dphi_dx_vec_t  = typename Eigen::Map<const typename Eigen::Matrix<NodalScalarType, Eigen::Dynamic, 1>>;
    using normal_vec_t   = typename Eigen::Map<const typename Eigen::Matrix<NodalScalarType, SpatialDim, 1>>;
    static_assert(std::is_same<nodal_scalar_t, scalar_t>::value,
                  "The nodal scalar type should be the derived scalar type.");
    static_assert(std::is_same<BasisScalarType, typename FEBasisType::scalar_t>::value,
                  "BasisScalarType incompatible with FEBasisType::scalar_t.");
    static_assert(ElemDim == FEBasisType::dim,
                  "FE Dimension should be same as element dimension.");

    FEShapeDerivative():
    _if_xyz       (false),
    _if_Jac       (false),
    _if_Jac_inv   (false),
    _if_detJ      (false),
    _if_JxW       (false),
    _if_dphi_dx   (false),
    _if_normal    (false),
    _fe_basis          (nullptr)
    { }
    
    ~FEShapeDerivative() {}
    
    inline void      set_compute_xyz(bool f) { _if_xyz = f;}
    
    inline void      set_compute_Jac(bool f) {
        
        _if_Jac = f;
        if (f) this->set_compute_xyz(true);
    }

    inline void      set_compute_Jac_inverse(bool f) {
        
        _if_Jac_inv = f;
        if (f) this->set_compute_Jac(true);
    }

    inline void     set_compute_detJ(bool f) {
        
        _if_detJ = f;
        if (f) this->set_compute_Jac(true);
    }
    
    inline void   set_compute_detJxW(bool f) {
        
        _if_JxW = f;
        if (f) this->set_compute_detJ(true);
    }
    
    inline void  set_compute_dphi_dx(bool f) {
        
        _if_dphi_dx = f;
        if (f) this->set_compute_Jac_inverse(true);
    }
    
    inline void   set_compute_normal(bool f) {
        
        _if_normal = f;
        if (f) this->set_compute_Jac(true);
    }
    
    inline void  set_fe_basis(FEBasisType& basis)
    {
        Assert0(!_fe_basis, "FE Basis already initialized.");
        
        _fe_basis = &basis;
    }
    
    template <typename ContextType>
    inline void reinit(const ContextType& c) {
        
        // for this class the number of basis functions should be equal to the number
        // of nodes
        Assert2(c.n_nodes() == _fe_basis->n_basis(),
                c.n_nodes(), _fe_basis->n_basis(),
                "Number of shape functions assumed equal to number of nodes");
        Assert2(c.elem_dim() == ElemDim,
                c.elem_dim(), ElemDim,
                "Incorrect dimension of element.");

        if (_if_xyz)
            MAST::FEBasis::Evaluation::compute_xyz
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType, ContextType>
            (c, *_fe_basis, _node_coord, _xyz);
        
        if (_if_Jac)
            MAST::FEBasis::Evaluation::compute_Jac
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType, ContextType>
            (c, *_fe_basis, _node_coord, _dx_dxi);
        
        if (_if_detJ)
            MAST::FEBasis::Evaluation::compute_detJ<NodalScalarType, ElemDim, SpatialDim>
            (_dx_dxi, _detJ);
        
        if (_if_JxW)
            MAST::FEBasis::Evaluation::compute_detJxW
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType, ContextType>
            (*_fe_basis, *_detJ, *_detJxW);
        
        if (_if_Jac_inv)
            MAST::FEBasis::Evaluation::compute_Jac_inv<NodalScalarType, ElemDim, SpatialDim>
            (_dx_dxi, _dxi_dx);
        
        if (_if_dphi_dx)
            MAST::FEBasis::Evaluation::compute_dphi_dx
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType>
            (*_fe_basis, *_dxi_dx, *_dphi_dx);
    }
    

    template <typename ContextType>
    inline void reinit_for_side(const ContextType& c, uint_t s) {
        
        // for this class the number of basis functions should be equal to the number
        // of nodes
        Assert2(c.n_nodes() == _fe_basis->n_basis(),
                c.n_nodes(), _fe_basis->n_basis(),
                "Number of shape functions assumed equal to number of nodes");
        Assert2(c.elem_dim() == ElemDim,
                c.elem_dim(), ElemDim,
                "Incorrect dimension of element.");

        if (_if_xyz)
            MAST::FEBasis::Evaluation::compute_xyz
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType, ContextType>
            (c, *_fe_basis, _node_coord, _xyz);
        
        if (_if_Jac)
            MAST::FEBasis::Evaluation::compute_Jac
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType, ContextType>
            (c, *_fe_basis, _node_coord, _dx_dxi);

        if (this->_if_detJ)
            MAST::FEBasis::Evaluation::compute_detJ_side
            <NodalScalarType, ElemDim, SpatialDim, ContextType>
            (c, s, *_dx_dxi, *_detJ);

        if (_if_JxW)
            MAST::FEBasis::Evaluation::compute_detJxW
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType, ContextType>
            (*_fe_basis, *_detJ, *_detJxW);
        
        if (_if_Jac_inv)
            MAST::FEBasis::Evaluation::compute_Jac_inv<NodalScalarType, ElemDim, SpatialDim>
            (_dx_dxi, _dxi_dx);
        
        if (_if_dphi_dx)
            MAST::FEBasis::Evaluation::compute_dphi_dx
            <NodalScalarType, ElemDim, SpatialDim, FEBasisType>
            (*_fe_basis, *_dxi_dx, *_dphi_dx);

        if (this->_if_normal)
            MAST::FEBasis::Evaluation::compute_side_tangent_and_normal
            <NodalScalarType, ElemDim, SpatialDim, ContextType>
            (c, s, *_dx_dxi, *_side_tangent, *_side_normal);
    }

    inline uint_t               order() const {
    
        Assert0(_fe_basis, "FE Basis not initialized.");
        return _fe_basis->order();
    }
    
    inline uint_t             n_basis() const {
    
        Assert0(_fe_basis, "FE Basis not initialized.");
        return _fe_basis->n_basis();
    }
    
    inline BasisScalarType         phi(uint_t qp, uint_t phi_i) const
    {
        Assert0(_fe_basis, "FE Basis not initialized.");
         return _fe_basis->phi(qp, phi_i);
    }
    
    inline BasisScalarType    dphi_dxi(uint_t qp, uint_t phi_i, uint_t xi_i) const
    {
    
        Assert0(_fe_basis, "FE Basis not initialized.");

        return _fe_basis->dphi_dxi(qp, phi_i, xi_i);
    }
    
    inline uint_t        n_q_points() const { return _fe_basis->n_q_points();}

    inline NodalScalarType  node_coord(uint_t nd, uint_t x_i) const
    {
        Assert0(_if_xyz, "Nodal and QPoint locations not requested");
        return _node_coord(x_i, nd);
    }

    inline NodalScalarType         xyz(uint_t qp, uint_t x_i) const
    {
        Assert0(_if_xyz, "Nodal and QPoint locations not requested");
        return _xyz(x_i, qp);
    }
    
    inline NodalScalarType        detJ(uint_t qp) const
    {
        Assert0(_if_detJ, "Jacobian computation not requested");
        return _detJ(qp);
    }

    inline NodalScalarType      detJxW(uint_t qp) const
    {
        Assert0(_if_JxW, "JxW computation not requested");
        return _detJxW(qp);
    }

    inline dx_dxi_mat_t      dx_dxi(uint_t qp) const
    {
        Assert0(_if_Jac, "Jacobian computation not requested");
        return _dx_dxi_mat_t(_dx_dxi.col(qp).data(), spatial_dim, ref_dim);
    }

    inline NodalScalarType      dx_dxi(uint_t qp, uint_t x_i, uint_t xi_i) const
    {
        Assert0(_if_Jac, "Jacobian computation not requested");
        return _dx_dxi(xi_i*spatial_dim+x_i, qp);
    }

    inline dxi_dx_mat_t      dxi_dx(uint_t qp) const
    {
        Assert0(_if_Jac_inv, "Jacobian inverse computation not requested");
        return _dxi_dx_mat_t(_dxi_dx.col(qp).data(), ref_dim, spatial_dim);
    }

    inline NodalScalarType      dxi_dx(uint_t qp, uint_t x_i, uint_t xi_i) const
    {
        Assert0(_if_Jac_inv, "Jacobian inverse computation not requested");
        return _dxi_dx(x_i*ref_dim+xi_i, qp);
    }

    inline const dphi_dx_mat_t
    dphi_dx(uint_t qp) const
    {
        Assert0(_if_dphi_dx, "Jacobian inverse computation not requested");

        return dphi_dx_mat_t(_dphi_dx.col(qp).data(), this->n_basis(), spatial_dim);
    }

    inline const dphi_dx_vec_t
    dphi_dx(uint_t qp, uint_t x_i) const
    {
        Assert0(_if_dphi_dx, "Jacobian inverse computation not requested");

        return dphi_dx_vec_t(_dphi_dx.col(qp).segment(x_i*this->n_basis(), this->n_basis()).data(),
                             this->n_basis());
    }
    
    inline NodalScalarType     dphi_dx(uint_t qp, uint_t phi_i, uint_t x_i) const
    {
        Assert0(_if_dphi_dx, "Jacobian inverse computation not requested");

        return _dphi_dx(qp, x_i*this->n_basis()+phi_i);
    }

    inline normal_vec_t normal(uint_t qp) const
    {
        Assert0(_if_normal, "Normal not requested");

        return normal_vec_t(_side_normal.col(qp).data(), SpatialDim);
    }

    inline NodalScalarType     normal(uint_t qp, uint_t x_i) const
    {
        Assert0(_if_normal, "Normal not requested");
        Assert2(x_i < SpatialDim,
                x_i, SpatialDim,
                "Invalid normal component index");

        return _side_normal(x_i, qp);
    }

protected:

    bool _if_xyz;
    bool _if_Jac;
    bool _if_Jac_inv;
    bool _if_detJ;
    bool _if_JxW;
    bool _if_dphi_dx;
    bool _if_normal;
    
    FEBasisType  *_fe_basis;
    
    Eigen::Matrix<NodalScalarType, SpatialDim, Eigen::Dynamic>            _node_coord;
    Eigen::Matrix<NodalScalarType, SpatialDim, Eigen::Dynamic>            _xyz;
    Eigen::Matrix<NodalScalarType, Eigen::Dynamic, 1>                     _detJ;
    Eigen::Matrix<NodalScalarType, Eigen::Dynamic, 1>                     _detJxW;
    Eigen::Matrix<NodalScalarType, SpatialDim*ElemDim, Eigen::Dynamic>    _dx_dxi;
    Eigen::Matrix<NodalScalarType, ElemDim*SpatialDim, Eigen::Dynamic>    _dxi_dx;
    Eigen::Matrix<NodalScalarType, Eigen::Dynamic, Eigen::Dynamic>        _dphi_dx;
    Eigen::Matrix<NodalScalarType, SpatialDim, Eigen::Dynamic>            _side_tangent;
    Eigen::Matrix<NodalScalarType, SpatialDim, Eigen::Dynamic>            _side_normal;
};

}  // Evaluation
}  // FEBasis
}  // MAST


#endif // __mast_fe_shape_derivative_h__
