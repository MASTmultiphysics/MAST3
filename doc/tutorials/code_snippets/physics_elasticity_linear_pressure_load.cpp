for (uint_t i=0; i<fe.n_q_points(); i++) {
    
    c.qp       = i;
    scalar_t p = _pressure->value(c) * _section->value(c);
    
    for (uint_t j=0; j<Dim; j++) {
        
        // j-th component of normal vector at ith quadrature point
        scalar_t nj = fe.normal(i, j);
        
        if (nj != 0.) {
            for (uint_t k=0; k<fe.n_basis(); k++)
                res(j*fe.n_basis() + k) -= fe.detJxW(i) * fe.phi(i, k) * p * nj;
        }
    }
}
