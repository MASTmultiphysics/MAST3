for (uint_t i=0; i<fe.n_q_points(); i++) {
    
    c.qp = i;
    
    _property->value(c, mat);
    MAST::Physics::Elasticity::LinearContinuum::strain
    <scalar_t, scalar_t, FEVarType, Dim>(*_fe_var_data, i, epsilon, Bxmat);
    stress = mat * epsilon;
    Bxmat.vector_mult_transpose(vec, stress);
    res += fe.detJxW(i) * vec;
    
    if (jac) {
        
        Bxmat.left_multiply(mat1, mat);
        Bxmat.right_multiply_transpose(mat2, mat1);
        (*jac) += fe.detJxW(i) * mat2;
    }
}
