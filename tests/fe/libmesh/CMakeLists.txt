target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/fe_basis_quad4.cpp)

#Quad4 basis function evaluation
add_test(NAME Quad4_ShapeFunctionDerivatives
    COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "Quad4_ShapeFunctionDerivatives")
set_tests_properties(Quad4_ShapeFunctionDerivatives
    PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     Quad4_ShapeFunctionDerivatives)

