target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/linear_strain_energy.cpp
               ${CMAKE_CURRENT_LIST_DIR}/isotropic_stiffness.cpp)

#Linear elasticity kernel
add_test(NAME LinearElasticStrainEnergy
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_strain_energy")
set_tests_properties(LinearElasticStrainEnergy
                     PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     LinearElasticStrainEnergy)

#Isotropic linear material
add_test(NAME IsotropicLinearMaterial
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "isotropic_linear_stiffness")
set_tests_properties(IsotropicLinearMaterial
                     PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     IsotropicLinearMaterial)


