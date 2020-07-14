target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/isotropic_stiffness.cpp
               ${CMAKE_CURRENT_LIST_DIR}/linear_strain_energy.cpp
               ${CMAKE_CURRENT_LIST_DIR}/pressure_load.cpp
               ${CMAKE_CURRENT_LIST_DIR}/stress_evaluation.cpp
               ${CMAKE_CURRENT_LIST_DIR}/von_mises_stress_complex_step_sensitivity.cpp
               ${CMAKE_CURRENT_LIST_DIR}/von_mises_stress_adolc_sensitivity.cpp)

#Linear elasticity kernel
add_test(NAME LinearElasticStrainEnergy
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_strain_energy")
set_tests_properties(LinearElasticStrainEnergy
        PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     LinearElasticStrainEnergy)


#Linear pressure load kernel
add_test(NAME LinearPressureLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_pressure_load")
set_tests_properties(LinearPressureLoad
        PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     LinearPressureLoad)


#Isotropic linear material
add_test(NAME IsotropicLinearMaterial
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "isotropic_linear_stiffness")
set_tests_properties(IsotropicLinearMaterial
        PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     IsotropicLinearMaterial)


#stress evaluation
add_test(NAME StressEvaluation
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "stress_evaluation")
set_tests_properties(StressEvaluation
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     StressEvaluation)


#von Mises stress: complex-step sensitivity
add_test(NAME vonMisesStressComplexStep
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "von_mises_stress_complex_step")
set_tests_properties(vonMisesStressComplexStep
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     vonMisesStressComplexStep)


#von Mises stress: adol-c sensitivity
if (ENABLE_ADOLC)
    add_test(NAME vonMisesStressAdolC
             COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "von_mises_stress_adolc")
    set_tests_properties(vonMisesStressAdolC
            PROPERTIES
            LABELS "SEQ"
            FIXTURES_SETUP     vonMisesStressAdolC)
endif()

