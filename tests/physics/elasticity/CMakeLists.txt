target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/flat_shell_face_pressure_load_complex_step_sensitivity.cpp
               ${CMAKE_CURRENT_LIST_DIR}/isotropic_stiffness.cpp
               ${CMAKE_CURRENT_LIST_DIR}/linear_continuum_acceleration.cpp
               ${CMAKE_CURRENT_LIST_DIR}/linear_strain_energy.cpp
               ${CMAKE_CURRENT_LIST_DIR}/linear_thermoelastic_load.cpp
               ${CMAKE_CURRENT_LIST_DIR}/mindlin_plate_strain_energy_complex_step_sensitivity.cpp
               ${CMAKE_CURRENT_LIST_DIR}/plate_bending_section_property_complex_step_sensitivity.cpp
               ${CMAKE_CURRENT_LIST_DIR}/pressure_load.cpp
               ${CMAKE_CURRENT_LIST_DIR}/stress_evaluation.cpp
               ${CMAKE_CURRENT_LIST_DIR}/traction_load.cpp
               ${CMAKE_CURRENT_LIST_DIR}/von_mises_stress_complex_step_sensitivity.cpp)

#Linear continuum acceleration kernel
add_test(NAME LinearContinuumAcceleration
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_continuum_acceleration")
set_tests_properties(LinearContinuumAcceleration
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     LinearContinuumAcceleration)


#Linear elasticity kernel
add_test(NAME LinearElasticStrainEnergy
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_strain_energy")
set_tests_properties(LinearElasticStrainEnergy
        PROPERTIES
        LABELS "SEQ"
        #FIXTURES_REQUIRED  "Element_Property_Card_1D_Structural;libMesh_Mesh_Generation_1d"
        FIXTURES_SETUP     LinearElasticStrainEnergy)


#Linear thermoelastic load kernel
add_test(NAME LinearThermoelasticLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_thermoelastic_load")
set_tests_properties(LinearThermoelasticLoad
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     LinearThermoelasticLoad)


#Linear pressure load kernel
add_test(NAME LinearPressureLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_pressure_load")
set_tests_properties(LinearPressureLoad
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     LinearPressureLoad)

#Flat shellface pressure load kernel
add_test(NAME FlatShellFaceLinearPressureLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "flat_shell_face_linear_pressure_load")
set_tests_properties(FlatShellFaceLinearPressureLoad
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     FlatShellFaceLinearPressureLoad)

#Linear traction load kernel
add_test(NAME LinearTractionLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_traction_load")
set_tests_properties(LinearTractionLoad
                     PROPERTIES
                     LABELS "SEQ"
                     FIXTURES_SETUP     LinearTractionLoad)


#Isotropic linear material
add_test(NAME IsotropicLinearMaterial
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "isotropic_linear_stiffness")
set_tests_properties(IsotropicLinearMaterial
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     IsotropicLinearMaterial)


#plate bending section property
add_test(NAME PlateBendingSectionPropertyComplexStep
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "plate_bending_section_complex_step")
set_tests_properties(PlateBendingSectionPropertyComplexStep
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     PlateBendingSectionPropertyComplexStep)


#Mindlin plate strain energy
add_test(NAME MindlinPlateStrainEnergyComplexStep
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "mindlin_plate_strain_energy_complex_step")
set_tests_properties(MindlinPlateStrainEnergyComplexStep
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     MindlinPlateStrainEnergyComplexStep)


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
if (MAST_ENABLE_ADOLC EQUAL 1)
    target_sources(mast_catch_tests
                    PRIVATE
                    ${CMAKE_CURRENT_LIST_DIR}/von_mises_stress_adolc_sensitivity.cpp)
    add_test(NAME vonMisesStressAdolC
             COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "von_mises_stress_adolc")
    set_tests_properties(vonMisesStressAdolC
            PROPERTIES
            LABELS "SEQ"
            FIXTURES_SETUP     vonMisesStressAdolC)
endif()

