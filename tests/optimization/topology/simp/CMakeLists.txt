target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/penalized_density_sensitivity.cpp
               ${CMAKE_CURRENT_LIST_DIR}/penalized_youngs_modulus_sensitivity.cpp)

#penalized density sensitivity
add_test(NAME PenalizedDensitySensitivity
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "penalized_density_sensitivity")
set_tests_properties(PenalizedDensitySensitivity
                    PROPERTIES
                    LABELS "SEQ"
                    FIXTURES_SETUP     PenalizedDensitySensitivity)


#penalized youngs modulus sensitivity
add_test(NAME PenalizedYoungsModulusSensitivity
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "penalized_youngs_modulus_sensitivity")
set_tests_properties(PenalizedYoungsModulusSensitivity
                    PROPERTIES
                    LABELS "SEQ"
                    FIXTURES_SETUP     PenalizedYoungsModulusSensitivity)

