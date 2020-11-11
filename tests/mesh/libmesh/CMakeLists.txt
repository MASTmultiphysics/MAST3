target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/geometric_filter.cpp)

target_include_directories(mast_catch_tests
                           PRIVATE
                           ${PROJECT_SOURCE_DIR}/examples)

#geometric filter transpose operation
add_test(NAME GeometricFilterTranspose
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "geometric_filter_transpose")
set_tests_properties(GeometricFilterTranspose
                    PROPERTIES
                    LABELS "SEQ"
                    FIXTURES_SETUP     GeometricFilterTranspose)

