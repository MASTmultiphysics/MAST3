add_executable(conduction_example_1
                example_1.cpp)

target_include_directories(conduction_example_1 PRIVATE
                            ${PROJECT_SOURCE_DIR}/include)

target_link_libraries(conduction_example_1 mast)

install(TARGETS conduction_example_1
    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)

# Test on single processor, PETSc built-in LU direct linear solver (sequential).
add_test(NAME conduction_example_1
    COMMAND $<TARGET_FILE:conduction_example_1> -ksp_type preonly -pc_type lu -options_view)
set_tests_properties(conduction_example_1
                     PROPERTIES
                     LABELS "SHORT;SEQ")
