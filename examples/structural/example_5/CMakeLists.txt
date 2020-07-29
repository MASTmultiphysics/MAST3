add_executable(structural_example_5
                example_5.cpp)

target_include_directories(structural_example_5 PRIVATE
                            ${PROJECT_SOURCE_DIR}/include,
                            ${PROJECT_SOURCE_DIR}/contrib)

target_link_libraries(structural_example_5 mast)

#install(TARGETS structural_example_5
#    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)

# Test on single processor, PETSc built-in LU direct linear solver (sequential).
#add_test(NAME structural_example_5
#    COMMAND $<TARGET_FILE:structural_example_5> -ksp_type preonly -pc_type lu -options_view)
#set_tests_properties(structural_example_5
#    PROPERTIES
#        LABELS "SHORT;SEQ")

# Test multiple processors, parallel direct linear solver using external MUMPS package.
#add_test(NAME structural_example_5_mpi
#    COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:structural_example_5>
#            -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package mumps -options_view)
#set_tests_properties(structural_example_5_mpi
#    PROPERTIES
#        LABELS "SHORT;MPI")
