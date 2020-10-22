add_executable(conduction_example_2
               example_2.cpp)

target_include_directories(conduction_example_2 PRIVATE
                            ${PROJECT_SOURCE_DIR}/include,
                            ${PROJECT_SOURCE_DIR}/contrib)

target_link_libraries(conduction_example_2 mast)

#install(TARGETS conduction_example_2
#    RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/examples)

# Test on single processor, PETSc built-in LU direct linear solver (sequential).
#add_test(NAME conduction_example_2
#    COMMAND $<TARGET_FILE:conduction_example_2> -ksp_type preonly -pc_type lu -options_view)
#set_tests_properties(conduction_example_2
#    PROPERTIES
#        LABELS "SHORT;SEQ")

# Test multiple processors, parallel direct linear solver using external MUMPS package.
#add_test(NAME conduction_example_2_mpi
#    COMMAND ${MPIEXEC_EXECUTABLE} -np 2 $<TARGET_FILE:conduction_example_2>
#            -ksp_type preonly -pc_type lu -pc_factor_mat_solver_package mumps -options_view)
#set_tests_properties(conduction_example_2_mpi
#    PROPERTIES
#        LABELS "SHORT;MPI")
