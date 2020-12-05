target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/constrained_hermitian_eigen_solver.cpp
               ${CMAKE_CURRENT_LIST_DIR}/hermitian_eigen_solver.cpp)

#SLEPc Constrained Hermitian eigensolver interface
add_test(NAME SLEPcConstrainedHermitianEigenSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "slepc_constrained_hermitian_interface")
set_tests_properties(SLEPcConstrainedHermitianEigenSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     SLEPcConstrainedHermitianEigenSolver)


#SLEPc Hermitian eigensolver interface
add_test(NAME SLEPcHermitianEigenSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "slepc_hermitian_interface")
set_tests_properties(SLEPcHermitianEigenSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     SLEPcHermitianEigenSolver)

