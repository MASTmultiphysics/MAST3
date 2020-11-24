target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/hermitian_eigen_solver.hpp)

#SLEPc Hermitian eigensolver interface
add_test(NAME SLEPcHermitianEigenSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "slepc_hermitian_interface")
set_tests_properties(SLEPcHermitianEigenSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     SLEPcHermitianEigenSolver)

