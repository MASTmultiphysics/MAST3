target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/constrained_generalized_hermitian_eigen_solver.cpp
               ${CMAKE_CURRENT_LIST_DIR}/hermitian_eigen_solver.cpp
               ${CMAKE_CURRENT_LIST_DIR}/nonlinear_solver.cpp)


#Constrained Generalized Hermitian eigen solver
add_test(NAME ConstrainedGeneralizedEigenHermitianEigenSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "eigen_constrained_generalized_hermitian_eigen_solver")
set_tests_properties(ConstrainedGeneralizedEigenHermitianEigenSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     ConstrainedGeneralizedEigenHermitianEigenSolver)



#Hermitian eigen solver
add_test(NAME EigenHermitianEigenSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "eigen_hermitian_eigen_solver")
set_tests_properties(EigenHermitianEigenSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     EigenHermitianEigenSolver)



#Eigen nonlinear solver
add_test(NAME EigenNonlinearSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "eigen_nonlinear_solver")
set_tests_properties(EigenNonlinearSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     EigenNonlinearSolver)

