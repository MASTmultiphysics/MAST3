target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/nonlinear_solver.cpp)

#Eigen nonlinear solver
add_test(NAME EigenNonlinearSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "eigen_nonlinear_solver")
set_tests_properties(EigenNonlinearSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     EigenNonlinearSolver)

