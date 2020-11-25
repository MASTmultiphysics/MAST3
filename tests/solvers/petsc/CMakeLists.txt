target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/nonlinear_solver.cpp)

#PETSc nonlinear solver
add_test(NAME PETScNonlinearSolver
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "petsc_nonlinear_solver")
set_tests_properties(PETScNonlinearSolver
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     PETScNonlinearSolver)

