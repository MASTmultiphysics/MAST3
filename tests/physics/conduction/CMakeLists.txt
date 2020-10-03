target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/linear_conduction_kernel.cpp)

#Linear conduction kernel
add_test(NAME LinearConductionKernel
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_conduction_kernel")
set_tests_properties(LinearConductionKernel
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     LinearConductionKernel)
