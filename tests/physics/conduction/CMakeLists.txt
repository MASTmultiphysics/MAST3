target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/flux_load.cpp
               ${CMAKE_CURRENT_LIST_DIR}/linear_conduction_kernel.cpp
               ${CMAKE_CURRENT_LIST_DIR}/source_kernel.cpp)

#Linear conduction kernel
add_test(NAME LinearConductionKernel
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "linear_conduction_kernel")
set_tests_properties(LinearConductionKernel
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     LinearConductionKernel)

#Surface flux load kernel
add_test(NAME SurfaceFluxLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "surface_flux_load")
set_tests_properties(SurfaceFluxLoad
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     SurfaceFluxLoad)

#Source load kernel
add_test(NAME SourceLoad
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "source_load")
set_tests_properties(SourceLoad
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     SourceLoad)
