target_sources(mast_catch_tests
               PRIVATE
               ${CMAKE_CURRENT_LIST_DIR}/gcmma_interface.cpp)

#GCMMA interface
add_test(NAME GCMMAInterface
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "gcmma_interface")
set_tests_properties(GCMMAInterface
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     GCMMAInterface)

