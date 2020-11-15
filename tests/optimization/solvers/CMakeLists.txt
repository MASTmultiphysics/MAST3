target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/gcmma_interface.cpp
               ${CMAKE_CURRENT_LIST_DIR}/snopt_interface.cpp)

#GCMMA interface
add_test(NAME GCMMAInterface
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "gcmma_interface")
set_tests_properties(GCMMAInterface
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     GCMMAInterface)

#SNOPT interface
add_test(NAME SNOPTInterface
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "snopt_interface")
set_tests_properties(SNOPTInterface
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     SNOPTInterface)

