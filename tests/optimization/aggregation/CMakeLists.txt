target_sources(mast_catch_tests
               PUBLIC
               ${CMAKE_CURRENT_LIST_DIR}/discrete_aggregation.cpp)

#discrete aggregation
add_test(NAME DiscreteAggregation
         COMMAND $<TARGET_FILE:mast_catch_tests> -w NoTests "discrete_aggregation")
set_tests_properties(DiscreteAggregation
        PROPERTIES
        LABELS "SEQ"
        FIXTURES_SETUP     DiscreteAggregation)

