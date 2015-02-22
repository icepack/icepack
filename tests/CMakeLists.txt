
set( test_list
read_arc.cpp
)


foreach(test ${test_list})
  string(REPLACE ".cpp" "" test ${test})
  add_executable(${test} ${PROJECT_SOURCE_DIR}/tests/${test}.cpp)
  DEAL_II_SETUP_TARGET(${test})
  target_link_libraries(${test} stuff)
  add_test(${test} ${test})
endforeach(test)