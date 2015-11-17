
set(test_list
  read_arc.cpp
  read_geotiff.cpp
  ice_thickness.cpp
  field.cpp
  field_dist.cpp
  shallow_shelf.cpp
  shallow_stream.cpp
)


foreach(test ${test_list})
  string(REPLACE ".cpp" "" test ${test})
  add_executable(${test} ${PROJECT_SOURCE_DIR}/test/${test}.cpp)
  DEAL_II_SETUP_TARGET(${test})
  target_link_libraries(${test} icepack)
  add_test(${test} ${test})
endforeach(test)
