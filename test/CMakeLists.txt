
function(icepack_test)
  foreach(test ${ARGV})
    add_executable(${test} ${test}.cpp)
    DEAL_II_SETUP_TARGET(${test})
    target_link_libraries(${test} icepack)
    add_test(${test} ${test})
  endforeach(test)
endfunction(icepack_test)

icepack_test(
  tensor_function_utils
  read_arc
  read_geotiff
  ice_thickness)

add_subdirectory(field)
add_subdirectory(ice_shelf)
add_subdirectory(ice_stream)
add_subdirectory(inverse)
