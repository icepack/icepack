
add_library(testing testing.cpp)
DEAL_II_SETUP_TARGET(testing)

# Build executables for each name in a list of source files
function(icepack_exec)
  foreach(arg ${ARGN})
    add_executable(${arg} ${arg}.cpp)
    DEAL_II_SETUP_TARGET(${arg})
    target_link_libraries(${arg} icepack testing)
  endforeach(arg)
endfunction(icepack_exec)

# Make a test from the name of a program and a list of command-line arguments
function(icepack_test test_name prog_name)
  add_test(${test_name} ${prog_name} ${ARGN})
endfunction(icepack_test)

# Build a program and make it a test with no arguments for each name in a list
# of source file names
function(icepack_tests)
  foreach(arg ${ARGN})
    icepack_exec(${arg})
    icepack_test(${arg} ${arg})
  endforeach(arg)
endfunction(icepack_tests)

icepack_tests(
  tensor_function_utils
  ice_thickness
  )

add_subdirectory(grid_data)
add_subdirectory(field)
add_subdirectory(physics)
add_subdirectory(ice_shelf)
add_subdirectory(ice_stream)
add_subdirectory(inverse)
