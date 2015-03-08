
set(example_list
  poisson_basic.cpp
)


foreach (example ${example_list})
  string(REPLACE ".cpp" "" example ${example})
  add_executable(${example} ${PROJECT_SOURCE_DIR}/examples/${example}.cpp)
  DEAL_II_SETUP_TARGET(${example})
endforeach(example)