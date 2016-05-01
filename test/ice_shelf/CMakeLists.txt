
icepack_tests(
  ice_shelf_adjoint
  ice_shelf_prognostic
  )

icepack_exec(
  ice_shelf_diagnostic
  )

icepack_test(
  ice_shelf_diagnostic_basic
  ice_shelf_diagnostic
  )

icepack_test(
  ice_shelf_diagnostic_grid_refinement
  ice_shelf_diagnostic
  "--refined"
  )

icepack_test(
  ice_shelf_diagnostic_q2_element
  ice_shelf_diagnostic
  "--q2"
  )
