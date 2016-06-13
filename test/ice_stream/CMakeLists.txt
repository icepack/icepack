
icepack_exec(
  ice_stream_diagnostic
  )

icepack_test(
  ice_stream_diagnostic_basic
  ice_stream_diagnostic
  )

icepack_test(
  ice_stream_diagnostic_grid_refinement
  ice_stream_diagnostic
  "--refined"
  )

icepack_test(
  ice_stream_diagnostic_q2_elements
  ice_stream_diagnostic
  "--q2"
  )
