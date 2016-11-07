
icepack_tests(
  error_functionals
  regularization
  gradient
  )

icepack_exec(
  inverse_ice_shelf
  )

icepack_test(
  inverse_ice_shelf_square_gradient
  inverse_ice_shelf
  "--square-gradient"
  )

icepack_test(
  inverse_ice_shelf_total_variation
  inverse_ice_shelf
  "--total-variation"
  )
