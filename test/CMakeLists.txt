# AutoPoly Test Suite

# Configure MLIR test suite
configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(AUTOPOLY_TEST_DEPENDS
  FileCheck count not
  autopoly-opt
  AutoPolyLib
)

add_lit_testsuite(check-autopoly "Running the AutoPoly regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${AUTOPOLY_TEST_DEPENDS}
)

set_target_properties(check-autopoly PROPERTIES FOLDER "Tests")

add_lit_testsuites(AUTOPOLY ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${AUTOPOLY_TEST_DEPENDS})

# Add subdirectories for different test categories
add_subdirectory(Analysis)
add_subdirectory(Transform)
# add_subdirectory(Passes)
# add_subdirectory(Integration)
