# Transform tests for AutoPoly

# Add test files to the build system
configure_file(tiling.mlir tiling.mlir COPYONLY)
configure_file(parallelization.mlir parallelization.mlir COPYONLY)
configure_file(fusion.mlir fusion.mlir COPYONLY)
configure_file(skewing.mlir skewing.mlir COPYONLY)

add_custom_target(TransformTests)
file(GLOB TRANSFORM_TESTS "*.mlir")
foreach(testfile ${TRANSFORM_TESTS})
  add_test(NAME transform-${testfile}
           COMMAND autopoly-mlir-opt ${testfile} -autopoly-scheduling)
endforeach()
