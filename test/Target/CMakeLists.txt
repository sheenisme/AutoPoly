# Target tests for AutoPoly

add_custom_target(TargetTests)
file(GLOB TARGET_TESTS "*.mlir")
foreach(testfile ${TARGET_TESTS})
  add_test(NAME target-${testfile}
           COMMAND autopoly-mlir-opt ${testfile} -autopoly-target-detection)
endforeach()