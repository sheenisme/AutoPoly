# Scheduling tests for AutoPoly

add_custom_target(SchedulingTests)
file(GLOB SCHEDULING_TESTS "*.mlir")
foreach(testfile ${SCHEDULING_TESTS})
  add_test(NAME scheduling-${testfile}
           COMMAND autopoly-mlir-opt ${testfile} -autopoly-scheduling)
endforeach()