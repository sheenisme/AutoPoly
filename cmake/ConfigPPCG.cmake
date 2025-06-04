# ============================================================================
# PPCG Library Configuration (Always built from source)
# ============================================================================
# PPCG (Polyhedral Parallel Code Generator) is the core library for this project.
# It transforms C code to parallel code using the polyhedral model.
# PPCG is ALWAYS built from source to ensure compatibility with the project.
# No external linking option is provided to prevent version incompatibilities.
message(STATUS "Configuring PPCG library build from source (required)")

# Create gitversion.h file (required by PPCG)
execute_process(
    COMMAND git describe --always 
    WORKING_DIRECTORY ${PPCG_DIR}
    OUTPUT_VARIABLE PPCG_GIT_HEAD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(WRITE ${PPCG_DIR}/gitversion.h "#define GIT_HEAD_ID \"${PPCG_GIT_HEAD_VERSION}\"")
message(STATUS "PPCG version: ${PPCG_GIT_HEAD_VERSION}")  

include_directories(ppcg PRIVATE
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
)

set(PPCG_SOURCES
    ${PPCG_DIR}/cpu.c
    ${PPCG_DIR}/cuda.c
    ${PPCG_DIR}/opencl.c
    ${PPCG_DIR}/cuda_common.c
    ${PPCG_DIR}/gpu.c
    ${PPCG_DIR}/gpu_array_tile.c
    ${PPCG_DIR}/gpu_group.c
    ${PPCG_DIR}/gpu_hybrid.c
    ${PPCG_DIR}/gpu_print.c
    ${PPCG_DIR}/gpu_tree.c
    ${PPCG_DIR}/grouping.c
    ${PPCG_DIR}/hybrid.c
    ${PPCG_DIR}/schedule.c
    ${PPCG_DIR}/ppcg_options.c
    ${PPCG_DIR}/print.c
    ${PPCG_DIR}/util.c
    ${PPCG_DIR}/version.c
    # Remove ppcg.c from the library to avoid main function conflict
    # ${PPCG_DIR}/ppcg.c
    # Add ppcg_wrapper.c file containing ppcg_main implementation
    ${AUTOPOLY_SOURCE_DIR}/lib/ppcg_wrapper.c
)

# Build PPCG library
add_library(ppcg ${PPCG_SOURCES})
target_link_libraries(ppcg isl pet)
if(GMP_FOUND)
    target_link_libraries(ppcg ${GMP_LIBRARIES})
endif()
if(MPFR_FOUND)
    target_link_libraries(ppcg ${MPFR_LIBRARIES})
endif()
if(OpenMP_FOUND)
    target_link_libraries(ppcg ${OpenMP_CXX_LIBRARIES})
endif()
if(OpenCL_FOUND)
    target_link_libraries(ppcg ${OpenCL_LIBRARIES})
endif()

# Set compile options to PPCG
target_compile_options(ppcg PRIVATE
    -Wno-sign-compare
    -Wno-cast-qual
    -Wno-discarded-qualifiers
    -Wno-implicit-fallthrough
    -Wno-unused-function
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-type-limits
    -Wno-return-type
    -Wno-extra
)

# Build PPCG executable
add_executable(ppcg_exe ${PPCG_DIR}/ppcg.c)
set_target_properties(ppcg_exe PROPERTIES OUTPUT_NAME "ppcg")
target_link_libraries(ppcg_exe PRIVATE ppcg isl pet)
if(GMP_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${GMP_LIBRARIES})
endif()
if(MPFR_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${MPFR_LIBRARIES})
endif()
if(OpenMP_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${OpenMP_CXX_LIBRARIES})
endif()
if(OpenCL_FOUND)
    target_link_libraries(ppcg_exe PRIVATE ${OpenCL_LIBRARIES})
endif()

# Set compile options to the binary of PPCG
target_compile_options(ppcg_exe PRIVATE
    -Wno-sign-compare
    -Wno-cast-qual
    -Wno-discarded-qualifiers
    -Wno-implicit-fallthrough
    -Wno-unused-function
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-type-limits
    -Wno-return-type
    -Wno-extra
)

# Set installation rules for PPCG
install(TARGETS ppcg ppcg_exe
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

# Install PPCG headers
file(GLOB PPCG_HEADERS "${PPCG_DIR}/*.h") 
install(FILES ${PPCG_HEADERS} 
    DESTINATION include/ppcg
)
