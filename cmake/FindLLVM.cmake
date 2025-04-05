# FindLLVM.cmake
# This module can find LLVM installation

# First, find llvm-config
find_program(LLVM_CONFIG
  NAMES llvm-config-14 llvm-config-10 llvm-config
  PATHS
  /usr/bin
  /usr/local/bin
  ${LLVM_ROOT}/bin
  NO_DEFAULT_PATH
)

if(NOT LLVM_CONFIG)
  message(FATAL_ERROR "Could not find llvm-config. Please set LLVM_ROOT to your LLVM installation.")
endif()

# Get LLVM version
execute_process(
  COMMAND ${LLVM_CONFIG} --version
  OUTPUT_VARIABLE LLVM_VERSION
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get LLVM installation prefix
execute_process(
  COMMAND ${LLVM_CONFIG} --prefix
  OUTPUT_VARIABLE LLVM_INSTALL_PREFIX
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get LLVM include directories
execute_process(
  COMMAND ${LLVM_CONFIG} --includedir
  OUTPUT_VARIABLE LLVM_INCLUDE_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get LLVM library directories
execute_process(
  COMMAND ${LLVM_CONFIG} --libdir
  OUTPUT_VARIABLE LLVM_LIBRARY_DIRS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get LLVM libraries
execute_process(
  COMMAND ${LLVM_CONFIG} --libs
  OUTPUT_VARIABLE LLVM_LIBRARIES_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert libraries list to CMake list
string(REPLACE " " ";" LLVM_LIBRARIES "${LLVM_LIBRARIES_RAW}")

# Get LLVM system libraries
execute_process(
  COMMAND ${LLVM_CONFIG} --system-libs
  OUTPUT_VARIABLE LLVM_SYSTEM_LIBS_RAW
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Convert system libraries list to CMake list
string(REPLACE " " ";" LLVM_SYSTEM_LIBS "${LLVM_SYSTEM_LIBS_RAW}")

# Combine all libraries
list(APPEND LLVM_LIBRARIES ${LLVM_SYSTEM_LIBS})

# Get LLVM compile flags
execute_process(
  COMMAND ${LLVM_CONFIG} --cxxflags
  OUTPUT_VARIABLE LLVM_CXXFLAGS
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Get LLVM CMake directory
execute_process(
  COMMAND ${LLVM_CONFIG} --cmakedir
  OUTPUT_VARIABLE LLVM_CMAKE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET
)

# If llvm-config --cmakedir failed, try to find it manually
if(NOT LLVM_CMAKE_DIR)
  find_path(LLVM_CMAKE_DIR
    NAMES LLVMConfig.cmake
    PATHS
    ${LLVM_INSTALL_PREFIX}/lib/cmake/llvm
    ${LLVM_INSTALL_PREFIX}/share/llvm/cmake
    /usr/lib/llvm-14/lib/cmake/llvm
    /usr/lib/llvm-10/lib/cmake/llvm
    /usr/share/llvm-14/cmake
    /usr/share/llvm-10/cmake
    NO_DEFAULT_PATH
  )
endif()

# Try to find LLVMConfig.cmake
if(LLVM_CMAKE_DIR)
  set(LLVM_DIR ${LLVM_CMAKE_DIR})
  find_package(LLVM CONFIG QUIET)
  
  # If found via CONFIG mode, use those values
  if(LLVM_FOUND)
    message(STATUS "Found LLVM via CONFIG mode: ${LLVM_DIR}")
    # Keep the variables we set above, as they might be more accurate
    # than what we get from LLVMConfig.cmake
    if(NOT DEFINED LLVM_VERSION)
      set(LLVM_VERSION ${LLVM_PACKAGE_VERSION})
    endif()
  endif()
endif()

# Handle the QUIETLY and REQUIRED arguments and set LLVM_FOUND to TRUE if all variables are not empty
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LLVM DEFAULT_MSG LLVM_LIBRARIES LLVM_INCLUDE_DIRS)

mark_as_advanced(LLVM_INCLUDE_DIRS LLVM_LIBRARIES LLVM_LIBRARY_DIRS LLVM_CXXFLAGS)