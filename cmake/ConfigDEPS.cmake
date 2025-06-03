# ============================================================================
# Common Dependencies Configuration
# ============================================================================

# Project compiler settings and options for Debug build
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Compile information
    message(STATUS "C compiler location: ${CMAKE_C_COMPILER}")
    message(STATUS "C++ compiler location: ${CMAKE_CXX_COMPILER}")

    # Show compiler search path
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-search-dirs
        OUTPUT_VARIABLE SEARCH_DIRS
    )
    message(STATUS "Compiler search path:\n${SEARCH_DIRS}")

    # Get CXX stdlib path
    execute_process(
        COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libstdc++.so
        OUTPUT_VARIABLE C_PLUS_STD_LIB_PATH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    message(STATUS "C++ library location: ${C_PLUS_STD_LIB_PATH}")
endif()

# Detection of MPFR OpenMP and OpenCL
if(USE_MPFR)
    find_package(MPFR REQUIRED)
    if(MPFR_FOUND)
        message(STATUS "Found MPFR: ${MPFR_FOUND}")
        message(STATUS "Added MPFR include directories: ${MPFR_INCLUDE_DIRS}")
        message(STATUS "Added MPFR libraries: ${MPFR_LIBRARIES}")
        include_directories(${MPFR_INCLUDE_DIRS})
    else()
        message(STATUS "MPFR not found")
    endif()
endif()
if(USE_OPENMP)
    find_package(OpenMP)
    if(OpenMP_FOUND)    
        message(STATUS "Found OpenMP: ${OpenMP_FOUND}")
        message(STATUS "OpenMP include directories: ${OpenMP_INCLUDE_DIRS}")
        message(STATUS "OpenMP libraries: ${OpenMP_LIBRARIES}")
    else()
        message(STATUS "OpenMP not found")
    endif()
endif()
if(USE_OPENCL)
    find_package(OpenCL)
    if(OpenCL_FOUND)
        message(STATUS "Found OpenCL: ${OpenCL_FOUND}")
        message(STATUS "OpenCL include directories: ${OpenCL_INCLUDE_DIRS}")
        message(STATUS "OpenCL libraries: ${OpenCL_LIBRARIES}")
    else()
        message(STATUS "OpenCL not found")
    endif()
endif()

# Detect GMP Library
find_package(GMP REQUIRED)
if(GMP_FOUND)
    # TODO： Clearify the MP means, and update setup code to use GMP instead of IMATH.
    set(USE_GMP_FOR_MP ON)
    set(USE_IMATH_FOR_MP OFF)
    set(USE_SMALL_INT_OPT OFF)
    message(STATUS "Found GMP: ${GMP_FOUND}, USE_GMP_FOR_MP=${USE_GMP_FOR_MP}")
    message(STATUS "Added GMP include directories: ${GMP_INCLUDE_DIRS}")
    message(STATUS "Added GMP libraries: ${GMP_LIBRARIES}")
    include_directories(${GMP_INCLUDE_DIRS})
else()
    # TODO： Clearify the MP means, and update setup code to use GMP instead of IMATH.
    set(USE_GMP_FOR_MP OFF)
    set(USE_IMATH_FOR_MP ON)
    set(USE_SMALL_INT_OPT ON)
    message(STATUS "GMP not found, USE_GMP_FOR_MP=${USE_GMP_FOR_MP}")
endif()

# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${AUTOSTASH_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${AUTOSTASH_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${AUTOSTASH_BINARY_DIR}/lib)
message(STATUS "Binary output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Library output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Archive output directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

# Set default installation prefix
set(CMAKE_INSTALL_PREFIX ${AUTOSTASH_SOURCE_DIR}/install)
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Set RPATH settings to ensure installed binaries can find shared libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)


# Determine compiler characteristics
# (These are referenced from Polly project configuration)
include(CheckCXXSourceCompiles)
include(CheckCSourceCompiles)
include(CheckIncludeFileCXX)
include(CheckIncludeFile)
include(CheckSymbolExists)
include(CheckTypeSize)

# Like check_c_source_compiles, but sets the result to either
# 0 (error while compiling) or 1 (compiled successfully)
# Required for compatibility with autotool's AC_CHECK_DECLS
function (check_c_source_compiles_numeric _prog _var)
    check_c_source_compiles("${_prog}" "${_var}")
    if ("${${_var}}")
        set("${_var}" 1 PARENT_SCOPE)
    else()
        set("${_var}" 0 PARENT_SCOPE)
    endif ()
endfunction ()
function (check_cxx_source_compiles_numeric _prog _var)
    check_cxx_source_compiles("${_prog}" "${_var}")
    if ("${${_var}}")
        set("${_var}" 1 PARENT_SCOPE)
    else()
        set("${_var}" 0 PARENT_SCOPE)
    endif ()
endfunction ()

# Check for the existance of a type
function (check_c_type_exists _type _files _variable)
    set(_includes "")
    foreach (file_name ${_files})
        set(_includes "${_includes}#include<${file_name}>\n")
    endforeach()
    check_c_source_compiles("
    ${_includes}
    ${_type} typeVar;
    int main(void) {
    return 0;
    }
    " ${_variable})
endfunction ()
