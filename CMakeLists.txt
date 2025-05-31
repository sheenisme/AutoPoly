cmake_minimum_required(VERSION 3.10)
project(AutoStash LANGUAGES C CXX)

# Set C/C++ standards
set(CMAKE_C_STANDARD 17)
set(CMAKE_CXX_STANDARD 17)

# # Configure compiler by clang and clang++
set(CMAKE_C_COMPILER clang CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER clang++ CACHE STRING "C++ compiler" FORCE)
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

# ============================================================================
# Compilation options (default: ON)
# ============================================================================
# General build options
option(BUILD_SHARED_LIBS "Build shared libraries" OFF)
option(USE_MPFR   "Enable MPFR   support" OFF)
option(USE_OPENMP "Enable OpenMP support" ON)
option(USE_OPENCL "Enable OpenCL support" ON)

# Third-party library build options
# These options control whether to build libraries from source or use external installations
option(BUILD_ISL "Build ISL library from source" ON)
option(BUILD_PET "Build PET library from source" ON)
option(BUILD_PET_EXECUTABLE "Build PET executable" ON)

# External library paths (used when not building from source)
# If BUILD_ISL or BUILD_PET is OFF, these paths must be set to valid installation directories
set(EXTERNAL_ISL_DIR "" CACHE PATH "Path to external ISL installation directory")
set(EXTERNAL_PET_DIR "" CACHE PATH "Path to external PET installation directory")

# Notes:PPCG is always built from source (not configurable)
# This ensures compatibility with the project and prevents linking to incompatible versions


# Set module search path for some dependencies tools
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

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

# Add LLVM/Clang detection
find_package(LLVM REQUIRED CONFIG)
if(LLVM_FOUND)
    message(STATUS "Found LLVM: ${LLVM_FOUND}")
    message(STATUS "LLVM include directories: ${LLVM_INCLUDE_DIRS}")
else()
    message(STATUS "LLVM not found")
endif()
find_package(Clang REQUIRED CONFIG)
if(Clang_FOUND)
    message(STATUS "Found Clang: ${Clang_FOUND}")
    message(STATUS "Clang include directories: ${CLANG_INCLUDE_DIRS}")
else()
    message(STATUS "Clang not found")
endif()

if(LLVM_FOUND)
    set(LLVM_REQUIRED_LIBS
        LLVMSupport
        LLVMCore
        LLVMBinaryFormat
        LLVMRemarks
        LLVMBitstreamReader
        LLVMDemangle
        LLVMTargetParser
        LLVMTextAPI
    )
    list(APPEND CMAKE_REQUIRED_INCLUDES ${LLVM_INCLUDE_DIRS})
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${LLVM_REQUIRED_LIBS})
    message(STATUS "Appended LLVM include: ${LLVM_INCLUDE_DIRS}")
    message(STATUS "Appended LLVM libraries: ${LLVM_REQUIRED_LIBS}")
endif()

if(Clang_FOUND)
    set(CLANG_REQUIRED_LIBS
        clangBasic
        clangLex
        clangFrontend
        clangSerialization
        clangParse 
        clangSema
        clangAnalysis
        clangAST
        clangEdit
        clangASTMatchers
        clangDriver
        clangSupport
    )
    list(APPEND CMAKE_REQUIRED_INCLUDES ${CLANG_INCLUDE_DIRS})
    list(APPEND CMAKE_REQUIRED_LIBRARIES ${CLANG_REQUIRED_LIBS})
    message(STATUS "Appended Clang include: ${CLANG_INCLUDE_DIRS}")
    message(STATUS "Appended Clang libraries: ${CLANG_REQUIRED_LIBS}")
endif()

# Set default build type to Debug
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Set output directories for binaries and libraries
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
message(STATUS "Binary output directory: ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
message(STATUS "Library output directory: ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
message(STATUS "Archive output directory: ${CMAKE_ARCHIVE_OUTPUT_DIRECTORY}")

# Set default installation prefix
set(CMAKE_INSTALL_PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/install)
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")

# Set RPATH settings to ensure installed binaries can find shared libraries
set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH FALSE)


# ============================================================================
# Third-party library paths configuration
# ============================================================================

# Set PPCG directory (always required and built from source)
set(PPCG_DIR ${CMAKE_CURRENT_SOURCE_DIR}/ppcg)

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

# ============================================================================
# ISL Library Configuration
# ============================================================================
# ISL (Integer Set Library) is a library for manipulating sets and relations of integer points
# bounded by linear constraints. It's a core dependency for PPCG and PET.
# Configure ISL library
if(BUILD_ISL)
    set(ISL_DIR ${PPCG_DIR}/isl)
    message(STATUS "Using ISL from source at: ${ISL_DIR}")

    ########################################################################
    #    Check compiler characteristics to set ISL compilation options     #
    #    (Referenced from Polly project configuration)                     #
    ########################################################################
    check_c_source_compiles("
    int func(void) __attribute__((__warn_unused_result__));
    int main(void) { return 0; }
    " HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
    set(GCC_WARN_UNUSED_RESULT)
    if (HAS_ATTRIBUTE_WARN_UNUSED_RESULT)
        set(GCC_WARN_UNUSED_RESULT "__attribute__((__warn_unused_result__))")
    endif ()

    check_c_source_compiles("
    __attribute__ ((unused)) static void foo(void);
    int main(void) { return 0; }
    " HAVE___ATTRIBUTE__)

    check_c_source_compiles_numeric("
    #include <strings.h>
    int main(void) { (void)ffs(0); return 0; }
    " HAVE_DECL_FFS)

    check_c_source_compiles_numeric("
    int main(void) { (void)__builtin_ffs(0); return 0; }
    " HAVE_DECL___BUILTIN_FFS)

    check_c_source_compiles_numeric("
    #include <intrin.h>
    int main(void) { (void)_BitScanForward(NULL, 0); return 0; }
    " HAVE_DECL__BITSCANFORWARD)

    if (NOT HAVE_DECL_FFS AND
        NOT HAVE_DECL___BUILTIN_FFS AND
        NOT HAVE_DECL__BITSCANFORWARD)
        message(FATAL_ERROR "No ffs implementation found")
    endif ()

    check_c_source_compiles_numeric("
    #include <strings.h>
    int main(void) { (void)strcasecmp(\"\", \"\"); return 0; }
    " HAVE_DECL_STRCASECMP)

    check_c_source_compiles_numeric("
    #include <string.h>
    int main(void) { (void)_stricmp(\"\", \"\"); return 0; }
    " HAVE_DECL__STRICMP)

    if (NOT HAVE_DECL_STRCASECMP AND NOT HAVE_DECL__STRICMP)
        message(FATAL_ERROR "No strcasecmp implementation found")
    endif ()

    check_c_source_compiles_numeric("
    #include <strings.h>
    int main(void) { (void)strncasecmp(\"\", \"\", 0); return 0; }
    " HAVE_DECL_STRNCASECMP)

    check_c_source_compiles_numeric("
    #include <string.h>
    int main(void) { (void)_strnicmp(\"\", \"\", 0); return 0; }
    " HAVE_DECL__STRNICMP)

    if (NOT HAVE_DECL_STRNCASECMP AND NOT HAVE_DECL__STRNICMP)
        message(FATAL_ERROR "No strncasecmp implementation found")
    endif ()

    check_c_source_compiles_numeric("
    #include <stdio.h>
    int main(void) { snprintf((void*)0, 0, \" \"); return 0; }
    " HAVE_DECL_SNPRINTF)

    check_c_source_compiles_numeric("
    #include <stdio.h>
    int main(void) { _snprintf((void*)0, 0, \" \"); return 0; }
    " HAVE_DECL__SNPRINTF)

    if (NOT HAVE_DECL_SNPRINTF AND NOT HAVE_DECL__SNPRINTF)
        message(FATAL_ERROR "No snprintf implementation found")
    endif ()

    # Create stdint.h file (Referenced from Polly project configuration)
    check_c_type_exists(uint8_t "" HAVE_UINT8T)
    check_c_type_exists(uint8_t "stdint.h" HAVE_STDINT_H)
    check_c_type_exists(uint8_t "inttypes.h" HAVE_INTTYPES_H)
    check_c_type_exists(uint8_t "sys/types.h" HAVE_SYS_INTTYPES_H)
    if (HAVE_UINT8T)
        set(INCLUDE_STDINT_H "")
    elseif (HAVE_STDINT_H)
        set(INCLUDE_STDINT_H "#include <stdint.h>")
    elseif (HAVE_INTTYPES_H)
        set(INCLUDE_STDINT_H "#include <inttypes.h>")
    elseif (HAVE_SYS_INTTYPES_H)
        set(INCLUDE_STDINT_H "#include <sys/inttypes.h>")
    else()
        message(FATAL_ERROR "No stdint.h or compatible found")
    endif ()
    file(WRITE "${CMAKE_BINARY_DIR}/include/isl/stdint.h.tmp"
        "${INCLUDE_STDINT_H}\n")
    configure_file("${CMAKE_BINARY_DIR}/include/isl/stdint.h.tmp"
        "${CMAKE_BINARY_DIR}/include/isl/isl/stdint.h" COPYONLY)
    
    # Create ISL configuration header file
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/isl_config.h.cmake" "${CMAKE_BINARY_DIR}/include/isl/isl_config.h")

    # Create gitversion.h file (required by ISL)
    execute_process(
        COMMAND git describe --always 
        WORKING_DIRECTORY ${ISL_DIR}
        OUTPUT_VARIABLE ISL_GIT_HEAD_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    file(WRITE ${CMAKE_BINARY_DIR}/include/isl/gitversion.h "#define GIT_HEAD_ID \"${ISL_GIT_HEAD_VERSION}\"")
    message(STATUS "ISL version: ${ISL_GIT_HEAD_VERSION}")

    # Set ISL Personal include directory
    include_directories(isl PRIVATE ${CMAKE_BINARY_DIR}/include/isl)

    # Include ISL headers directories
    include_directories(isl PUBLIC
        ${ISL_DIR}
        ${ISL_DIR}/imath
        ${ISL_DIR}/include
    )
else()
    if(EXTERNAL_ISL_DIR STREQUAL "")
        message(FATAL_ERROR "EXTERNAL_ISL_DIR must be set when BUILD_ISL is OFF")
    endif()
    set(ISL_DIR ${EXTERNAL_ISL_DIR})
    message(STATUS "Using external ISL from: ${ISL_DIR}")

    # Include ISL headers
    include_directories(isl PUBLIC
        ${ISL_DIR}
        ${ISL_DIR}/imath
        ${ISL_DIR}/include
    )
endif()

# Build ISL library
if(BUILD_ISL)
    # Build ISL from source
    message(STATUS "Configuring ISL library build from source")
    
    # ISL library source files - based on Polly project configuration
    set(ISL_SOURCES
        ${ISL_DIR}/isl_aff.c
        ${ISL_DIR}/isl_aff_map.c
        ${ISL_DIR}/isl_affine_hull.c
        ${ISL_DIR}/isl_arg.c
        ${ISL_DIR}/isl_ast.c
        ${ISL_DIR}/isl_ast_build.c
        ${ISL_DIR}/isl_ast_build_expr.c
        ${ISL_DIR}/isl_ast_codegen.c
        ${ISL_DIR}/isl_ast_graft.c
        ${ISL_DIR}/basis_reduction_tab.c
        ${ISL_DIR}/isl_bernstein.c
        ${ISL_DIR}/isl_blk.c
        ${ISL_DIR}/isl_bound.c
        ${ISL_DIR}/isl_box.c
        ${ISL_DIR}/isl_coalesce.c
        ${ISL_DIR}/isl_constraint.c
        ${ISL_DIR}/isl_convex_hull.c
        ${ISL_DIR}/isl_ctx.c
        ${ISL_DIR}/isl_deprecated.c
        ${ISL_DIR}/isl_dim_map.c
        ${ISL_DIR}/isl_equalities.c
        ${ISL_DIR}/isl_factorization.c
        ${ISL_DIR}/isl_farkas.c
        ${ISL_DIR}/isl_ffs.c
        ${ISL_DIR}/isl_flow.c
        ${ISL_DIR}/isl_fold.c
        ${ISL_DIR}/isl_hash.c
        ${ISL_DIR}/isl_id_to_ast_expr.c
        ${ISL_DIR}/isl_id_to_id.c
        ${ISL_DIR}/isl_id_to_pw_aff.c
        ${ISL_DIR}/isl_ilp.c
        ${ISL_DIR}/isl_input.c
        ${ISL_DIR}/isl_local.c
        ${ISL_DIR}/isl_local_space.c
        ${ISL_DIR}/isl_lp.c
        ${ISL_DIR}/isl_map.c
        ${ISL_DIR}/isl_map_list.c
        ${ISL_DIR}/isl_map_simplify.c
        ${ISL_DIR}/isl_map_subtract.c
        ${ISL_DIR}/isl_map_to_basic_set.c
        ${ISL_DIR}/isl_mat.c
        ${ISL_DIR}/isl_morph.c
        ${ISL_DIR}/isl_id.c
        ${ISL_DIR}/isl_obj.c
        ${ISL_DIR}/isl_options.c
        ${ISL_DIR}/isl_output.c
        ${ISL_DIR}/isl_point.c
        ${ISL_DIR}/isl_polynomial.c
        ${ISL_DIR}/isl_printer.c
        ${ISL_DIR}/print.c
        ${ISL_DIR}/isl_range.c
        ${ISL_DIR}/isl_reordering.c
        ${ISL_DIR}/isl_sample.c
        ${ISL_DIR}/isl_scan.c
        ${ISL_DIR}/isl_schedule.c
        ${ISL_DIR}/isl_schedule_band.c
        ${ISL_DIR}/isl_schedule_node.c
        ${ISL_DIR}/isl_schedule_read.c
        ${ISL_DIR}/isl_schedule_tree.c
        ${ISL_DIR}/isl_schedule_constraints.c
        ${ISL_DIR}/isl_scheduler.c
        ${ISL_DIR}/isl_scheduler_clustering.c
        ${ISL_DIR}/isl_scheduler_scc.c
        ${ISL_DIR}/isl_set_list.c
        ${ISL_DIR}/isl_sort.c
        ${ISL_DIR}/isl_space.c
        ${ISL_DIR}/isl_stream.c
        ${ISL_DIR}/isl_seq.c
        ${ISL_DIR}/isl_set_to_ast_graft_list.c
        ${ISL_DIR}/isl_stride.c
        ${ISL_DIR}/isl_tab.c
        ${ISL_DIR}/isl_tab_pip.c
        ${ISL_DIR}/isl_tarjan.c
        ${ISL_DIR}/isl_transitive_closure.c
        ${ISL_DIR}/isl_union_map.c
        ${ISL_DIR}/isl_val.c
        ${ISL_DIR}/isl_vec.c
        ${ISL_DIR}/isl_version.c
        ${ISL_DIR}/isl_vertices.c
        # GMP related files
        ${ISL_DIR}/isl_gmp.c
        ${ISL_DIR}/isl_val_gmp.c
        # Template implemented C files
        ${ISL_DIR}/set_to_map.c
        ${ISL_DIR}/set_from_map.c
        ${ISL_DIR}/uset_to_umap.c
        ${ISL_DIR}/uset_from_umap.c
    )
    
    # Build ISL library
    add_library(isl ${ISL_SOURCES})
    if(GMP_FOUND)
        target_link_libraries(isl ${GMP_LIBRARIES})
    endif()
    if(MPFR_FOUND)
        target_link_libraries(isl ${MPFR_LIBRARIES})
    endif()

    # Add debug options to ISL for better error diagnosis
    target_compile_options(isl PRIVATE -g)

    # Set installation rules for ISL
    install(TARGETS isl
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(DIRECTORY ${ISL_DIR}/include/ DESTINATION include/)
else()
    # Use external ISL library
    message(STATUS "Using external ISL library from: ${ISL_DIR}")
    
    # Find the ISL library in the specified directory
    find_library(ISL_LIBRARY isl PATHS ${ISL_DIR}/lib ${ISL_DIR}/lib64 NO_DEFAULT_PATH)
    
    if(NOT ISL_LIBRARY)
        message(FATAL_ERROR "Could not find ISL library in ${ISL_DIR}/lib or ${ISL_DIR}/lib64")
    endif()

    # Create imported target for ISL
    add_library(isl UNKNOWN IMPORTED)
    set_target_properties(isl PROPERTIES IMPORTED_LOCATION ${ISL_LIBRARY})
    message(STATUS "Found external ISL library: ${ISL_LIBRARY}")

    # Add include directories for external ISL
    include_directories(${ISL_DIR}/include)
endif()


# ============================================================================
# PET Library Configuration
# ============================================================================
# PET (Polyhedral Extraction Tool) is a library for extracting polyhedral models from C source code.
# It uses Clang for parsing C code and provides a polyhedral representation that can be used by PPCG.
if(BUILD_PET)
    set(PET_DIR ${PPCG_DIR}/pet)
    message(STATUS "Using PET from source at: ${PET_DIR}")

    # Create gitversion.h file (required by PET)
    execute_process(
        COMMAND git describe --always 
        WORKING_DIRECTORY ${PET_DIR}
        OUTPUT_VARIABLE PET_GIT_HEAD_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(NOT PET_GIT_HEAD_VERSION)
        set(PET_GIT_HEAD_VERSION "unknown")
    endif()
    file(WRITE ${CMAKE_BINARY_DIR}/include/pet/gitversion.h "#define GIT_HEAD_ID \"${PET_GIT_HEAD_VERSION}\"")

    # Set PET version
    string(REGEX REPLACE "^pet-([0-9.]+).*" "\\1" PET_VERSION "${PET_GIT_HEAD_VERSION}")
    if(NOT PET_VERSION OR PET_VERSION STREQUAL PET_GIT_HEAD_VERSION)
        set(PET_VERSION "0.11.8")  # fallback version
    endif()
    message(STATUS "PET version: ${PET_VERSION}")

    ########################################################################
    # Check compiler characteristics to set PET compilation options        #
    ########################################################################
    message(STATUS "Checking Clang features for PET compatibility")

    # Set up compiler flags for Clang detection
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "AppleClang")
        set(CMAKE_REQUIRED_FLAGS "${CMAKE_REQUIRED_FLAGS} -Wno-error")
    endif()

    # Find LLVM and Clang prefix
    if(LLVM_FOUND)
        set(CLANG_PREFIX ${LLVM_INSTALL_PREFIX})
    else()
        # Try to find clang executable and derive prefix
        find_program(CLANG_EXECUTABLE clang)
        if(CLANG_EXECUTABLE)
            get_filename_component(CLANG_BIN_DIR ${CLANG_EXECUTABLE} DIRECTORY)
            get_filename_component(CLANG_PREFIX ${CLANG_BIN_DIR} DIRECTORY)
        else()
            set(CLANG_PREFIX "/usr")
        endif()
    endif()
    message(STATUS "CLANG_PREFIX: ${CLANG_PREFIX}")

    # Check for standard headers
    check_include_file("dlfcn.h" HAVE_DLFCN_H)
    check_include_file("inttypes.h" HAVE_INTTYPES_H)
    check_include_file("stdint.h" HAVE_STDINT_H)
    check_include_file("stdio.h" HAVE_STDIO_H)
    check_include_file("stdlib.h" HAVE_STDLIB_H)
    check_include_file("string.h" HAVE_STRING_H)
    check_include_file("strings.h" HAVE_STRINGS_H)
    check_include_file("sys/stat.h" HAVE_SYS_STAT_H)
    check_include_file("sys/types.h" HAVE_SYS_TYPES_H)
    check_include_file("unistd.h" HAVE_UNISTD_H)

    # Set STDC_HEADERS if all standard headers are available
    if(HAVE_STDIO_H AND HAVE_STDLIB_H AND HAVE_STRING_H AND HAVE_STDINT_H)
        set(STDC_HEADERS 1)
    endif()

    # Check for Clang-specific headers
    check_cxx_source_compiles("
        #include <clang/Basic/SourceLocation.h>
        int main() { return 0; }" 
        HAVE_BASIC_SOURCELOCATION_H)

    check_cxx_source_compiles("
        #include <clang/Basic/DiagnosticOptions.h>
        int main() { return 0; }" 
        HAVE_BASIC_DIAGNOSTICOPTIONS_H)

    check_cxx_source_compiles("
        #include <clang/Lex/HeaderSearchOptions.h>
        int main() { return 0; }" 
        HAVE_LEX_HEADERSEARCHOPTIONS_H)

    check_cxx_source_compiles("
        #include <clang/Lex/PreprocessorOptions.h>
        int main() { return 0; }" 
        HAVE_LEX_PREPROCESSOROPTIONS_H)

    check_cxx_source_compiles("
        #include <clang/Basic/LangStandard.h>
        int main() { return 0; }" 
        HAVE_CLANG_BASIC_LANGSTANDARD_H)

    check_cxx_source_compiles("
        #include <llvm/Option/Arg.h>
        int main() { return 0; }" 
        HAVE_LLVM_OPTION_ARG_H)

    check_cxx_source_compiles("
        #include <llvm/ADT/OwningPtr.h>
        int main() { return 0; }" 
        HAVE_ADT_OWNINGPTR_H)

    check_cxx_source_compiles("
        #include <llvm/TargetParser/Host.h>
        int main() { return 0; }" 
        HAVE_TARGETPARSER_HOST_H)

    # Check for method name variations (backwards compatibility)
    check_cxx_source_compiles("
        #include <llvm/TargetParser/Host.h>
        #include <clang/Basic/TargetInfo.h>
        int main() {
            auto T = llvm::sys::getDefaultTargetTriple();
            return 0;
        }"
        HAVE_GETDEFAULTTARGETTRIPPLE)
    if(NOT HAVE_GETDEFAULTTARGETTRIPPLE)
        set(getDefaultTargetTriple "getHostTriple")
    endif()

    # Check for DiagnosticInfo vs Diagnostic compatibility
    check_cxx_source_compiles("
        #include <clang/Basic/Diagnostic.h>
        int main() {
            clang::DiagnosticInfo* DI = nullptr;
            return 0;
        }"
        HAVE_DIAGNOSTIC_INFO)
    
    if(NOT HAVE_DIAGNOSTIC_INFO)
        set(DiagnosticInfo "Diagnostic")
    endif()

    # Check for DiagnosticsEngine vs Diagnostic
    check_cxx_source_compiles("
        #include <clang/Basic/Diagnostic.h>
        int main() {
            clang::DiagnosticsEngine* DE = nullptr;
            return 0;
        }"
        HAVE_DIAGNOSTICS_ENGINE)
    
    if(NOT HAVE_DIAGNOSTICS_ENGINE)
        set(DiagnosticsEngine "Diagnostic")
    endif()

    # Check HandleTopLevelDecl return type
    check_cxx_source_compiles("
        #include <clang/AST/ASTConsumer.h>
        #include <clang/AST/DeclGroup.h>
        class TestConsumer : public clang::ASTConsumer {
        public:
            bool HandleTopLevelDecl(clang::DeclGroupRef D) override { return true; }
        };
        int main() { return 0; }"
        HANDLETOPLEVELRETURN_IS_BOOL)
    
    if(HANDLETOPLEVELRETURN_IS_BOOL)
        set(HandleTopLevelDeclReturn "bool")
        set(HandleTopLevelDeclContinue "true")
    else()
        set(HandleTopLevelDeclReturn "void")
        set(HandleTopLevelDeclContinue "")
    endif()

    # Check if CreateTargetInfo takes shared_ptr
    check_cxx_source_compiles("
        #include <clang/Basic/TargetInfo.h>
        #include <clang/Basic/Diagnostic.h>
        #include <memory>
        int main() {
            std::shared_ptr<clang::TargetOptions> TO;
            clang::DiagnosticsEngine DE(nullptr, nullptr);
            clang::TargetInfo::CreateTargetInfo(DE, TO);
            return 0;
        }"
        CREATETARGETINFO_TAKES_SHARED_PTR)

    # Check if CreateTargetInfo takes pointer
    if(NOT CREATETARGETINFO_TAKES_SHARED_PTR)
        check_cxx_source_compiles("
            #include <clang/Basic/TargetInfo.h>
            #include <clang/Basic/Diagnostic.h>
            int main() {
                clang::TargetOptions TO;
                clang::DiagnosticsEngine DE(nullptr, nullptr);
                clang::TargetInfo::CreateTargetInfo(DE, &TO);
                return 0;
            }"
            CREATETARGETINFO_TAKES_POINTER)
    endif()

    # Check if Driver::BuildCompilation takes ArrayRef
    check_cxx_source_compiles("
        #include <clang/Driver/Driver.h>
        #include <llvm/ADT/ArrayRef.h>
        #include <clang/Basic/DiagnosticIDs.h>
        #include <clang/Basic/DiagnosticOptions.h>
        #include <llvm/Support/VirtualFileSystem.h>
        
        int main() { 
            clang::DiagnosticIDs diagIDs;
            clang::DiagnosticOptions diagOpts;
            clang::DiagnosticsEngine diagEngine(&diagIDs, &diagOpts);
            clang::driver::Driver D(\"\", \"\", diagEngine);
            llvm::ArrayRef<const char*> Args;
            D.BuildCompilation(Args);
            return 0; 
        }"
        USE_ARRAYREF)

    # Check if CompilerInvocation::CreateFromArgs takes ArrayRef
    check_cxx_source_compiles("
        #include <clang/Frontend/CompilerInvocation.h>
        #include <llvm/ADT/ArrayRef.h>
        #include <memory>

        int main() { 
            llvm::ArrayRef<const char*> Args;
            clang::DiagnosticsEngine DE(nullptr, nullptr);
            auto invocation = std::make_shared<clang::CompilerInvocation>();
            clang::CompilerInvocation::CreateFromArgs(*invocation, Args, DE);
            return 0; 
        }"
        CREATE_FROM_ARGS_TAKES_ARRAYREF)
    
    check_cxx_source_compiles("
        #include <clang/Lex/HeaderSearchOptions.h>
        int main() {
            using namespace clang;
            clang::HeaderSearchOptions HSO;
            HSO.AddPath(\"\", clang::frontend::Angled, false, false);
            return 0;
        }" ADDPATH_TAKES_4_ARGUMENTS)

    # Check if CompilerInstance::createPreprocessor takes TranslationUnitKind
    check_cxx_source_compiles("
        #include <clang/Frontend/CompilerInstance.h>
        int main() {
            clang::CompilerInstance CI;
            CI.createPreprocessor(clang::TU_Complete);
            return 0;
        }"
        CREATEPREPROCESSOR_TAKES_TUKIND)

    # Check for DecayedType availability
    check_cxx_source_compiles("
        #include <clang/AST/Type.h>
        int main() { 
            clang::DecayedType* DT = nullptr;
            return 0; 
        }"
        HAVE_DECAYEDTYPE)

    # Check if SourceManager has setMainFileID method
    check_cxx_source_compiles("
        #include <clang/Basic/SourceManager.h>
        int main() {
            clang::SourceManager* SM = nullptr;
            SM->setMainFileID(clang::FileID());
            return 0;
        }"
        HAVE_SETMAINFILEID)

    # Check if SourceManager has findLocationAfterToken method
    check_cxx_source_compiles("
        #include <clang/Lex/Lexer.h>
        #include <clang/Basic/SourceManager.h>
        
        int main() {
            clang::SourceManager* SM = nullptr;
            clang::SourceLocation SL;
            clang::Lexer::findLocationAfterToken(SL, clang::tok::semi, *SM, clang::LangOptions(), false);
            return 0;
        }"
        HAVE_FINDLOCATIONAFTERTOKEN)

    # Check if SourceManager has translateLineCol method
    check_cxx_source_compiles("
        #include <clang/Basic/SourceManager.h>
        int main() {
            clang::SourceManager* SM = nullptr;
            SM->translateLineCol(clang::FileID(), 1, 1);
            return 0;
        }"
        HAVE_TRANSLATELINECOL)

    # Check if ASTContext::getTypeInfo returns TypeInfo
    check_cxx_source_compiles("
        #include <clang/AST/ASTContext.h>
        int main() {
            clang::ASTContext* AC = nullptr;
            clang::TypeInfo TI = AC->getTypeInfo(clang::QualType());
            return 0;
        }"
        GETTYPEINFORETURNSTYPEINFO)

    # Check for setLangDefaults location and signature
    check_cxx_source_compiles("
        #include <clang/Basic/LangOptions.h>
        #include <clang/Basic/TargetInfo.h>
        #include <vector>
        #include <string>

        int main() { 
            clang::LangOptions LO;
            std::vector<std::string> Includes;
            clang::LangOptions::setLangDefaults(LO, clang::Language::C, llvm::Triple(), 
                                            Includes, clang::LangStandard::lang_unspecified); 
            return 0; 
        }" 
        SETLANGDEFAULTS_IS_LANGOPTIONS)

    if(SETLANGDEFAULTS_IS_LANGOPTIONS)
        set(SETLANGDEFAULTS "LangOptions")
    else()
        set(SETLANGDEFAULTS "CompilerInvocation")
    endif()

    # Check for InputKind::Language definition
    check_cxx_source_compiles("
        #include <clang/Frontend/FrontendOptions.h>
        int main() { 
            clang::Language L = clang::Language::C; 
            return 0; 
        }" 
        IK_C_IS_LANGUAGE_C)
    if(IK_C_IS_LANGUAGE_C)
        set(IK_C "Language::C")
    else()
        set(IK_C "InputKind::C")
    endif()

    # Check if setLangDefaults takes 5 arguments
    check_cxx_source_compiles("
        #include <clang/Basic/TargetOptions.h>
        #include <clang/Lex/PreprocessorOptions.h>
        #include <clang/Frontend/CompilerInstance.h>
        #include <string>
        #include <vector>

        struct setLangDefaultsArg4 {
            setLangDefaultsArg4(clang::PreprocessorOptions &PO) : PO(PO) {}
            operator clang::PreprocessorOptions &() { return PO; }
            operator std::vector<std::string> &() { return PO.Includes; }

            clang::PreprocessorOptions &PO;
        };
        int main() {
            using namespace clang;
            CompilerInstance *Clang;
            TargetOptions TO;
            llvm::Triple T(TO.Triple);
            PreprocessorOptions PO;
            ${SETLANGDEFAULTS}::setLangDefaults(Clang->getLangOpts(), ${IK_C},
                    T, setLangDefaultsArg4(PO),
                    LangStandard::lang_unspecified);
            return 0;
        }"
        SETLANGDEFAULTS_TAKES_5_ARGUMENTS)

    # Check if CompilerInstance::setInvocation takes shared_ptr
    check_cxx_source_compiles("
        #include <clang/Frontend/CompilerInstance.h>
        #include <clang/Frontend/CompilerInvocation.h>
        #include <memory>
        int main() {
            clang::CompilerInstance CI;
            auto invocation = std::make_shared<clang::CompilerInvocation>();
            CI.setInvocation(invocation);
            return 0;
        }"
        SETINVOCATION_TAKES_SHARED_PTR)

    # Check for getBeginLoc/getEndLoc methods
    check_cxx_source_compiles("
        #include <clang/AST/Decl.h>
        int main() {
            clang::FunctionDecl* FD = nullptr;
            clang::SourceLocation begin = FD->getBeginLoc();
            clang::SourceLocation end = FD->getEndLoc();
            return 0;
        }"
        HAVE_BEGIN_END_LOC)

    # Check if DiagnosticsEngine has setDiagnosticGroupWarningAsError
    check_cxx_source_compiles("
        #include <clang/Basic/Diagnostic.h>
        int main() {
            clang::DiagnosticsEngine DE(nullptr, nullptr);
            DE.setDiagnosticGroupWarningAsError(\"test\", true);
            return 0;
        }"
        HAVE_SET_DIAGNOSTIC_GROUP_WARNING_AS_ERROR)

    # Check for ext_implicit_function_decl_c99
    check_cxx_source_compiles("
        #include <clang/Basic/DiagnosticCategories.h>
        int main() {
            auto DiagID = clang::diag::ext_implicit_function_decl_c99;
            return 0;
        }"
        HAVE_EXT_IMPLICIT_FUNCTION_DECL_C99)
    if(NOT HAVE_EXT_IMPLICIT_FUNCTION_DECL_C99)
        set(ext_implicit_function_decl_c99 "ext_implicit_function_decl")
    endif()

    # Check for StmtRange class
    check_cxx_source_compiles("
        #include <clang/AST/StmtIterator.h>
        int main() { 
            clang::StmtRange SR;
            return 0; 
        }"
        HAVE_STMTRANGE)

    # Check for nested ArraySizeModifier in ArrayType
    check_cxx_source_compiles("
        #include <clang/AST/Type.h>
        int main() {
            clang::ArrayType::ArraySizeModifier ASM;
            return 0;
        }"
        USE_NESTED_ARRAY_SIZE_MODIFIER)

    # Check if CompilerInstance::createDiagnostics takes argc/argv
    check_cxx_source_compiles("
        #include <clang/Frontend/CompilerInstance.h>
        int main() {
            using namespace clang;
            CompilerInstance *Clang;
            Clang->createDiagnostics();
            return 0;
        }"
        NO_CREATEDIAGNOSTICS_TAKES_ARG)
    if(NOT NO_CREATEDIAGNOSTICS_TAKES_ARG)
        set(CREATEDIAGNOSTICS_TAKES_ARG "1")
    endif()

    # Check for DiagnosticConsumer vs DiagnosticClient
    check_cxx_source_compiles("
        #include <clang/Basic/Diagnostic.h>
        int main() {
            clang::DiagnosticConsumer* DC = nullptr;
            return 0;
        }"
        HAVE_DIAGNOSTIC_CONSUMER)
    if(NOT HAVE_DIAGNOSTIC_CONSUMER)
        set(DiagnosticConsumer "DiagnosticClient")
    endif()

    # Check for TypedefNameDecl vs TypedefDecl
    check_cxx_source_compiles("
        #include <clang/AST/Decl.h>
        int main() {
            clang::TypedefNameDecl* TND = nullptr;
            return 0;
        }"
        HAVE_TYPEDEFNAMEDECL)
    if(NOT HAVE_TYPEDEFNAMEDECL)
        set(TypedefNameDecl "TypedefDecl")
    endif()

    # Check for PragmaIntroducer vs PragmaIntroducerKind
    check_cxx_source_compiles("
        #include <clang/Lex/Pragma.h>
        int main() {
            clang::PragmaIntroducer PI;
            return 0;
        }"
        HAVE_PRAGMAINTRODUCER)
    if(NOT HAVE_PRAGMAINTRODUCER)
        set(PragmaIntroducer "PragmaIntroducerKind")
    endif()

    # Check for expansion vs instantiation methods
    check_cxx_source_compiles("
        #include <clang/Basic/SourceManager.h>
        int main() {
            using namespace clang;
            SourceManager *sm = nullptr;
            SourceLocation sl;
            sm->getExpansionColumnNumber(sl);
            return 0;
        }"
        HAVE_GETEXPANSIONCOLUMNNUMBER)

    check_cxx_source_compiles("
        #include <clang/Basic/SourceManager.h>
        int main() {
            using namespace clang;
            SourceManager *sm = nullptr;
            SourceLocation sl;
            sm->getExpansionLineNumber(sl);
            return 0;
        }"
        HAVE_GETEXPANSIONLINENUMBER)

    check_cxx_source_compiles("
        #include <clang/Basic/SourceManager.h>
        int main() {
            using namespace clang;
            SourceManager *sm = nullptr;
            SourceLocation sl;
            sm->getExpansionLoc(sl);
            return 0;
        }"
        HAVE_GETEXPANSIONLOC)
    if(NOT HAVE_GETEXPANSIONCOLUMNNUMBER)
        set(getExpansionColumnNumber "getInstantiationColumnNumber")
    endif()
    if(NOT HAVE_GETEXPANSIONLINENUMBER)
        set(getExpansionLineNumber "getInstantiationLineNumber")
    endif()
    if(NOT HAVE_GETEXPANSIONLOC)
        set(getExpansionLoc "getInstantiationLoc")
    endif()

    # Check for getLangOpts vs getLangOptions
    check_cxx_source_compiles("
        #include <clang/Frontend/CompilerInstance.h>
        int main() {
            clang::CompilerInstance CI;
            const clang::LangOptions& LO = CI.getLangOpts();
            return 0;
        }"
        HAVE_GETLANGOPTS)
    if(NOT HAVE_GETLANGOPTS)
        set(getLangOpts "getLangOptions")
    endif()

    # Check for getLocWithOffset vs getFileLocWithOffset
    check_cxx_source_compiles("
        #include <clang/Basic/SourceManager.h>
        int main() {
            clang::SourceLocation *loc = nullptr;
            auto res = loc->getLocWithOffset(1);
            return 0;
        }"
        HAVE_GETLOCWITHOFFSET)
    if(NOT HAVE_GETLOCWITHOFFSET)
        set(getLocWithOffset "getFileLocWithOffset")
    endif()

    # Check for getReturnType vs getResultType
    check_cxx_source_compiles("
        #include <clang/AST/Type.h>
        int main() {
            clang::FunctionType* FT = nullptr;
            clang::QualType QT = FT->getReturnType();
            return 0;
        }"
        HAVE_GETRETURNTYPE)
    if(NOT HAVE_GETRETURNTYPE)
        set(getReturnType "getResultType")
    endif()

    # Check for getTypedefNameForAnonDecl vs getTypedefForAnonDecl
    check_cxx_source_compiles("
        #include <clang/AST/Decl.h>
        int main() {
            clang::RecordDecl *decl = nullptr;
            auto res = decl->getTypedefNameForAnonDecl();
            return 0;
        }"
        HAVE_GETTYPEDEFNAMEFORANONDECL)
    if(NOT HAVE_GETTYPEDEFNAMEFORANONDECL)
        set(getTypedefNameForAnonDecl "getTypedefForAnonDecl")
    endif()

    # Check for initializeBuiltins vs InitializeBuiltins
    check_cxx_source_compiles("
        #include <clang/Basic/Builtins.h>
        #include <clang/Basic/LangOptions.h>
        #include <clang/Lex/Preprocessor.h>
        int main() {
            clang::LangOptions LO;
            clang::Preprocessor* PP = nullptr;
            PP->getBuiltinInfo().initializeBuiltins(PP->getIdentifierTable(), LO);
            return 0;
        }"
        HAVE_INITIALIZEBUILTINS)
    if(NOT HAVE_INITIALIZEBUILTINS)
        set(initializeBuiltins "InitializeBuiltins")
    endif()

    # Handle true/false for boolean flags in CMake
    if(STDC_HEADERS)
        set(STDC_HEADERS 1)
    endif()
    if(HAVE_DLFCN_H)
        set(HAVE_DLFCN_H 1)
    endif()
    if(HAVE_INTTYPES_H)
        set(HAVE_INTTYPES_H 1)
    endif()
    if(HAVE_STDINT_H)
        set(HAVE_STDINT_H 1)
    endif()
    if(HAVE_STDIO_H)
        set(HAVE_STDIO_H 1)
    endif()
    if(HAVE_STDLIB_H)
        set(HAVE_STDLIB_H 1)
    endif()
    if(HAVE_STRINGS_H)
        set(HAVE_STRINGS_H 1)
    endif()
    if(HAVE_STRING_H)
        set(HAVE_STRING_H 1)
    endif()
    if(HAVE_SYS_STAT_H)
        set(HAVE_SYS_STAT_H 1)
    endif()
    if(HAVE_SYS_TYPES_H)
        set(HAVE_SYS_TYPES_H 1)
    endif()
    if(HAVE_UNISTD_H)
        set(HAVE_UNISTD_H 1)
    endif()

    # Create PET configuration header file
    configure_file("${CMAKE_CURRENT_SOURCE_DIR}/cmake/pet_config.h.cmake" "${CMAKE_BINARY_DIR}/include/pet/config.h")

    # Include PET configuration directory
    include_directories(pet PRIVATE ${CMAKE_BINARY_DIR}/include/pet)

    # Include PET headers directories
    include_directories(pet PRIVATE
        ${PET_DIR}/include
        ${ISL_DIR}/include
    )

    # Add Clang dependency libraries
    set(CLANG_REQUIRED_LIBS
        clangFrontend
        clangSerialization
        clangParse 
        clangSema
        clangAnalysis
        clangAST
        clangLex
        clangEdit
        clangASTMatchers
        clangDriver
        clangBasic
        clangSupport
    )

    message(STATUS "PET Clang compatibility checks completed")

else()
    if(EXTERNAL_PET_DIR STREQUAL "")
        message(FATAL_ERROR "EXTERNAL_PET_DIR must be set when BUILD_PET is OFF")
    endif()
    set(PET_DIR ${EXTERNAL_PET_DIR})
    message(STATUS "Using external PET from: ${PET_DIR}")
endif()

if(BUILD_PET)
    # Build PET from source
    message(STATUS "Configuring PET library build from source")

    # Complete PET source file list (according to Makefile.am)
    set(PET_SOURCES
        ${PET_DIR}/aff.c
        ${PET_DIR}/array.c
        ${PET_DIR}/clang.cc
        ${PET_DIR}/context.c
        ${PET_DIR}/expr.c
        ${PET_DIR}/expr_arg.c
        ${PET_DIR}/expr_plus.cc
        ${PET_DIR}/filter.c
        ${PET_DIR}/id.cc
        ${PET_DIR}/isl_id_to_pet_expr.c
        ${PET_DIR}/inlined_calls.cc
        ${PET_DIR}/inliner.cc
        ${PET_DIR}/killed_locals.cc
        ${PET_DIR}/loc.c
        ${PET_DIR}/nest.c
        ${PET_DIR}/options.c
        ${PET_DIR}/patch.c
        ${PET_DIR}/pet_expr_to_isl_pw_aff.c
        ${PET_DIR}/print.c
        ${PET_DIR}/tree.c
        ${PET_DIR}/tree2scop.c
        ${PET_DIR}/scan.cc
        ${PET_DIR}/scop.c
        ${PET_DIR}/scop_plus.cc
        ${PET_DIR}/skip.c
        ${PET_DIR}/substituter.cc
        ${PET_DIR}/summary.c
        ${PET_DIR}/value_bounds.c
        ${PET_DIR}/version.cc
        ${PET_DIR}/pet.cc
    )

    # Build pet library
    add_library(pet ${PET_SOURCES})

    # Add include directories for PET
    if(LLVM_FOUND)
        target_include_directories(pet PRIVATE ${LLVM_INCLUDE_DIRS})
    endif()
    if(Clang_FOUND)
        target_include_directories(pet PRIVATE ${CLANG_INCLUDE_DIRS})
    endif()
    
    # Add required libraries for PET (TODO: fix errors when clang not found)
    target_link_libraries(pet isl ${CLANG_REQUIRED_LIBS})
    if(GMP_FOUND)
        target_link_libraries(pet ${GMP_LIBRARIES})
    endif()
    if(MPFR_FOUND)
        target_link_libraries(pet ${MPFR_LIBRARIES})
    endif()
    if(LLVM_FOUND)
        target_link_libraries(pet ${LLVM_LIBRARIES})
    endif()
    if(Clang_FOUND)
        target_link_libraries(pet ${CLANG_LIBRARIES})
    endif()

    # Set installation rules for PET
    install(TARGETS pet
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(DIRECTORY ${PET_DIR}/include/ DESTINATION include/pet)
    
    # Build PET executable
    if(BUILD_PET_EXECUTABLE)
        message(STATUS "Configuring PET executable build from source")
        # Check libyaml library (required by pet executable)
        find_package(YAML REQUIRED)
        if(YAML_FOUND)
            message(STATUS "LibYaml library found")
            message(STATUS "LibYaml include directories: ${YAML_INCLUDE_DIRS}")
            message(STATUS "LibYaml libraries: ${YAML_LIBRARIES}")
        else ()
            message(FATAL_ERROR "LibYaml library not found")
        endif()

        # Build PET executable from source
        add_executable(pet_exe
            ${PET_DIR}/main.c
            ${PET_DIR}/dummy.cc
            ${PET_DIR}/emit.c
        )
        set_target_properties(pet_exe PROPERTIES OUTPUT_NAME "pet")
        target_include_directories(pet_exe PRIVATE
            ${YAML_INCLUDE_DIRS}
        )

        # Link PET executable with required libraries
        target_link_libraries(pet_exe PRIVATE
            pet
            isl
            ${CLANG_REQUIRED_LIBS}
            ${YAML_LIBRARIES}
        )
        if(GMP_FOUND)
            target_link_libraries(pet_exe PRIVATE ${GMP_LIBRARIES})
        endif()
        if(MPFR_FOUND)
            target_link_libraries(pet_exe PRIVATE ${MPFR_LIBRARIES})
        endif()
        if(LLVM_FOUND)
            target_link_libraries(pet_exe PRIVATE ${LLVM_LIBRARIES})
        endif()
        if(Clang_FOUND)
            target_link_libraries(pet_exe PRIVATE ${CLANG_LIBRARIES})
        endif()

        # Set installation rules for PET_EXECUTABLE
        install(TARGETS pet_exe
            RUNTIME DESTINATION bin
        )
    endif()
else()
    # Use external PET library
    message(STATUS "Using external PET library from: ${PET_DIR}")
    
    # Find the PET library in the specified directory
    find_library(PET_LIBRARY pet PATHS ${PET_DIR}/lib ${PET_DIR}/lib64 NO_DEFAULT_PATH)
    
    if(NOT PET_LIBRARY)
        message(FATAL_ERROR "Could not find PET library in ${PET_DIR}/lib or ${PET_DIR}/lib64")
    endif()
    
    # Create imported target for PET
    add_library(pet UNKNOWN IMPORTED)
    set_target_properties(pet PROPERTIES IMPORTED_LOCATION ${PET_LIBRARY})
    message(STATUS "Found external PET library: ${PET_LIBRARY}")
    
    # Add include directories for external PET
    include_directories(${PET_DIR}/include)
endif()

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
file(WRITE ${CMAKE_BINARY_DIR}/include/ppcg/gitversion.h "#define GIT_HEAD_ID \"${PPCG_GIT_HEAD_VERSION}\"")
message(STATUS "PPCG version: ${PPCG_GIT_HEAD_VERSION}")  

# Include private header for PPCG
include_directories(ppcg PRIVATE ${CMAKE_BINARY_DIR}/include/ppcg)

# Include public directories for PPCG
include_directories(ppcg PUBLIC
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
    ${CMAKE_CURRENT_SOURCE_DIR}/ppcg_wrapper.c
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



# ============================================================================
# Note on third-party library installation:
# - PPCG: Always installed (required component, built from source)
# - ISL: Installed only if BUILD_ISL=ON (built from source)
# - PET: Installed only if BUILD_PET=ON (built from source)
# 
# Each library's installation rules are defined in their respective sections above.
# This separation makes it clear which components belong to the project and which
# are third-party dependencies.
# ============================================================================


# ============================================================================
# Main Application Configuration
# ============================================================================
# This section configures the main AutoStash application, which uses the PPCG, PET, and ISL libraries
# to perform code transformations and optimizations.

message(STATUS "Configuring main AutoStash application")

# Create gitversion.h file for AutoStash
execute_process(
    COMMAND git describe --always
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE _AUTOSTASH_GIT_HEAD_VERSION_
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(WRITE ${CMAKE_BINARY_DIR}/include/gitversion.h "#define GIT_HEAD_ID \"${_AUTOSTASH_GIT_HEAD_VERSION_}\"")
message(STATUS "AutoStash version: ${_AUTOSTASH_GIT_HEAD_VERSION_}")

# Include private header for AutoStash
include_directories(AutoStash PRIVATE ${CMAKE_BINARY_DIR}/include/)

# Build the main application
add_executable(AutoStash main.cpp)
set_target_properties(AutoStash PROPERTIES OUTPUT_NAME "autoStash")
target_link_libraries(AutoStash ppcg)

# Export include directories so other projects can use these libraries
set(AutoStash_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PPCG_DIR}
    ${ISL_DIR}/include
    ${PET_DIR}/include
    CACHE INTERNAL "AutoStash include directories"
)

# Install AutoStash application
install(TARGETS AutoStash
    RUNTIME DESTINATION bin
)
