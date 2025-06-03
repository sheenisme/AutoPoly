# ============================================================================
# ISL Library Configuration
# ============================================================================
# ISL (Integer Set Library) is a library for manipulating sets and relations of integer points
# bounded by linear constraints. It's a core dependency for PPCG and PET.

message(STATUS "ISL source dir: ${ISL_DIR}")

# Create gitversion.h file (required by ISL)
execute_process(
    COMMAND git describe --always 
    WORKING_DIRECTORY ${ISL_DIR}
    OUTPUT_VARIABLE ISL_GIT_HEAD_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
file(WRITE ${ISL_DIR}/gitversion.h "#define GIT_HEAD_ID \"${ISL_GIT_HEAD_VERSION}\"")
message(STATUS "ISL version: ${ISL_GIT_HEAD_VERSION}")

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
file(WRITE "${AUTOSTASH_BINARY_DIR}/include/isl/stdint.h.tmp"
    "${INCLUDE_STDINT_H}\n")
configure_file("${AUTOSTASH_BINARY_DIR}/include/isl/stdint.h.tmp"
    "${AUTOSTASH_BINARY_DIR}/include/isl/isl/stdint.h" COPYONLY)

# Create ISL configuration header file
configure_file("${AUTOSTASH_SOURCE_DIR}/cmake/isl_config.h.cmake" "${AUTOSTASH_BINARY_DIR}/include/isl/isl_config.h")
include_directories(isl PRIVATE ${AUTOSTASH_BINARY_DIR}/include/isl)

# Include ISL headers directories
include_directories(isl PRIVATE
    ${ISL_DIR}
    ${ISL_DIR}/imath
    ${ISL_DIR}/include
)

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

# Set compile options to ISL
target_compile_options(isl PRIVATE -g
    -Wno-sign-compare
    -Wno-cast-qual
    -Wno-discarded-qualifiers
    -Wno-implicit-fallthrough
    -Wno-unused-function
    -Wno-unused-variable
    -Wno-unused-but-set-variable
    -Wno-type-limits
    -Wno-return-type
)

# Set installation rules for ISL
install(TARGETS isl
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)
install(DIRECTORY ${ISL_DIR}/include/ DESTINATION include/)
