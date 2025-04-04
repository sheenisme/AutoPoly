# FindMPFR.cmake
# Find MPFR library
# Defines:
#  MPFR_FOUND - Found MPFR library
#  MPFR_INCLUDE_DIRS - MPFR header file directories
#  MPFR_LIBRARIES - MPFR libraries

# Include useful find functions
include(FindPackageHandleStandardArgs)

# Find the library
find_library(MPFR_LIBRARIES
  NAMES mpfr libmpfr
  PATHS
    ${MPFR_ROOT}
    $ENV{MPFR_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES lib lib64
)

# Find the include directory
find_path(MPFR_INCLUDE_DIRS
  NAMES mpfr.h
  PATHS
    ${MPFR_ROOT}
    $ENV{MPFR_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES include
)

# Handle the REQUIRED argument and set the *_FOUND variable
find_package_handle_standard_args(MPFR
  REQUIRED_VARS MPFR_LIBRARIES MPFR_INCLUDE_DIRS
)

# Mark as advanced
mark_as_advanced(MPFR_INCLUDE_DIRS MPFR_LIBRARIES) 