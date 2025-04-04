# FindGMP.cmake
# Find GMP library
# Defines:
#  GMP_FOUND - GMP library was found
#  GMP_INCLUDE_DIRS - GMP header file directories
#  GMP_LIBRARIES - GMP libraries

# Include useful find functions
include(FindPackageHandleStandardArgs)

# Find library
find_library(GMP_LIBRARIES
  NAMES gmp libgmp
  PATHS
    ${GMP_ROOT}
    $ENV{GMP_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES lib lib64
)

# Find include directory
find_path(GMP_INCLUDE_DIRS
  NAMES gmp.h
  PATHS
    ${GMP_ROOT}
    $ENV{GMP_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES include
)

# Handle the REQUIRED argument and set the *_FOUND variable
find_package_handle_standard_args(GMP
  REQUIRED_VARS GMP_LIBRARIES GMP_INCLUDE_DIRS
)

# Mark as advanced
mark_as_advanced(GMP_INCLUDE_DIRS GMP_LIBRARIES)