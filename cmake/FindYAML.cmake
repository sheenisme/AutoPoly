# FindYAML.cmake
# Find YAML library (libyaml)
# Defines:
#  YAML_FOUND - YAML library was found
#  YAML_INCLUDE_DIRS - YAML header file directories
#  YAML_LIBRARIES - YAML libraries

# Include useful find functions
include(FindPackageHandleStandardArgs)

# Find library
find_library(YAML_LIBRARIES
  NAMES yaml libyaml
  PATHS
    ${YAML_ROOT}
    $ENV{YAML_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES lib lib64
)

# Find include directory
find_path(YAML_INCLUDE_DIRS
  NAMES yaml.h
  PATHS
    ${YAML_ROOT}
    $ENV{YAML_ROOT}
    $ENV{CONDA_PREFIX}
    /usr
    /usr/local
    /opt/local
  PATH_SUFFIXES include
)

# Handle the REQUIRED argument and set the *_FOUND variable
find_package_handle_standard_args(YAML
  REQUIRED_VARS YAML_LIBRARIES YAML_INCLUDE_DIRS
)

# Mark as advanced
mark_as_advanced(YAML_INCLUDE_DIRS YAML_LIBRARIES)