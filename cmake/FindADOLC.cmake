# This module relies on ADOLC_DIR being set.
#
# ADOLC_FOUND - system has ADOLC.
# ADOLC_INCLUDE_DIRS - ADOLC include directories.
# ADOLC_LIBRARIES - ADOLC libraries.

# Find the headers.
find_path(ADOLC_INCLUDE_DIR adolc/adolc.h
          HINTS ${ADOLC_DIR}/include)

# Find the libraries.
find_library(ADOLC_LIBRARY
             NAMES adolc
             HINTS ${ADOLC_DIR}/lib
                   ${ADOLC_DIR}/lib64
                   ${ADOLC_DIR}/x86_64-linux-gnu
                   ${ADOLC_DIR}/i386-linux-gnu)

# Set variables.
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ADOLC
                                  REQUIRED_VARS ADOLC_LIBRARY ADOLC_INCLUDE_DIR)

if (ADOLC_FOUND)
    set (MAST_ENABLE_ADOLC 1)
else()
    set (MAST_ENABLE_ADOLC 0)
endif()

mark_as_advanced(ADOLC_INCLUDE_DIR ADOLC_LIBRARY ADOLC_FOUND MAST_ENABLE_ADOLC)

set(ADOLC_LIBRARIES ${ADOLC_LIBRARY})
set(ADOLC_INCLUDE_DIRS ${ADOLC_INCLUDE_DIR})
