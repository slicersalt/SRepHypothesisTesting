cmake_minimum_required(VERSION 3.13.4)

project(SRepHypothesisTesting)

#-----------------------------------------------------------------------------
# Extension meta-information
set(EXTENSION_HOMEPAGE "")
set(EXTENSION_CATEGORY "Skeleton, topology")
set(EXTENSION_CONTRIBUTORS "Jared Vicory, Josh Cates (Kitware)")
set(EXTENSION_DESCRIPTION "Hypothesis testing for populations of s-reps")
set(EXTENSION_ICONURL "")
set(EXTENSION_SCREENSHOTURLS "")
set(EXTENSION_DEPENDS "SRep") # Specified as a list or "NA" if no dependencies

#-----------------------------------------------------------------------------
# Extension dependencies
find_package(Slicer REQUIRED)
include(${Slicer_USE_FILE})

#-----------------------------------------------------------------------------
# Extension modules
add_subdirectory(SRepHypothesisTesting)
## NEXT_MODULE

#-----------------------------------------------------------------------------
include(${Slicer_EXTENSION_GENERATE_CONFIG})
include(${Slicer_EXTENSION_CPACK})
