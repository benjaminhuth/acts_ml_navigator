cmake_minimum_required(VERSION 3.0)
cmake_policy(SET CMP0057 NEW)

# Module path
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

# Project settings
project(logging_actor CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Compiler checks
set(CLANG FALSE)
set(GCC FALSE)
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CLANG TRUE)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(GCC TRUE)
endif()

# Warnings
include(cmake/compiler_warnings.cmake)
if(MSVC)
    add_compile_options(${MSVC_WARNINGS})
elseif(${CLANG})
    add_compile_options(${CLANG_WARNINGS})
elseif(${GCC})
    add_compile_options(${GCC_WARNINGS})
endif()

# Sanitizers enabled on Debug-mode
#if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND (${GCC} OR ${CLANG}) )
    #add_compile_options("-fsanitize=address,leak,undefined")
    #add_link_options("-fsanitize=address,leak,undefined")
#endif()

# Boost
find_package(Boost 1.69 MODULE REQUIRED COMPONENTS program_options)

# Acts
find_package(Acts REQUIRED COMPONENTS Core PluginOnnx)

# Acts Examples
set(ACTS_ROOT /home/benjamin/Dokumente/acts_project/acts)
set(ACTS_BINARY_DIR ${ACTS_ROOT}/build)

include_directories(SYSTEM
    ${ACTS_ROOT}/Examples/Framework/include
    ${ACTS_ROOT}/Examples/Detectors/Common/include
    ${ACTS_ROOT}/Examples/Detectors/GenericDetector/include
    ${ACTS_ROOT}/Examples/Detectors/TGeoDetector/include
    ${ACTS_ROOT}/Examples/Detectors/MagneticField/include
    ${ACTS_ROOT}/Examples/Run/Common/include
    ${ACTS_ROOT}/Examples/Algorithms/Propagation/include
    ${ACTS_ROOT}/Examples/Io/Csv/include
)

add_library(ActsExamplesCommon SHARED IMPORTED)
add_dependencies(ActsExamplesCommon ActsCore)
set_target_properties(ActsExamplesCommon PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesCommon.so)

add_library(ActsExamplesFramework SHARED IMPORTED)
add_dependencies(ActsExamplesFramework ActsCore)
set_target_properties(ActsExamplesFramework PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesFramework.so)

add_library(ActsExamplesDetectorGeneric SHARED IMPORTED)
add_dependencies(ActsExamplesDetectorGeneric ActsCore)
set_target_properties(ActsExamplesDetectorGeneric PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesDetectorGeneric.so)

add_library(ActsExamplesDetectorTGeo SHARED IMPORTED)
add_dependencies(ActsExamplesDetectorTGeo ActsCore)
set_target_properties(ActsExamplesDetectorTGeo PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesDetectorTGeo.so)

add_library(ActsExamplesMagneticField SHARED IMPORTED)
add_dependencies(ActsExamplesMagneticField ActsCore)
set_target_properties(ActsExamplesMagneticField PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesMagneticField.so)

add_library(ActsExamplesIoCsv SHARED IMPORTED)
add_dependencies(ActsExamplesIoCsv ActsCore)
set_target_properties(ActsExamplesIoCsv PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesIoCsv.so)

# subdirectories
add_subdirectory(logger)
add_subdirectory(utilities/geo_id_decoder)
add_subdirectory(utilities/volume_name_extractor)
add_subdirectory(utilities/kd_tree)
add_subdirectory(utilities/interactive_track_viewer)
add_subdirectory(navigator)
