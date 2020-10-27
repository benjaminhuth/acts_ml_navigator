cmake_minimum_required(VERSION 3.0)

# Project settings
project(logging_actor CXX)
set(CMAKE_CXX_STANDARD 17)
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
if(CMAKE_BUILD_TYPE STREQUAL "Debug" AND (${GCC} OR ${CLANG}) )
    add_compile_options("-fsanitize=address,leak,undefined")
    add_link_options("-fsanitize=address,leak,undefined")
endif()    

# Executables
add_executable(logging_actor main.cpp)
add_executable(geo_id_decoder geo_id_decoder.cpp)

# Boost
find_package(Boost 1.69 MODULE REQUIRED COMPONENTS program_options)
target_link_libraries(logging_actor Boost::program_options)

# Setup Acts
set(ACTS_ROOT /home/benjamin/Dokumente/acts_project/acts)
set(ACTS_BINARY_DIR ${ACTS_ROOT}/build)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
target_link_libraries(logging_actor Eigen3::Eigen)

include_directories(SYSTEM
    ${ACTS_ROOT}/Core/include
    ${ACTS_ROOT}/Examples/Framework/include
    ${ACTS_ROOT}/Examples/Detectors/Common/include
    ${ACTS_ROOT}/Examples/Detectors/GenericDetector/include
    ${ACTS_ROOT}/Examples/Detectors/MagneticField/include
    ${ACTS_ROOT}/Examples/Run/Common/include
    ${ACTS_ROOT}/Examples/Algorithms/Propagation/include
)

add_library(ActsCore SHARED IMPORTED)
set_target_properties(ActsCore PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsCore.so)

add_library(ActsExamplesCommon SHARED IMPORTED)
set_target_properties(ActsExamplesCommon PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesCommon.so)

add_library(ActsExamplesFramework SHARED IMPORTED)
set_target_properties(ActsExamplesFramework PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesFramework.so)

add_library(ActsExamplesDetectorGeneric SHARED IMPORTED)
set_target_properties(ActsExamplesDetectorGeneric PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesDetectorGeneric.so)

add_library(ActsExamplesMagneticField SHARED IMPORTED)
set_target_properties(ActsExamplesMagneticField PROPERTIES IMPORTED_LOCATION ${ACTS_BINARY_DIR}/lib64/libActsExamplesMagneticField.so)

# Link
target_link_libraries(logging_actor ActsCore ActsExamplesCommon ActsExamplesFramework ActsExamplesDetectorGeneric ActsExamplesMagneticField)
target_link_libraries(geo_id_decoder ActsCore)
