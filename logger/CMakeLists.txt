add_executable(logger logger.cpp)

target_link_libraries(logger Boost::program_options)
target_link_libraries(logger Eigen3::Eigen)
target_link_libraries(logger ActsCore ActsExamplesCommon ActsExamplesFramework ActsExamplesDetectorGeneric ActsExamplesMagneticField)

configure_file(run_logger.sh run_logger.sh COPYONLY)
