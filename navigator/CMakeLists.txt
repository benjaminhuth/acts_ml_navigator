add_executable(navigation_test main.cpp)

target_link_libraries(navigation_test Boost::program_options)
target_link_libraries(navigation_test Eigen3::Eigen)
target_link_libraries(navigation_test 
    ActsCore
    ActsPluginOnnx
    ActsExamplesCommon
    ActsExamplesFramework
    ActsExamplesDetectorGeneric
    ActsExamplesMagneticField
    ActsExamplesIoCsv
)
