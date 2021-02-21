#include <ctime>

// ActsCore
#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/MagneticField/ConstantBField.hpp>
#include <Acts/MagneticField/InterpolatedBFieldMap.hpp>
#include <Acts/MagneticField/SharedBField.hpp>
#include <Acts/Propagator/AtlasStepper.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/Navigator.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Propagator/StraightLineStepper.hpp>

// ActsExamples
#include <ActsExamples/Detector/IBaseDetector.hpp>
#include <ActsExamples/GenericDetector/GenericDetector.hpp>
#include <ActsExamples/Framework/RandomNumbers.hpp>
#include <ActsExamples/Framework/Sequencer.hpp>
#include <ActsExamples/Geometry/CommonGeometry.hpp>
#include <ActsExamples/Options/CommonOptions.hpp>
#include <ActsExamples/Plugins/BField/BFieldOptions.hpp>
#include <ActsExamples/Plugins/BField/ScalableBField.hpp>
#include <ActsExamples/Propagation/PropagationOptions.hpp>
#include <ActsExamples/Utilities/Paths.hpp>
#include <ActsExamples/TGeoDetector/TGeoDetector.hpp>

// My own propagation algorithm
#include "MyPropagationAlgorithm.hpp"

decltype(SurfaceLogger::s_storage) SurfaceLogger::s_storage;
decltype(SurfaceLogger::s_valid_geoids) SurfaceLogger::s_valid_geoids;

int main(int argc, char **argv) 
{
    GenericDetector generic_detector;
    TGeoDetector tgeo_detector;
    
    auto desc = ActsExamples::Options::makeDefaultOptions();
    ActsExamples::Options::addSequencerOptions(desc);
    ActsExamples::Options::addGeometryOptions(desc);
    ActsExamples::Options::addMaterialOptions(desc);
    ActsExamples::Options::addBFieldOptions(desc);
    ActsExamples::Options::addRandomNumbersOptions(desc);
    ActsExamples::Options::addPropagationOptions(desc);
    ActsExamples::Options::addOutputOptions(desc);
    
    desc.add_options()
//         ("gen-false-samples",
//          boost::program_options::value<bool>()->default_value(false),
//          "generate false samples by manipulating the direction on each surface")
//         ("gen-false-samples-angle-diff-min", 
//         boost::program_options::value<double>()->default_value(10.), 
//         "minimum angle the manipulated direction differs from the original one (in degree)")
//         ("gen-false-samples-angle-diff_max", 
//         boost::program_options::value<double>()->default_value(30.), 
//         "maximum angle the manipulated direction differs from the original one (in degree)")
        ("detector-type", boost::program_options::value<std::string>()->default_value("generic"), "'generic' or 'tgeo'");

    // Add specific options for detectors
    generic_detector.addOptions(desc);
    tgeo_detector.addOptions(desc);
    auto vm = ActsExamples::Options::parse(desc, argc, argv);
    if (vm.empty()) 
    {
        return EXIT_FAILURE;
    }
    
    // Which detector?
    throw_assert(vm["detector-type"].as<std::string>() == "generic" || vm["detector-type"].as<std::string>() == "tgeo", "dtector type must be 'generic' or 'tgeo'");
    
    ActsExamples::IBaseDetector *detector;
    if( vm["detector-type"].as<std::string>() == "generic" )
        detector = &generic_detector;
    else
        detector = &tgeo_detector;
    
    // Sequencer configuration    
    const auto sequencer_config = ActsExamples::Options::readSequencerConfig(vm);
    ActsExamples::Sequencer sequencer(sequencer_config);
    
    auto logLevel = ActsExamples::Options::readLogLevel(vm);

    // The geometry, material and decoration
    auto geometry = ActsExamples::Geometry::build(vm, *detector);
    auto tGeometry = geometry.first;
    auto contextDecorators = geometry.second;
    for (auto cdr : contextDecorators) {
        sequencer.addContextDecorator(cdr);
    }
    
    // init set of sensitive surfaces in SurfaceLogger     
    SurfaceLogger::s_valid_geoids.insert(tGeometry->getBeamline()->geometryId());
    tGeometry->visitSurfaces([&](const Acts::Surface* surface)
    {
        SurfaceLogger::s_valid_geoids.insert(surface->geometryId());
    });

    // Create the random number engine
    auto randomNumberSvcCfg = ActsExamples::Options::readRandomNumbersConfig(vm);
    auto randomNumberSvc =
        std::make_shared<ActsExamples::RandomNumbers>(randomNumberSvcCfg);

    // Navigator
    Acts::Navigator navigator(tGeometry);
    
    // Magnetic Field
    auto bFieldVar = ActsExamples::Options::readBField(vm);

    std::visit([&](auto& bField) 
    {
        // Resolve the bfield map and create the propgator
        using field_type = typename std::decay_t<decltype(bField)>::element_type;
        Acts::SharedBField<field_type> fieldMap(bField);
        using field_map_type = decltype(fieldMap);

        auto stepper = Acts::EigenStepper<field_map_type>{std::move(fieldMap)};
        
        using Stepper = decltype(stepper);
        using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
        Propagator propagator(std::move(stepper), std::move(navigator));

        // Read the propagation config and create the algorithms
        auto pAlgConfig = ActsExamples::Options::readPropagationConfig(vm, propagator);
        pAlgConfig.randomNumberSvc = randomNumberSvc;
        
        std::optional<std::pair<double,double>> do_dir_manip;
//         if( vm["gen-false-samples"].as<bool>() )
//         {
//             std::cout << "Enabled direction manipulation ("
//                       << vm["gen-false-samples-angle-diff-min"].as<double>() << ", "
//                       << vm["gen-false-samples-angle-diff-max"].as<double>() << ")!" << std::endl;
//             
//             do_dir_manip = {
//                 vm["gen-false-samples-angle-diff_min"].as<double>(),
//                 vm["gen-false-samples-angle-diff_max"].as<double>()
//             };
//         }
        
        sequencer.addAlgorithm(std::make_shared<MyPropagationAlgorithm<Propagator>>(pAlgConfig, logLevel, do_dir_manip));
    },
    bFieldVar);
    
    // run sequencer
    int sequencer_exit_code = 0;
    sequencer_exit_code = sequencer.run();
    
    // Generate filename
    char date_str[20];
    
    auto time = std::time(nullptr);
    auto tm = std::localtime(&time);
    std::strftime( date_str,sizeof( date_str ),"%y%m%d-%H%M%S",tm);
    
    auto event_str = "-n" + std::to_string(sequencer_config.events);
    
    std::string filename = "data-" + std::string(date_str) + event_str + ".csv";
    
    // output file
    const std::string outputDir = vm["output-dir"].template as<std::string>();
    std::fstream output_file(outputDir + filename, std::fstream::out | std::fstream::trunc);
                      
    output_file << SurfaceLogger::s_storage.data();
    
    return sequencer_exit_code;
}
