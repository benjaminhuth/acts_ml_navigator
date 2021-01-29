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

#include "MLNavigator.hpp"

int main(int argc, char **argv) 
{
    GenericDetector detector;
    
    auto desc = ActsExamples::Options::makeDefaultOptions();
    ActsExamples::Options::addSequencerOptions(desc);
    ActsExamples::Options::addGeometryOptions(desc);
    ActsExamples::Options::addMaterialOptions(desc);
    ActsExamples::Options::addBFieldOptions(desc);
    ActsExamples::Options::addRandomNumbersOptions(desc);
    ActsExamples::Options::addPropagationOptions(desc);
    ActsExamples::Options::addOutputOptions(desc);
    
    
    desc.add_options()("nav_model", 
                       boost::program_options::value<std::string>(), 
                       "path of a ONNX Model for navigation")
                      ("graph_data", 
                       boost::program_options::value<std::string>(), 
                       "path to the propgagation log from which the graph is built")
                      ("bpsplit_z", 
                       boost::program_options::value<std::string>(),
                       "path to the beampipe split file");

    // Add specific options for this geometry
    detector.addOptions(desc);
    auto vm = ActsExamples::Options::parse(desc, argc, argv);
    if (vm.empty()) 
    {
        return EXIT_FAILURE;
    }
    
    const auto sequencer_config = ActsExamples::Options::readSequencerConfig(vm);
    ActsExamples::Sequencer sequencer(sequencer_config);
    
    auto logLevel = ActsExamples::Options::readLogLevel(vm);

    // The geometry, material and decoration
    auto geometry = ActsExamples::Geometry::build(vm, detector);
    auto tGeometry = geometry.first;
    auto contextDecorators = geometry.second;
    for (auto cdr : contextDecorators) {
        sequencer.addContextDecorator(cdr);
    }

    // Create the random number engine
    auto randomNumberSvcCfg = ActsExamples::Options::readRandomNumbersConfig(vm);
    auto randomNumberSvc =
        std::make_shared<ActsExamples::RandomNumbers>(randomNumberSvcCfg);

    // Navigator
//     Acts::Navigator navigator(tGeometry);
    
    AllINeedForMLNavigation n{
        std::make_shared<OnnxModel<3,1>>(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "navigation_model"), vm["nav_model"].as<std::string>()),
        tGeometry,
        parseGraphFromCSV(vm["graph_data"].as<std::string>(), *tGeometry),
        loadBPSplitZBounds(vm["bpsplit_z"].as<std::string>())
    };
    
    MLNavigator navigator(n);
    
    // Magnetic Field
    auto bFieldVar = ActsExamples::Options::readBField(vm);

    std::visit([&](auto& bField) 
    {
        // Resolve the bfield map and create the propgator
        using field_type = typename std::decay_t<decltype(bField)>::element_type;
        Acts::SharedBField<field_type> fieldMap(bField);
        using field_map_type = decltype(fieldMap);
        
        using Stepper = Acts::EigenStepper<field_map_type>;

        Stepper stepper{std::move(fieldMap)};

        using Propagator = Acts::Propagator<Stepper, MLNavigator>;
        Propagator propagator(std::move(stepper), std::move(navigator));

        // Read the propagation config and create the algorithms
        auto pAlgConfig = ActsExamples::Options::readPropagationConfig(vm, propagator);
        pAlgConfig.randomNumberSvc = randomNumberSvc;
        sequencer.addAlgorithm(std::make_shared<ActsExamples::PropagationAlgorithm<Propagator>>(pAlgConfig, logLevel));
    },
    bFieldVar);
    
    // run sequencer
    return sequencer.run();
}
