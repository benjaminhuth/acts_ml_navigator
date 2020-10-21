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

#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/MagneticField/ConstantBField.hpp>
#include <Acts/MagneticField/InterpolatedBFieldMap.hpp>
#include <Acts/MagneticField/SharedBField.hpp>
#include <Acts/Propagator/AtlasStepper.hpp>
#include <Acts/Propagator/EigenStepper.hpp>
#include <Acts/Propagator/Navigator.hpp>
#include <Acts/Propagator/Propagator.hpp>
#include <Acts/Propagator/StraightLineStepper.hpp>

// My own propagation algorithm
#include "MyPropagationAlgorithm.hpp"

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

    // Add specific options for this geometry
    detector.addOptions(desc);
    auto vm = ActsExamples::Options::parse(desc, argc, argv);
    if (vm.empty()) 
    {
        return EXIT_FAILURE;
    }
    
    ActsExamples::Sequencer sequencer(ActsExamples::Options::readSequencerConfig(vm));
    
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
    Acts::Navigator navigator(tGeometry);
    
    // Magnetic Field
    auto bFieldVar = ActsExamples::Options::readBField(vm);

    std::visit([&](auto& bField) 
    {
        // Resolve the bfield map and create the propgator
        using field_type = typename std::decay_t<decltype(bField)>::element_type;
        Acts::SharedBField<field_type> fieldMap(bField);
        using field_map_type = decltype(fieldMap);
        
        using EStepper = Acts::EigenStepper<field_map_type>;
        using AStepper = Acts::AtlasStepper<field_map_type>;
        using SStepper = Acts::StraightLineStepper;
        std::optional<std::variant<EStepper, AStepper, SStepper>> var_stepper;

        // translate option to variant
        if (vm["prop-stepper"].template as<int>() == 0)
            var_stepper = Acts::StraightLineStepper{};
        else if (vm["prop-stepper"].template as<int>() == 1)
            var_stepper = Acts::EigenStepper<field_map_type>{std::move(fieldMap)};
        else if (vm["prop-stepper"].template as<int>() == 2)
            var_stepper = Acts::AtlasStepper<field_map_type>{std::move(fieldMap)};

        // resolve stepper, setup propagator
        std::visit([&](auto& stepper) 
        {
            using Stepper = std::decay_t<decltype(stepper)>;
            using Propagator = Acts::Propagator<Stepper, Acts::Navigator>;
            Propagator propagator(std::move(stepper), std::move(navigator));

            // Read the propagation config and create the algorithms
            auto pAlgConfig = ActsExamples::Options::readPropagationConfig(vm, propagator);
            pAlgConfig.randomNumberSvc = randomNumberSvc;
            sequencer.addAlgorithm(std::make_shared<MyPropagationAlgorithm<Propagator>>(pAlgConfig, logLevel));
        },
        *var_stepper);
    },
    bFieldVar);
    
    sequencer.run();
}
