#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/Geometry/Layer.hpp>
#include <Acts/Geometry/TrackingVolume.hpp>
#include <Acts/Surfaces/Surface.hpp>

#include <ActsExamples/GenericDetector/GenericDetector.hpp>
#include <ActsExamples/Geometry/CommonGeometry.hpp>
#include <ActsExamples/Options/CommonOptions.hpp>
#include <ActsExamples/TGeoDetector/TGeoDetector.hpp>

using geoid_name_table_t = std::vector<std::tuple<uint64_t, std::string, Acts::Vector3>>;

void to_ostream(std::ostream &os, const geoid_name_table_t &table);

int main(int argc, char ** argv)
{    
    GenericDetector generic_detector;
    TGeoDetector tgeo_detector;
    
    Acts::GeometryContext gctx;
    
    auto desc = ActsExamples::Options::makeDefaultOptions();
    ActsExamples::Options::addGeometryOptions(desc);
    ActsExamples::Options::addMaterialOptions(desc);
    generic_detector.addOptions(desc);
    tgeo_detector.addOptions(desc);
        
    desc.add_options()
        ("detector-type", boost::program_options::value<std::string>()->default_value("generic"), "'generic' or 'tgeo'");
    
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
    
    auto geometry = ActsExamples::Geometry::build(vm, *detector);
    const auto tGeometry = geometry.first;
    
    geoid_name_table_t table;
    table.reserve(20'000); 
    
    if( const auto beamline = tGeometry->getBeamline(); beamline )
        table.push_back(std::make_tuple(beamline->geometryId().value(),"Beamline",Acts::Vector3::Zero()));
    
    tGeometry->visitSurfaces([&](const Acts::Surface* surface)
    {    
        if( const auto layer = surface->associatedLayer(); layer )
            if( const auto tvolume = layer->trackingVolume(); tvolume )
                table.push_back(std::make_tuple(surface->geometryId().value(), tvolume->volumeName(), surface->center(gctx)));
    });
    
    // Output
    std::fstream output_file("detector_surfaces.csv", std::fstream::out | std::fstream::trunc);
    
    to_ostream(output_file, table);
}

void to_ostream(std::ostream& os, const geoid_name_table_t &table)
{
    os << "ordinal_id,geo_id,volume,x,y,z\n";
    
    for(std::size_t i=0; i<table.size(); ++i)
        os << i << "," << std::get<0>(table[i]) << "," 
                       << std::get<1>(table[i]) << "," 
                       << std::get<2>(table[i])[0] << ","
                       << std::get<2>(table[i])[1] << ","
                       << std::get<2>(table[i])[2] << "\n";
}
