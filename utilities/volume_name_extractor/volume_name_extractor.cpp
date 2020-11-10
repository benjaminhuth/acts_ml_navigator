#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/Geometry/Layer.hpp>
#include <Acts/Geometry/TrackingVolume.hpp>
#include <Acts/Surfaces/Surface.hpp>

#include <ActsExamples/GenericDetector/GenericDetector.hpp>
#include <ActsExamples/Geometry/CommonGeometry.hpp>
#include <ActsExamples/Options/CommonOptions.hpp>

using geoid_name_table_t = std::vector<std::pair<uint64_t,std::string>>;

void to_ostream(std::ostream &os, const geoid_name_table_t &table);

int main(int argc, char ** argv)
{    
    GenericDetector detector;
    
    auto desc = ActsExamples::Options::makeDefaultOptions();
    ActsExamples::Options::addGeometryOptions(desc);
    ActsExamples::Options::addMaterialOptions(desc);
    detector.addOptions(desc);
    
    auto vm = ActsExamples::Options::parse(desc, argc, argv);
    if (vm.empty()) 
    {
        return EXIT_FAILURE;
    }
    
    auto geometry = ActsExamples::Geometry::build(vm, detector);
    const auto tGeometry = geometry.first;
    
    geoid_name_table_t table;
    table.reserve(20'000); 
    
    if( const auto beamline = tGeometry->getBeamline(); beamline )
        table.push_back(std::make_pair(beamline->geometryId().value(),"Beamline"));
    
    tGeometry->visitSurfaces([&](const Acts::Surface* surface)
    {    
        if( const auto layer = surface->associatedLayer(); layer )
            if( const auto tvolume = layer->trackingVolume(); tvolume )
                table.push_back(std::make_pair(surface->geometryId().value(), tvolume->volumeName()));
    });
    
    // Output
    std::fstream output_file("detector_surfaces.csv", std::fstream::out | std::fstream::trunc);
    
    to_ostream(output_file, table);
}

void to_ostream(std::ostream& os, const geoid_name_table_t &table)
{
    os << "ordinal_id,geo_id,volume\n";
    
    for(std::size_t i=0; i<table.size(); ++i)
        os << i << "," << table[i].first << "," << table[i].second << "\n";
}
