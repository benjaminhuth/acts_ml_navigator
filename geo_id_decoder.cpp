#include <iostream>

#include "Acts/Geometry/GeometryIdentifier.hpp"

int main(int argc, char **argv)
{
    if( argc != 2 )
    {
        std::cout << "Usage: " << argv[0] << " geo_id_to_encode" << std::endl;
        return 0;
    }
    
    Acts::GeometryIdentifier id(std::atoll(argv[1]));
    
    std::cout << argv[1] << ": " << id << std::endl;
}
