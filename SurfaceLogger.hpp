#ifndef SURFACELOGGER_HPP_INCLUDED
#define SURFACELOGGER_HPP_INCLUDED

#include <vector>
#include <optional>
#include <iostream>

#include <Acts/Utilities/Definitions.hpp>
#include <Acts/Geometry/GeometryIdentifier.hpp>

struct SurfaceLogger
{
    struct edge_info_t
    {
        Acts::Vector3D start_pos;
        Acts::Vector3D start_dir;
        Acts::GeometryIdentifier start_id;
        std::optional<Acts::GeometryIdentifier> end_id;
    };
    
    struct result_type 
    {
        std::vector<edge_info_t> edges;
        
        ~result_type()
        {
            std::cout << "start_id,end_id\n";
            for(const auto &edge : edges)
                std::cout << edge.start_id.value() << "," << edge.end_id->value() << "\n";
            std::cout << std::endl;
        }
    };
    
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper, result_type& result) const 
    {
        if( state.navigation.currentSurface == nullptr )
            return;
        
        
        if( result.edges.empty() || result.edges.back().end_id.has_value() )
        {
            result.edges.push_back(edge_info_t{
                stepper.position(state.stepping),
                stepper.direction(state.stepping),
                state.navigation.currentSurface->geometryId(),
                std::nullopt
            });
            
            return;
        }

        auto &last_edge = result.edges.back();
        
        if( state.navigation.navigationBreak )
        {
            // If navigation terminates and there is a unfinished edge
            if( !last_edge.end_id.has_value() )
                result.edges.pop_back();
            
            return;
        }
        
        
        // If the new surface 
        if( !last_edge.end_id.has_value() && state.navigation.currentSurface->geometryId() != last_edge.start_id )
            last_edge.end_id = state.navigation.currentSurface->geometryId();
    }

    // Not used
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t&, const stepper_t&) const 
    {
      
    }
};

#endif // SURFACELOGGER_HPP_INCLUDED
