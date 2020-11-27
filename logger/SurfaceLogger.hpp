#ifndef SURFACELOGGER_HPP_INCLUDED
#define SURFACELOGGER_HPP_INCLUDED

#include <vector>
#include <set>
#include <optional>
#include <iostream>
#include <mutex>

#include <Acts/Utilities/Definitions.hpp>
#include <Acts/Geometry/GeometryIdentifier.hpp>

struct SurfaceLogger
{
    /// Struct that holds all infos that are captured by the logger
    struct edge_info_t
    {
        Acts::Vector3D start_pos;
        Acts::Vector3D start_dir;
        double start_qop;
        std::pair<Acts::GeometryIdentifier,Acts::Vector3D> start_surface;
        std::optional<std::pair<Acts::GeometryIdentifier,Acts::Vector3D>> end_surface;
    };
    
    /// The result type. On destruction it adds the collected data to a static
    /// class
    struct result_type 
    {
        std::vector<edge_info_t> edges;
        
        ~result_type()
        {
            if( !edges.back().end_surface.has_value() )
                edges.pop_back();
            
            storage.thread_safe_push_back( std::move(edges) );
        }
    };
    
    /// Static class that holds all collected results. Allows thread save push_back.
    static class storage_t
    {
        std::vector<std::vector<edge_info_t>> stored_data;
        std::mutex push_back_mutex;
        
    public:
        storage_t() = default;
        storage_t(const storage_t &) = delete;
        
        void thread_safe_push_back(std::vector<edge_info_t> &&v)
        {
            std::lock_guard<std::mutex> push_back_guard(push_back_mutex);
            stored_data.push_back(std::move(v));
        }
        
        const auto &data() const
        {
            return stored_data;
        }
    } storage;
    
    /// All valid geoids (sensitive surfaces and beampipe at the moment)
    static std::set<Acts::GeometryIdentifier> valid_geoids;
    
    /// Call to the logger
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper, result_type& result) const 
    {
        if( state.navigation.currentSurface == nullptr )
            return;
        
        const auto &surface = state.navigation.currentSurface;
        
        if( result.edges.empty() || result.edges.back().end_surface.has_value() )
        {
            // only add sensitive surfaces or beam pipe
            if( valid_geoids.find(surface->geometryId()) != valid_geoids.end() )
                result.edges.push_back(edge_info_t{
                    stepper.position(state.stepping),
                    stepper.direction(state.stepping),
                    stepper.charge(state.stepping)/stepper.momentum(state.stepping),
                    {surface->geometryId(),surface->center(state.geoContext.get())},
                    std::nullopt
                });
            
            return;
        }

        auto &last_edge = result.edges.back();        
        
        // If the new surface is different from the old one, set end of edge
        if( !last_edge.end_surface.has_value() && surface->geometryId() != last_edge.start_surface.first )
            if( valid_geoids.find(surface->geometryId()) != valid_geoids.end() )
            {
                // end of last edge
                last_edge.end_surface = {surface->geometryId(),surface->center(state.geoContext.get())};
                
                // start of new edge
                result.edges.push_back(edge_info_t{
                    stepper.position(state.stepping),
                    stepper.direction(state.stepping),
                    stepper.charge(state.stepping)/stepper.momentum(state.stepping),
                    {surface->geometryId(),surface->center(state.geoContext.get())},
                    std::nullopt
                });
            }
    }

    // Not used
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t&, const stepper_t&) const 
    {
      
    }
};

std::ostream &operator<<(std::ostream &os, const std::vector<std::vector<SurfaceLogger::edge_info_t>> &data)
{
    os << "start_id,start_x,start_y,start_z,end_id,end_x,end_y,end_z,pos_x,pos_y,pos_z,dir_x,dir_y,dir_z,qop\n";
    
    for( const auto &track : data )
    {
        for( const auto &e : track )
        {
            os << e.start_surface.first.value() << ","
               << e.start_surface.second(0) << ","
               << e.start_surface.second(1) << ","
               << e.start_surface.second(2) << ","
               << e.end_surface->first.value() << ","
               << e.end_surface->second(0) << ","
               << e.end_surface->second(1) << ","
               << e.end_surface->second(2) << ","
               << e.start_pos(0) << ","
               << e.start_pos(1) << ","
               << e.start_pos(2) << ","
               << e.start_dir(0) << ","
               << e.start_dir(1) << ","
               << e.start_dir(2) << ","
               << e.start_qop << std::endl;
        }
    }
    
    return os;
}

#endif // SURFACELOGGER_HPP_INCLUDED
