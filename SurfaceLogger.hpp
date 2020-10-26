#ifndef SURFACELOGGER_HPP_INCLUDED
#define SURFACELOGGER_HPP_INCLUDED

#include <vector>
#include <optional>
#include <iostream>
#include <mutex>

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
    
    struct result_type 
    {
        std::vector<edge_info_t> edges;
        
        ~result_type()
        {
            if( static_cast<int>(edges.back().end_id->value()) == 0 )
                edges.pop_back();
            
            storage.thread_safe_push_back( std::move(edges) );
        }
    };
    
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper, result_type& result) const 
    {
        if( state.navigation.currentSurface == nullptr || state.navigation.currentSurface->geometryId().value() == 0 )
            return;
        
        auto &current_surface_id = state.navigation.currentSurface->geometryId();
        auto targetSurface = state.navigation.targetSurface;
        using OptId = std::optional<Acts::GeometryIdentifier>;
        
        if( result.edges.empty() )
        {
            result.edges.push_back(edge_info_t{
                stepper.position(state.stepping),
                stepper.direction(state.stepping),
                current_surface_id,
                targetSurface ? OptId(targetSurface->geometryId()) : OptId(std::nullopt)
            });
        }
        else
        {
            auto &last_edge = result.edges.back();
            
            last_edge.end_id = targetSurface ? OptId(targetSurface->geometryId()) : OptId(std::nullopt);
                
            if( state.navigation.targetReached == true )
            {               
                result.edges.push_back(edge_info_t{
                    stepper.position(state.stepping),
                    stepper.direction(state.stepping),
                    current_surface_id,
                    std::nullopt
                });
                
                return;
            }
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
    os << "start_id,end_id,dir_x,dir_y,dir_z,pos_x,pos_y,pos_z\n";
    
    for( const auto &track : SurfaceLogger::storage.data() )
    {
        for( const auto &e : track )
        {
            os << e.start_id.value() << ","
               << e.end_id->value() << ","
               << e.start_dir(0) << ","
               << e.start_dir(1) << ","
               << e.start_dir(2) << ","
               << e.start_pos(0) << ","
               << e.start_pos(1) << ","
               << e.start_pos(2) << std::endl;
        }
    }
    
    return os;
}

#endif // SURFACELOGGER_HPP_INCLUDED
