#ifndef SURFACELOGGER_HPP_INCLUDED
#define SURFACELOGGER_HPP_INCLUDED

#include <vector>
#include <set>
#include <optional>
#include <iostream>
#include <mutex>
#include <cmath>

#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Geometry/GeometryIdentifier.hpp>

auto angle_between(const Acts::Vector3 &a, const Acts::Vector3 &b)
{
    return std::acos( a.dot(b) / ( a.norm() * b.norm() ) );
}


struct SurfaceLogger
{
    /// All valid geoids (sensitive surfaces and beampipe at the moment)
    static std::set<Acts::GeometryIdentifier> s_valid_geoids;
    
    bool m_do_direction_manipulation = false;
    double m_angle_diff_min = 0.0;
    double m_angle_diff_max = 0.0;
    double m_dim_shift = 0.3;
    
    /// Struct that holds all infos that are captured by the logger
    struct edge_info_t
    {
        Acts::Vector3 start_pos;
        Acts::Vector3 start_dir;
        double start_qop;
        std::pair<Acts::GeometryIdentifier,Acts::Vector3> start_surface;
        std::optional<std::pair<Acts::GeometryIdentifier,Acts::Vector3>> end_surface;
    };
    
    /// The result type
    struct result_type 
    {
        bool is_already_exported = false;
        std::vector<edge_info_t> edges;
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
    } s_storage;
    
    
    auto manipulate_direction(const Acts::Vector3 &dir) const
    {
        Acts::Vector3 new_dir = dir;
        double angle_diff = 0;
        
        do
        {
            Acts::Vector3 shift_vec = Acts::Vector3::Random() * m_dim_shift;
            new_dir = dir + shift_vec;
            angle_diff = angle_between(dir, new_dir) * 360. / (2*M_PI);
        } 
        while( angle_diff < m_angle_diff_min || angle_diff > m_angle_diff_max );
        
        return new_dir.normalized();
    }
    
    /// Call to the logger
    template <typename propagator_state_t, typename stepper_t>
    void operator()(propagator_state_t& state, const stepper_t& stepper, result_type& result) const 
    {
        auto push_back_edge = [&state, &stepper, &result](const auto &surface)
        {
            result.edges.push_back(edge_info_t{
                stepper.position(state.stepping),
                stepper.direction(state.stepping),
                stepper.charge(state.stepping)/stepper.momentum(state.stepping),
                {surface->geometryId(),surface->center(state.geoContext.get())},
                std::nullopt
            });
            
        };
        
        // Export if necessary
        if( state.navigation.navigationBreak && !result.is_already_exported )
        {
            if( !result.edges.back().end_surface.has_value() )
                result.edges.pop_back();
            
            s_storage.thread_safe_push_back( std::move(result.edges) );
            result.is_already_exported = true;
        }
        
        if( state.navigation.currentSurface == nullptr )
            return;
        
        // Set the start of the edge at the beginning (or when else necessary)
        const auto &surface = state.navigation.currentSurface;
        
        if( result.edges.empty() || result.edges.back().end_surface.has_value() )
        {
            if( s_valid_geoids.find(surface->geometryId()) != s_valid_geoids.end() )
                push_back_edge(surface);
            
            if( m_do_direction_manipulation )
                state.stepping.pars.template segment<3>(Acts::eFreeDir0) = manipulate_direction(stepper.direction(state.stepping));
            
            return;
        }

        auto &last_edge = result.edges.back();        
        
        // If the new surface is different from the old one, set end of edge
        if( !last_edge.end_surface.has_value() && surface->geometryId() != last_edge.start_surface.first )
            if( s_valid_geoids.find(surface->geometryId()) != s_valid_geoids.end() )
            {
                // end of last edge
                last_edge.end_surface = {surface->geometryId(),surface->center(state.geoContext.get())};
                
                // start of new edge
                push_back_edge(surface);
                
                if( m_do_direction_manipulation )
                    state.stepping.pars.template segment<3>(Acts::eFreeDir0) = manipulate_direction(stepper.direction(state.stepping));
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
