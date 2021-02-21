#pragma once
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <charconv>

#include <Acts/Surfaces/Surface.hpp>
#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Geometry/GeometryIdentifier.hpp>
#include <Acts/Utilities/Logger.hpp>
#include <Acts/Geometry/TrackingGeometry.hpp>
#include <Acts/Geometry/TrackingVolume.hpp>

#include "OnnxModel.hpp"
#include "csv.hpp"
    
using namespace std::string_literals;


struct NoSurfaceLeftException : std::runtime_error
{
    NoSurfaceLeftException(std::string msg) : std::runtime_error(msg) {}
};




auto parseGraphFromCSV(const std::filesystem::path &csv_path,
                       const Acts::TrackingGeometry &tgeo)
{
    auto file = std::ifstream(csv_path);
    
    std::vector<std::pair<Acts::GeometryIdentifier, Acts::GeometryIdentifier>> graph_data;
    
    bool header = true;
    for( const auto &row : CSVRange(file) )
    {
        if( header )
        {
            header = false;
            continue;
        }
        
        uint64_t start_id, target_id;
        
        auto ec0 = std::from_chars(row[0].begin(), row[0].end(), start_id);
        auto ec1 = std::from_chars(row[4].begin(), row[4].end(), target_id);
        
        if( ec0.ec != std::errc() || ec1.ec != std::errc() )
            throw std::runtime_error("Conversion failed");
        
        graph_data.push_back({start_id, target_id});
    }
    
    std::map<Acts::GeometryIdentifier, std::set<Acts::GeometryIdentifier>> id_map;
    
    for(const auto &connection : graph_data)
        id_map[connection.first].insert(connection.second);
    
    std::map<const Acts::Surface *, std::set<const Acts::Surface *>> pointer_map;
    
    for(const auto &[id, target_ids] : id_map)
    {
        auto start_ptr = tgeo.findSurface(id);
        
        if( id == 0 )
            start_ptr = tgeo.getBeamline();
        
        std::set<const Acts::Surface *> target_ptrs;
        
        for(const auto target_id : target_ids)
            target_ptrs.insert(tgeo.findSurface(target_id));
        
        if( !start_ptr || std::ranges::any_of(target_ptrs, [](auto a){ return a == nullptr; }) )
            throw std::runtime_error("Conversion from GeoID to Surface* failed");
            
        pointer_map[start_ptr] = target_ptrs;
    }
    
    return pointer_map;
}


auto loadBPSplitZBounds(const std::filesystem::path &path)
{
    std::ifstream file(path);
    
    std::vector<double> bounds;
    std::string str;
    
    while( std::getline(file, str) )
        bounds.push_back(std::stod(str));
    
    return bounds;
}



struct AllINeedForMLNavigation
{
    std::shared_ptr<OnnxModel<3,1>> model;
    std::shared_ptr<const Acts::TrackingGeometry> tgeo;
    std::map<const Acts::Surface *, std::set<const Acts::Surface *>> surfaceToGraphTargets;  
    std::vector<double> bpsplitBounds;
};



class MLNavigator
{    
public:
    static constexpr int embeddingDim = 3;
    using EmbeddingVector = Eigen::Matrix<float, embeddingDim, 1>;
    
private:
    std::shared_ptr<OnnxModel<3,1>> m_navigation_model;
    std::shared_ptr<const Acts::TrackingGeometry> m_tracking_geo;
    std::map<const Acts::Surface *, std::set<const Acts::Surface *>> m_surfaceToGraphTargets;  
    std::vector<double> m_bpsplitBounds;
    
public:
    struct State
    {
        // Current set of surfaces considered by the navigator
        std::vector<const Acts::Surface *> navSurfaces;
        std::vector<const Acts::Surface *>::const_iterator navSurfaceIter;
        const Acts::Surface *currentSurface;
        const Acts::Surface *startSurface;
        
        bool navigationBreak = false;
        bool targetReached = false;
        
        const Acts::TrackingVolume *currentVolume = nullptr;
        const Acts::Surface *targetSurface = nullptr;
        
    };
    
    MLNavigator(const AllINeedForMLNavigation &n = AllINeedForMLNavigation()) :
        m_navigation_model(n.model),
        m_tracking_geo(n.tgeo),
        m_surfaceToGraphTargets(n.surfaceToGraphTargets),
        m_bpsplitBounds(n.bpsplitBounds)
    {
        std::cout << "Initialilzed ML Navigator\n";
        std::cout << "Graph Map has " << m_surfaceToGraphTargets.size() << " entries\n";
        std::cout << "BPSplit Bounds have " << m_bpsplitBounds.size() << " entries" << std::endl;
    }
    
    
    /// The whole purpose of this function is to set currentSurface, if possible
    template <typename propagator_state_t, typename stepper_t>
    void status(propagator_state_t& state, const stepper_t& stepper) const 
    {
        try
        {
            const auto& logger = state.options.logger;
            ACTS_VERBOSE(">>>>>>>> STATUS <<<<<<<<<");
            
            if(state.navigation.navigationBreak)
                return;
            
            // Handle initialization
            if(state.navigation.navSurfaces.empty())
            {
                ACTS_VERBOSE("We have no navSurfaceIter, so are during intialization hopefully");
                
                // TODO This is hacky, but for now assume we start at beamline
                state.navigation.currentSurface = m_tracking_geo->getBeamline();
                state.navigation.currentVolume = m_tracking_geo->highestTrackingVolume();
                return;
            }
                
            
            // Navigator status always resets the current surface
            state.navigation.currentSurface = nullptr;
            
            // Establish the surface status
            auto surfaceStatus = stepper.updateSurfaceStatus(state.stepping, **state.navigation.navSurfaceIter, false);
            
            if (surfaceStatus == Acts::Intersection3D::Status::onSurface) 
            {
                // Set the current surface
                state.navigation.currentSurface = *state.navigation.navSurfaceIter;
                ACTS_VERBOSE("Current surface set to  " << state.navigation.currentSurface->geometryId());
                
                // Release Stepsize
                ACTS_VERBOSE("Release Stepsize");
                stepper.releaseStepSize(state.stepping);
                
                // Reset state
                state.navigation.navSurfaces.clear();
                state.navigation.navSurfaceIter = state.navigation.navSurfaces.end();
                
                // Check if we can navigate further
                if( !m_surfaceToGraphTargets.contains(state.navigation.currentSurface) )
                {
                    ACTS_VERBOSE("The current surface was not found in graph map, so we stop the navigation here!");
                    state.navigation.navigationBreak = true;
                    state.navigation.currentVolume = nullptr;
                }
                else
                {
                    ACTS_VERBOSE("The current Surface was found in the graph Map, so we can go on.");
                }
            } 
            else if (surfaceStatus == Acts::Intersection3D::Status::reachable) 
            {
                ACTS_VERBOSE("Next surface reachable at distance  " << stepper.outputStepSize(state.stepping));
            }
            else
            {
                ACTS_VERBOSE("Surface unreachable or missed, hopefully the target(...) call fixes this");
            }
        }
        catch(std::exception &e)
        {
            throw std::runtime_error("Error in MLNavigator::status - "s + e.what());
        }
    }
    
    
    template <typename propagator_state_t, typename stepper_t>
    void target(propagator_state_t& state, const stepper_t& stepper) const 
    {
        const auto& logger = state.options.logger;
        
        try
        {
            ACTS_VERBOSE(">>>>>>>> TARGET <<<<<<<<<");
            const auto& navstate = state.navigation;
            
            // Predict new targets if there are no candidates in state
            if (navstate.navSurfaceIter == navstate.navSurfaces.end()) 
            {
                // This means also we are currently on a surface
                assert(navstate.currentSurface != nullptr);
                ACTS_VERBOSE("It seems like we are on a surface and must predict new targets");
                
                predict_new_target(state.navigation, logger, state.stepping.pars);
            }
            
            // It seems like we are done
            if (navstate.navigationBreak)
            {
                ACTS_VERBOSE("No target Surface, job done.");
                return;
            }
            
            // Check if we are in a correct state
            assert(!state.navigation.navSurfaces.empty());
            assert(state.navigation.navSurfaceIter != state.navigation.navSurfaces.end());
            
            // Navigator target always resets the current surface
            // It is set later by the status call if possible
            state.navigation.currentSurface = nullptr;

            ACTS_VERBOSE("Ask for SurfaceStatus of currently most probable target");
            
            // Establish & update the surface status
            auto surfaceStatus = stepper.updateSurfaceStatus(state.stepping, **state.navigation.navSurfaceIter, false);
            
            ACTS_VERBOSE("After updateSurfaceStatus");
            
            // Everything OK
            if (surfaceStatus == Acts::Intersection3D::Status::reachable) 
            {
                ACTS_VERBOSE("Navigation stepSize set to " << stepper.outputStepSize(state.stepping));
            }
            // Try another surface
            else if (surfaceStatus == Acts::Intersection3D::Status::unreachable) 
            {
                ACTS_VERBOSE("Surface not reachable anymore, search another one which is reachable");
                
                state.navigation.navSurfaces.erase( state.navigation.navSurfaceIter );
                state.navigation.navSurfaceIter = navstate.navSurfaces.end();
                
                if( navstate.navSurfaces.empty() )
                    throw NoSurfaceLeftException("there is now surface left to try");
                
                for(auto iter=navstate.navSurfaces.begin(); iter != navstate.navSurfaces.end(); ++iter)
                {
                    auto new_status = stepper.updateSurfaceStatus(state.stepping, **iter, false);
                    
                    if( new_status == Acts::Intersection3D::Status::reachable )
                    {
                        state.navigation.navSurfaceIter = iter;
                        break;
                    }
                }
                
                if( navstate.navSurfaceIter == navstate.navSurfaces.end() )
                    throw NoSurfaceLeftException("we did not find a suitable surface. In this case we should do a KNN-search, but not yet implemented");
            } 
            // Something strange happended
            else 
            {
                throw std::runtime_error("surface status is 'missed' or 'on_surface', thats not what we want here");
            }
        }
        catch(NoSurfaceLeftException &e)
        {
            ACTS_ERROR("No surface left ("s + e.what() + ") -> Stop navigation for now!");
            
            state.navigation.currentVolume = nullptr;
            state.navigation.navigationBreak = true;
            return;
        }
        catch(std::exception &e)
        {
            throw std::runtime_error("Error in MLNavigator::target() - "s + e.what());
        }
    }
    
private:
    void predict_new_target(State &nav_state, const Acts::LoggerWrapper &logger, const Acts::FreeVector &free_params) const
    {
        auto apply_beampipe_split = [&](double z_pos)
        {
            for(auto it=m_bpsplitBounds.cbegin(); it != std::prev(m_bpsplitBounds.cend()); ++it)
                if( z_pos >= *it && z_pos < *std::next(it) )
                    return EmbeddingVector{ 0.f, 0.f, static_cast<float>(*it + 0.5*(*std::next(it) - *it)) };
                
            throw std::runtime_error("Could not apply embedding");
        };
        
        
        Acts::GeometryContext gctx;
        
        const auto curSurf = nav_state.currentSurface;  
        
        ACTS_VERBOSE("Predict new targets on surface " << curSurf->geometryId() << " - " << curSurf->geometryId().value());
        
        if( !m_surfaceToGraphTargets.contains(curSurf) )
        {
            ACTS_ERROR("Surface was not found in graph map!");
            std::abort();
        };
        
        const auto &possible_targets = m_surfaceToGraphTargets.at(curSurf);
        
        const Eigen::Vector4f in_params = free_params.segment<4>(Acts::eFreeDir0).cast<float>();
        const EmbeddingVector start_emb = curSurf->geometryId().value() == 0 ? apply_beampipe_split(free_params[Acts::eFreePos2])
                                                                             : curSurf->center(gctx).cast<float>();
                
        std::vector<std::pair<float, const Acts::Surface *>> predictions;
        predictions.reserve(possible_targets.size());
        
        ACTS_VERBOSE("Start target prediction loop with " << possible_targets.size() << " targets");
        
        for( auto target : possible_targets )
        {            
            const Eigen::Vector3f target_embedding = target->center(gctx).cast<float>();
            
            auto output = std::tuple<Eigen::Matrix<float, 1, 1>>();
            auto input = std::tuple{ start_emb, target_embedding, in_params };
            
            m_navigation_model->predict(output, input);
            
            predictions.push_back({std::get<0>(output)[0], target});
        }
        
        ACTS_VERBOSE("Finished target prediction loop");
        
        std::ranges::sort(predictions, [&](auto a, auto b){ return a.first > b.first; });
        
        ACTS_VERBOSE("Highest score is " << predictions[0].first << " (" << predictions[0].second->geometryId() << ")");
        
        std::vector<const Acts::Surface *> target_surfaces(predictions.size());
        std::ranges::transform(predictions, target_surfaces.begin(), [](auto a){ return a.second; });
        
        nav_state.navSurfaces = target_surfaces;
        nav_state.navSurfaceIter = nav_state.navSurfaces.begin();
        
        ACTS_VERBOSE("Set 'navSurfaces' and 'navSurfaceIter'");
    }
};
