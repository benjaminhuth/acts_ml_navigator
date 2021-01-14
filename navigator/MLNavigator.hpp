#pragma once
#include <filesystem>

#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Geometry/GeometryIdentifier.hpp>

#include "OnnxModel.hpp"


struct MLNavigatorOptions
{
    std::filesystem::path navModelPath;
    std::filesystem::path embModelPath;
    std::filesystem::path detectorCSVPath; 
};

class MLNavigator
{    
public:
    static constexpr int embeddingDim = 10;
    
    using EmbeddingVector = Eigen::Matrix<float, embeddingDim, 1>;
    
    struct State
    {
    };
    
    MLNavigator(const MLNavigatorOptions &options) :
        m_navigation_model(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "navigation_model"), options.navModelPath),
        m_embedding_model(Ort::Env(ORT_LOGGING_LEVEL_WARNING, "embedding_model"), options.embModelPath)
    {
    }
    
    template <typename propagator_state_t, typename stepper_t>
    void status(propagator_state_t& state, const stepper_t& stepper) const 
    {
        ///////////////////////////////////////////////////////////////
        // ALL THIS IS COPIED FROM DIRECT NAVIGATOR, NOT YET ADJUSTED
        ///////////////////////////////////////////////////////////////
        
        
        const auto& logger = state.options.logger;
        ACTS_VERBOSE("Entering navigator::status.");
        
        // Navigator status always resets the current surface
        state.navigation.currentSurface = nullptr;
        
        // Check if we are on surface
        if (state.navigation.navSurfaceIter != state.navigation.navSurfaces.end()) 
        {   
            // Establish the surface status
            auto surfaceStatus = stepper.updateSurfaceStatus(state.stepping, **state.navigation.navSurfaceIter, false);
            
            if (surfaceStatus == Acts::Intersection3D::Status::onSurface) 
            {
                // Set the current surface
                state.navigation.currentSurface = *state.navigation.navSurfaceIter;
                ACTS_VERBOSE("Current surface set to  "
                            << state.navigation.currentSurface->geometryId())
                
                // Move the sequence to the next surface
                ++state.navigation.navSurfaceIter;
                
                if (state.navigation.navSurfaceIter != state.navigation.navSurfaces.end()) 
                {
                    ACTS_VERBOSE("Next surface candidate is  "
                                << (*state.navigation.navSurfaceIter)->geometryId());
                    stepper.releaseStepSize(state.stepping);
                }
            } 
            else if (surfaceStatus == Acts::Intersection3D::Status::reachable) 
            {
                ACTS_VERBOSE("Next surface reachable at distance  "
                            << stepper.outputStepSize(state.stepping));
            }
        }
    }
    
    
    template <typename propagator_state_t, typename stepper_t>
    void target(propagator_state_t& state, const stepper_t& stepper) const 
    {
        ///////////////////////////////////////////////////////////////
        // ALL THIS IS COPIED FROM DIRECT NAVIGATOR, NOT YET ADJUSTED
        ///////////////////////////////////////////////////////////////
        
        const auto& logger = state.options.logger;
        ACTS_VERBOSE("Entering navigator::target.");

        // Navigator target always resets the current surface
        state.navigation.currentSurface = nullptr;

        if (state.navigation.navSurfaceIter != state.navigation.navSurfaces.end()) 
        {
            // Establish & update the surface status
            auto surfaceStatus = stepper.updateSurfaceStatus(
                state.stepping, **state.navigation.navSurfaceIter, false);
            
            if (surfaceStatus == Acts::Intersection3D::Status::unreachable) 
            {
                ACTS_VERBOSE("Surface not reachable anymore, switching to next one in sequence");
                
                // Move the sequence to the next surface
                ++state.navigation.navSurfaceIter;
            } 
            else 
            {
                ACTS_VERBOSE("Navigation stepSize set to " << stepper.outputStepSize(state.stepping));
            }
        } 
        else 
        {
            state.navigation.navigationBreak = true;
            
            // If no externally provided target is given, the target is reached
            if (state.navigation.targetSurface == nullptr) 
            {
                state.navigation.targetReached = true;
                // Announce it then
                ACTS_VERBOSE("No target Surface, job done.");
            }
        }
    }
    
private:
    const OnnxModel<2,1> m_navigation_model;
    const OnnxModel<1,1> m_embedding_model;
    
    std::map<const Acts::Surface *, EmbeddingVector> m_surfaceToEmbedding;
    std::map<uint64_t, const Acts::Surface *> m_;
};
