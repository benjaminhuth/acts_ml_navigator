#ifndef RADIUS_ABORTER_HPP
#define RADIUS_ABORTER_HPP

#include <iostream>
#include <atomic>

#include <Acts/Definitions/Algebra.hpp>
#include <Acts/Definitions/Units.hpp>

using namespace Acts::UnitLiterals;

struct TrackMLBoundaryAborter {
    TrackMLBoundaryAborter() = default;
  double m_max_radius = 1100_mm;
  double m_max_z = 3000_mm;
  
  static std::atomic<int> s_num_abortions;

  template <typename propagator_state_t, typename stepper_t>
  bool operator()(propagator_state_t& state,
                  const stepper_t& stepper) const 
  {
    Acts::Vector3 pos = stepper.position(state.stepping);
    
    if( auto radius = pos.segment<2>(0).norm(); radius > m_max_radius 
        || std::abs(pos[2]) > m_max_z )
    {
        std::cout << "RadiusAborter stopped propagation at radius " << radius
                  << "and z-pos " << pos[2] << ", so fare had " 
                  << ++s_num_abortions << " abortions!" << std::endl;
        
        state.navigation.targetReached = true;
        state.navigation.navigationBreak = true;
        return true;
    }
    
    return false;
  }
};

std::atomic<int> TrackMLBoundaryAborter::s_num_abortions = 0;

#endif
