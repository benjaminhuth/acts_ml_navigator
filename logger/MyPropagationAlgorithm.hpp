// This file is part of the Acts project.
//
// Copyright (C) 2017 CERN for the benefit of the Acts project
//
// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at http://mozilla.org/MPL/2.0/.

#pragma once

#include "Acts/EventData/NeutralTrackParameters.hpp"
#include "Acts/EventData/TrackParameters.hpp"
#include "Acts/Propagator/AbortList.hpp"
#include "Acts/Propagator/ActionList.hpp"
#include "Acts/Propagator/DenseEnvironmentExtension.hpp"
#include "Acts/Propagator/MaterialInteractor.hpp"
#include "Acts/Propagator/Navigator.hpp"
#include "Acts/Propagator/Propagator.hpp"
#include "Acts/Propagator/StandardAborters.hpp"
#include "Acts/Propagator/detail/SteppingLogger.hpp"
#include "Acts/Surfaces/PerigeeSurface.hpp"
#include "Acts/Definitions/Algebra.hpp"
#include "Acts/Definitions/Units.hpp"
#include "Acts/Utilities/Helpers.hpp"
#include "ActsExamples/Framework/BareAlgorithm.hpp"
#include "ActsExamples/Framework/ProcessCode.hpp"
#include "ActsExamples/Framework/RandomNumbers.hpp"
#include "ActsExamples/Framework/WhiteBoard.hpp"
#include "ActsExamples/Propagation/PropagationAlgorithm.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <random>

#include "SurfaceLogger.hpp"

using namespace Acts::UnitLiterals;


/// Using some short hands for Recorded Material
using RecordedMaterial = Acts::MaterialInteractor::result_type;

/// And recorded material track
/// - this is start:  position, start momentum
///   and the Recorded material
using RecordedMaterialTrack =
    std::pair<std::pair<Acts::Vector3, Acts::Vector3>, RecordedMaterial>;

/// Finally the output of the propagation test
using PropagationOutput =
    std::pair<std::vector<Acts::detail::Step>, RecordedMaterial>;

/// @brief this test algorithm performs test propagation
/// within the Acts::Propagator
///
/// If the propagator is equipped appropriately, it can
/// also be used to test the Extrapolator within the geomtetry
///
/// @tparam propagator_t Type of the Propagator to be tested
template <typename propagator_t>
class MyPropagationAlgorithm : public ActsExamples::BareAlgorithm {
 public:
  using Config = typename ActsExamples::PropagationAlgorithm<propagator_t>::Config;

  /// Constructor
  /// @param [in] cnf is the configuration struct
  /// @param [in] loglevel is the loggin level
  MyPropagationAlgorithm(const Config& cnf, Acts::Logging::Level loglevel);

  /// Framework execute method
  /// @param [in] the algorithm context for event consistency
  /// @return is a process code indicating succes or not
  ActsExamples::ProcessCode execute(
      const ActsExamples::AlgorithmContext& context) const final override;

 private:
  Config m_cfg;  ///< the config class

  /// Private helper method to create a corrleated covariance matrix
  /// @param[in] rnd is the random engine
  /// @param[in] gauss is a gaussian distribution to draw from
  std::optional<Acts::BoundSymMatrix> generateCovariance(
      ActsExamples::RandomEngine& rnd,
      std::normal_distribution<double>& gauss) const;

  /// Templated execute test method for
  /// charged and netural particles
  ///
  // @tparam parameters_t type of the parameters objects (charged/neutra;)
  ///
  /// @param [in] context The Context for this call
  /// @param [in] startParameters the start parameters
  /// @param [in] pathLengthe the path limit of this propagation
  ///
  /// @return collection of Propagation steps for further analysis
  template <typename parameters_t>
  PropagationOutput executeTest(
      const ActsExamples::AlgorithmContext& context, const parameters_t& startParameters,
      double pathLength = std::numeric_limits<double>::max()) const;
};



////////////////////////
// IMPLEMENTATION
////////////////////////



template <typename propagator_t>
std::optional<Acts::BoundSymMatrix>
MyPropagationAlgorithm<propagator_t>::generateCovariance(
    ActsExamples::RandomEngine& rnd,
    std::normal_distribution<double>& gauss) const {
  if (m_cfg.covarianceTransport) {
    // We start from the correlation matrix
    Acts::BoundSymMatrix newCov(m_cfg.correlations);
    // Then we draw errors according to the error values
    Acts::BoundVector covs_smeared = m_cfg.covariances;
    for (size_t k = 0; k < size_t(covs_smeared.size()); ++k) {
      covs_smeared[k] *= gauss(rnd);
    }
    // and apply a double loop
    for (size_t i = 0; i < size_t(newCov.rows()); ++i) {
      for (size_t j = 0; j < size_t(newCov.cols()); ++j) {
        (newCov)(i, j) *= covs_smeared[i];
        (newCov)(i, j) *= covs_smeared[j];
      }
    }
    return newCov;
  }
  return std::nullopt;
}

template <typename propagator_t>
MyPropagationAlgorithm<propagator_t>::MyPropagationAlgorithm(
    const MyPropagationAlgorithm<propagator_t>::Config& cfg,
    Acts::Logging::Level loglevel)
    : BareAlgorithm("MyPropagationAlgorithm", loglevel), m_cfg(cfg) 
{
    
}


template <typename propagator_t>
template <typename parameters_t>
PropagationOutput MyPropagationAlgorithm<propagator_t>::executeTest(
    const ActsExamples::AlgorithmContext& context, const parameters_t& startParameters,
    double pathLength) const {
  ACTS_DEBUG("Test propagation/extrapolation starts");

  PropagationOutput pOutput;
  
  // This is the outside in mode
  if (m_cfg.mode == 0) {
    // The step length logger for testing & end of world aborter
    using MaterialInteractor = Acts::MaterialInteractor;
    using SteppingLogger = Acts::detail::SteppingLogger;
    using EndOfWorld = Acts::EndOfWorldReached;
    
    // Action list and abort list WITH NEW LOGGER
    using ActionList = Acts::ActionList<SteppingLogger, MaterialInteractor, SurfaceLogger>;
    using AbortList = Acts::AbortList<EndOfWorld>;
    using PropagatorOptions =
        Acts::DenseStepperPropagatorOptions<ActionList, AbortList>;

    PropagatorOptions options(context.geoContext, context.magFieldContext,
                              Acts::LoggerWrapper{logger()});
    options.pathLimit = pathLength;

    // Activate loop protection at some pt value
    options.loopProtection =
        (startParameters.transverseMomentum() < m_cfg.ptLoopers);

    // Switch the material interaction on/off & eventually into logging mode
    auto& mInteractor = options.actionList.get<MaterialInteractor>();
    mInteractor.multipleScattering = m_cfg.multipleScattering;
    mInteractor.energyLoss = m_cfg.energyLoss;
    mInteractor.recordInteractions = m_cfg.recordMaterialInteractions;

    // Set a maximum step size
    options.maxStepSize = m_cfg.maxStepSize;

    // Propagate using the propagator
    const auto& result =
        m_cfg.propagator.propagate(startParameters, options).value();
    auto steppingResults = result.template get<SteppingLogger::result_type>();

    // Set the stepping result
    pOutput.first = std::move(steppingResults.steps);
    // Also set the material recording result - if configured
    if (m_cfg.recordMaterialInteractions) {
      auto materialResult =
          result.template get<MaterialInteractor::result_type>();
      pOutput.second = std::move(materialResult);
    }
  }
  return pOutput;
}



template <typename propagator_t>
ActsExamples::ProcessCode MyPropagationAlgorithm<propagator_t>::execute(
    const ActsExamples::AlgorithmContext& context) const {
  // Create a random number generator
  ActsExamples::RandomEngine rng =
      m_cfg.randomNumberSvc->spawnGenerator(context);

  // Standard gaussian distribution for covarianmces
  std::normal_distribution<double> gauss(0., 1.);

  // Setup random number distributions for some quantities
  std::uniform_real_distribution<double> phiDist(m_cfg.phiRange.first,
                                                 m_cfg.phiRange.second);
  std::uniform_real_distribution<double> etaDist(m_cfg.etaRange.first,
                                                 m_cfg.etaRange.second);
  std::uniform_real_distribution<double> ptDist(m_cfg.ptRange.first,
                                                m_cfg.ptRange.second);
  std::uniform_real_distribution<double> qDist(0., 1.);

  std::shared_ptr<const Acts::PerigeeSurface> surface =
      Acts::Surface::makeShared<Acts::PerigeeSurface>(
          Acts::Vector3(0., 0., 0.));

  // Output : the propagation steps
  std::vector<std::vector<Acts::detail::Step>> propagationSteps;
  propagationSteps.reserve(m_cfg.ntests);

  // Output (optional): the recorded material
  std::vector<RecordedMaterialTrack> recordedMaterial;
  if (m_cfg.recordMaterialInteractions) {
    recordedMaterial.reserve(m_cfg.ntests);
  }

  // loop over number of particles
  for (size_t it = 0; it < m_cfg.ntests; ++it) {
    /// get the d0 and z0
    double d0 = m_cfg.d0Sigma * gauss(rng);
    double z0 = m_cfg.z0Sigma * gauss(rng);
    double phi = phiDist(rng);
    double eta = etaDist(rng);
    double theta = 2 * atan(exp(-eta));
    double pt = ptDist(rng);
    double p = pt / sin(theta);
    double charge = qDist(rng) > 0.5 ? 1. : -1.;
    double qop = charge / p;
    double t = m_cfg.tSigma * gauss(rng);
    // parameters
    Acts::BoundVector pars;
    pars << d0, z0, phi, theta, qop, t;
    // some screen output

    Acts::Vector3 sPosition(0., 0., 0.);
    Acts::Vector3 sMomentum(0., 0., 0.);

    // The covariance generation
    auto cov = generateCovariance(rng, gauss);

    // execute the test for charged particles
    PropagationOutput pOutput;
    if (charge) {
      // charged extrapolation - with hit recording
      Acts::BoundTrackParameters startParameters(surface, std::move(pars),
                                                 std::move(cov));
      sPosition = startParameters.position(context.geoContext);
      sMomentum = startParameters.momentum();
      pOutput = executeTest(context, startParameters);
    } else {
      // execute the test for neeutral particles
      Acts::NeutralBoundTrackParameters neutralParameters(
          surface, std::move(pars), std::move(cov));
      sPosition = neutralParameters.position(context.geoContext);
      sMomentum = neutralParameters.momentum();
      pOutput = executeTest(context, neutralParameters);
    }
    // Record the propagator steps
    propagationSteps.push_back(std::move(pOutput.first));
    if (m_cfg.recordMaterialInteractions &&
        pOutput.second.materialInteractions.size()) {
      // Create a recorded material track
      RecordedMaterialTrack rmTrack;
      // Start position
      rmTrack.first.first = std::move(sPosition);
      // Start momentum
      rmTrack.first.second = std::move(sMomentum);
      // The material
      rmTrack.second = std::move(pOutput.second);
      // push it it
      recordedMaterial.push_back(std::move(rmTrack));
    }
  }

  // Write the propagation step data to the event store
  context.eventStore.add(m_cfg.propagationStepCollection,
                         std::move(propagationSteps));

  // Write the recorded material to the event store
  if (m_cfg.recordMaterialInteractions) {
    context.eventStore.add(m_cfg.propagationMaterialCollection,
                           std::move(recordedMaterial));
  }

  return ActsExamples::ProcessCode::SUCCESS;
}
