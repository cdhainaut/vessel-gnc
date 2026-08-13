#pragma once

#include "vessel_gnc/state.hpp"

namespace vessel_gnc {

// Advance the vessel state by one fixed step dt with the classical Runge-Kutta
// 4 method. Control and environment are held constant over the step
// (zero-order hold). Assumes finite inputs and dt > 0.
State rk4_step(const State& state,
               const Control& control,
               const Environment& environment,
               const ModelParams& params,
               double dt);

}  // namespace vessel_gnc
