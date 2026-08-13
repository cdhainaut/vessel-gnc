#pragma once

#include "vessel_gnc/state.hpp"

namespace vessel_gnc {

// Advance the actuator state by one fixed step dt (RK4 on the rate-limited
// first-order response, docs/model.md §5):
//     dT/dt = clamp((T_cmd - T) / tau_T, -T_dot_max, +T_dot_max)
//     dN/dt = clamp((N_cmd - N) / tau_N, -N_dot_max, +N_dot_max)
// with the state projected back into the saturation bounds after the step.
// The command is clamped to the actuator bounds before the response.
// Assumes finite inputs and dt > 0.
ActuatorState actuator_step(const ActuatorState& actuator,
                            const Control& command,
                            const ModelParams& params,
                            double dt);

}  // namespace vessel_gnc
