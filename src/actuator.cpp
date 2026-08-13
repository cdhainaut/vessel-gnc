#include "vessel_gnc/actuator.hpp"

#include <algorithm>
#include <cmath>

namespace vessel_gnc {

namespace {

// Smooth rate-limited proportional response (docs/model.md §5): the hard
// clamp is replaced by rate * tanh(...), which keeps |dot| <= rate strictly
// and makes the model C1 (fast convergence for gradient-based solvers).
// Small-signal behaviour is the plain first-order lag.
double thrust_dot(double thrust, double thrust_cmd, const ModelParams& params) {
    const double command = std::clamp(thrust_cmd, params.thrust_min, params.thrust_max);
    const double scaled_error = (command - thrust) / params.thrust_time_constant;
    return params.thrust_rate_limit * std::tanh(scaled_error / params.thrust_rate_limit);
}

double moment_dot(double moment, double moment_cmd, const ModelParams& params) {
    const double command = std::clamp(moment_cmd, params.moment_min, params.moment_max);
    const double scaled_error = (command - moment) / params.moment_time_constant;
    return params.moment_rate_limit * std::tanh(scaled_error / params.moment_rate_limit);
}

}  // namespace

ActuatorState actuator_step(const ActuatorState& actuator,
                            const Control& command,
                            const ModelParams& params,
                            double dt) {
    // The two actuator channels are decoupled: RK4 on each scalar ODE.
    const double h = dt;
    const double k1_t = thrust_dot(actuator.thrust, command.thrust, params);
    const double k1_m = moment_dot(actuator.yaw_moment, command.yaw_moment, params);
    const double k2_t =
        thrust_dot(actuator.thrust + h / 2.0 * k1_t, command.thrust, params);
    const double k2_m =
        moment_dot(actuator.yaw_moment + h / 2.0 * k1_m, command.yaw_moment, params);
    const double k3_t =
        thrust_dot(actuator.thrust + h / 2.0 * k2_t, command.thrust, params);
    const double k3_m =
        moment_dot(actuator.yaw_moment + h / 2.0 * k2_m, command.yaw_moment, params);
    const double k4_t = thrust_dot(actuator.thrust + h * k3_t, command.thrust, params);
    const double k4_m = moment_dot(actuator.yaw_moment + h * k3_m, command.yaw_moment, params);

    ActuatorState next;
    next.thrust = std::clamp(
        actuator.thrust + h / 6.0 * (k1_t + 2.0 * k2_t + 2.0 * k3_t + k4_t),
        params.thrust_min, params.thrust_max);
    next.yaw_moment = std::clamp(
        actuator.yaw_moment + h / 6.0 * (k1_m + 2.0 * k2_m + 2.0 * k3_m + k4_m),
        params.moment_min, params.moment_max);
    return next;
}

}  // namespace vessel_gnc
