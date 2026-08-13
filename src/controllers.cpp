#include "vessel_gnc/controllers.hpp"

#include <algorithm>
#include <cmath>

namespace vessel_gnc {

double wrap_to_pi(double angle) {
    return std::atan2(std::sin(angle), std::cos(angle));
}

HeadingController::HeadingController(PidGains gains, double moment_limit, double integrator_limit)
    : gains_(gains), moment_limit_(moment_limit), integrator_limit_(integrator_limit) {}

void HeadingController::reset() {
    integral_ = 0.0;
}

double HeadingController::update(double heading_ref, double heading, double yaw_rate, double dt) {
    const double error = wrap_to_pi(heading_ref - heading);
    const double out_pd = gains_.kp * error - gains_.kd * yaw_rate;
    const double out = out_pd + integral_;
    // Anti-windup: freeze the integrator when saturated in the same direction.
    if (!(std::abs(out) >= moment_limit_ && error * out > 0.0)) {
        integral_ =
            std::clamp(integral_ + gains_.ki * error * dt, -integrator_limit_, integrator_limit_);
    }
    return std::clamp(out_pd + integral_, -moment_limit_, moment_limit_);
}

SpeedController::SpeedController(PidGains gains, double thrust_limit, double integrator_limit)
    : gains_(gains), thrust_limit_(thrust_limit), integrator_limit_(integrator_limit) {}

void SpeedController::reset() {
    integral_ = 0.0;
}

double SpeedController::update(double speed_ref, double surge_speed, double dt) {
    const double error = speed_ref - surge_speed;
    const double out = gains_.kp * error + integral_;
    // Anti-windup: freeze the integrator when saturated in the same direction.
    if (!(std::abs(out) >= thrust_limit_ && error * out > 0.0)) {
        integral_ =
            std::clamp(integral_ + gains_.ki * error * dt, -integrator_limit_, integrator_limit_);
    }
    return std::clamp(out, -thrust_limit_, thrust_limit_);
}

PidGains default_heading_gains() {
    // Tuned against the default vessel (docs/control.md §3): the yaw plant is
    // heavily damped (N_r / m33 ~ 5 s^-1), so kp dominates and the loop is
    // overdamped; ki removes the steady-state heading offset.
    return PidGains{12.0, 0.1, 0.5};
}

PidGains default_speed_gains() {
    // Tuned against the default vessel (docs/control.md §3): the surge plant
    // time constant is m11 / (X_u + 2 X_|u|u u_eq) ~ 0.8 s. The PI slow pole
    // sits at ki / (kp + d') ~ 0.21 s^-1 (tau ~ 5 s), keeping the loop
    // aperiodic (two real poles).
    return PidGains{25.0, 15.0, 0.0};
}

}  // namespace vessel_gnc
