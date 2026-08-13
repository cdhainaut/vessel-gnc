#pragma once

#include "vessel_gnc/state.hpp"

namespace vessel_gnc {

// Wrap an angle to (-pi, pi].
double wrap_to_pi(double angle);

// PID gains (units depend on the controlled quantity).
struct PidGains {
    double kp = 0.0;
    double ki = 0.0;
    double kd = 0.0;
};

// Heading controller: PID on the wrapped heading error, derivative on the
// measured yaw rate (no derivative kick), conditional integration with
// integrator clamping (anti-windup) and symmetric output saturation.
// See docs/control.md §2 for the formulation and tuning.
class HeadingController {
public:
    HeadingController(PidGains gains, double moment_limit, double integrator_limit);

    void reset();

    // One control step of duration dt [s]. Returns the yaw moment [N m],
    // clamped to [-moment_limit, moment_limit].
    double update(double heading_ref, double heading, double yaw_rate, double dt);

private:
    PidGains gains_;
    double moment_limit_;
    double integrator_limit_;
    double integral_ = 0.0;
};

// Surge speed controller: PI on the speed error with conditional integration
// (anti-windup) and symmetric thrust saturation. See docs/control.md §2.
class SpeedController {
public:
    SpeedController(PidGains gains, double thrust_limit, double integrator_limit);

    void reset();

    // One control step of duration dt [s]. Returns the surge thrust [N],
    // clamped to [-thrust_limit, thrust_limit].
    double update(double speed_ref, double surge_speed, double dt);

private:
    PidGains gains_;
    double thrust_limit_;
    double integrator_limit_;
    double integral_ = 0.0;
};

// Gains tuned for the default vessel parameters (docs/control.md §3).
PidGains default_heading_gains();
PidGains default_speed_gains();

}  // namespace vessel_gnc
