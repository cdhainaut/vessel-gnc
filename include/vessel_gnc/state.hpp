#pragma once

namespace vessel_gnc {

// Vessel state: horizontal-plane position and heading (inertial frame) plus
// body-frame velocities. SI units throughout; angles in radians.
// Frames: inertial NED projection (x -> North, y -> East), body x -> forward
// (surge), body y -> starboard (sway); see docs/model.md §1.
struct State {
    double x = 0.0;    // [m]     inertial position, North component
    double y = 0.0;    // [m]     inertial position, East component
    double psi = 0.0;  // [rad]   heading, from North, clockwise positive
    double u = 0.0;    // [m/s]   surge velocity (body x, forward)
    double v = 0.0;    // [m/s]   sway velocity (body y, starboard)
    double r = 0.0;    // [rad/s] yaw rate (body z, clockwise positive)
};

// Actuator command: desired surge force and yaw moment (docs/model.md §5).
// The plant does not apply commands instantly: it goes through the actuator
// dynamics (ActuatorState / actuator_step).
struct Control {
    double thrust = 0.0;      // [N]   surge force, positive forward
    double yaw_moment = 0.0;  // [N m] yaw moment, clockwise positive
};

// Actuator state: the forces actually applied to the vessel, after the
// first-order response lag, rate limiting and saturation of the command
// (docs/model.md §5).
struct ActuatorState {
    double thrust = 0.0;      // [N]
    double yaw_moment = 0.0;  // [N m]
};

// Ambient disturbances, both expressed in the inertial (NED) frame.
struct Environment {
    double current_north = 0.0;  // [m/s] ambient current velocity, North component
    double current_east = 0.0;   // [m/s] ambient current velocity, East component
    double wind_north = 0.0;     // [N]   wind force on the hull, North component
    double wind_east = 0.0;      // [N]   wind force on the hull, East component
};

// Vessel model parameters: masses, damping coefficients and actuator limits.
// Defaults are illustrative small-USV values, see docs/model.md §6.
struct ModelParams {
    // Rigid body
    double mass = 30.0;       // [kg]
    double inertia_z = 4.0;   // [kg m^2] yaw inertia about the centre of gravity
    // Added mass (diagonal approximation)
    double added_mass_x = 5.0;    // [kg]
    double added_mass_y = 20.0;   // [kg]
    double added_inertia_z = 2.0; // [kg m^2]
    // Linear damping
    double lin_damping_u = 2.0;   // [N s/m]
    double lin_damping_v = 150.0; // [N s/m]
    double lin_damping_r = 30.0;  // [N m s/rad]
    // Quadratic damping
    double quad_damping_u = 20.0;   // [N s^2/m^2]
    double quad_damping_v = 250.0;  // [N s^2/m^2]
    double quad_damping_r = 60.0;   // [N m s^2/rad^2]
    // Actuator saturation
    double thrust_min = -20.0;  // [N]
    double thrust_max = 60.0;   // [N]
    double moment_min = -6.0;   // [N m]
    double moment_max = 6.0;    // [N m]
    // Actuator dynamics (illustrative first-order response with rate limits,
    // see docs/model.md §5). The time constants are slow enough to remain
    // well resolved by the NMPC model step (0.4 s) and the rate limits bind
    // for full-scale commands.
    double thrust_time_constant = 1.0;  // [s]
    double moment_time_constant = 0.6;  // [s]
    double thrust_rate_limit = 50.0;    // [N/s]
    double moment_rate_limit = 8.0;     // [N m/s]
};

}  // namespace vessel_gnc
