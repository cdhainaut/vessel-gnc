// Unit and physical-validation tests for kinematics, dynamics and the
// environment model. See docs/model.md §8 for the mapping to validation cases.

#include <gtest/gtest.h>

#include <cmath>

#include <Eigen/Dense>

#include "vessel_gnc/dynamics.hpp"

using namespace vessel_gnc;

namespace {
constexpr double kTol = 1e-9;
}

TEST(RotationMatrix, IsOrthonormal) {
    for (double psi : {0.0, 0.3, 1.7, -2.9, 6.28}) {
        const Eigen::Matrix2d R = rotation_matrix(psi);
        const Eigen::Matrix2d I = R * R.transpose();
        EXPECT_NEAR(I(0, 0), 1.0, 1e-12);
        EXPECT_NEAR(I(0, 1), 0.0, 1e-12);
        EXPECT_NEAR(I(1, 0), 0.0, 1e-12);
        EXPECT_NEAR(I(1, 1), 1.0, 1e-12);
        EXPECT_NEAR(R.determinant(), 1.0, 1e-12);
    }
}

TEST(Kinematics, PureSurgeNorth) {
    // Heading North, surge only -> motion along +x.
    State s;
    s.u = 2.0;
    const State d = derivative(s, Control{}, Environment{}, default_params());
    EXPECT_NEAR(d.x, 2.0, kTol);
    EXPECT_NEAR(d.y, 0.0, kTol);
    EXPECT_NEAR(d.psi, 0.0, kTol);
}

TEST(Kinematics, PureSurgeEast) {
    // Heading East (psi = +pi/2), surge only -> motion along +y.
    State s;
    s.psi = M_PI_2;
    s.u = 2.0;
    const State d = derivative(s, Control{}, Environment{}, default_params());
    EXPECT_NEAR(d.x, 0.0, kTol);
    EXPECT_NEAR(d.y, 2.0, kTol);
}

TEST(Kinematics, PureYaw) {
    // Yaw rate only -> heading rate only, no translation.
    State s;
    s.r = 0.3;
    const State d = derivative(s, Control{}, Environment{}, default_params());
    EXPECT_NEAR(d.psi, 0.3, kTol);
    EXPECT_NEAR(d.x, 0.0, kTol);
    EXPECT_NEAR(d.y, 0.0, kTol);
}

TEST(Dynamics, ZeroForceZeroDampingKeepsVelocity) {
    // Plan validation case A: no actuation, no damping -> velocity unchanged.
    // Valid for states with v = r = 0 (the Coriolis term vanishes; for mixed
    // velocities the undamped system exchanges energy between axes while
    // conserving kinetic energy — see CoriolisConservesKineticEnergy).
    ModelParams p = default_params();
    p.lin_damping_u = 0.0;
    p.lin_damping_v = 0.0;
    p.lin_damping_r = 0.0;
    p.quad_damping_u = 0.0;
    p.quad_damping_v = 0.0;
    p.quad_damping_r = 0.0;
    State s;
    s.psi = 0.7;
    s.u = 1.5;
    const State d = derivative(s, Control{}, Environment{}, p);
    EXPECT_NEAR(d.u, 0.0, kTol);
    EXPECT_NEAR(d.v, 0.0, kTol);
    EXPECT_NEAR(d.r, 0.0, kTol);
    // Kinematic part: eta_dot = R(psi) nu.
    EXPECT_NEAR(d.x, 1.5 * std::cos(0.7), kTol);
    EXPECT_NEAR(d.y, 1.5 * std::sin(0.7), kTol);
    EXPECT_NEAR(d.psi, 0.0, kTol);
}

TEST(Dynamics, DampingDeceleratesForwardMotion) {
    State s;
    s.u = 2.0;
    const State d = derivative(s, Control{}, Environment{}, default_params());
    EXPECT_LT(d.u, 0.0);         // drag opposes forward motion
    EXPECT_NEAR(d.v, 0.0, kTol); // symmetric hull: no sway force at v = 0
    EXPECT_NEAR(d.r, 0.0, kTol); // no yaw moment at r = 0
}

TEST(Dynamics, CoriolisConservesKineticEnergy) {
    // Without damping and external forces, C(nu) does no work:
    // nu^T M nu_dot = -nu^T C(nu) nu = 0 (C is skew-symmetric).
    ModelParams p = default_params();
    p.lin_damping_u = 0.0;
    p.lin_damping_v = 0.0;
    p.lin_damping_r = 0.0;
    p.quad_damping_u = 0.0;
    p.quad_damping_v = 0.0;
    p.quad_damping_r = 0.0;
    State s;
    s.u = 1.2;
    s.v = -0.4;
    s.r = 0.3;
    const State d = derivative(s, Control{}, Environment{}, p);
    const double m11 = p.mass + p.added_mass_x;
    const double m22 = p.mass + p.added_mass_y;
    const double m33 = p.inertia_z + p.added_inertia_z;
    const double power = m11 * s.u * d.u + m22 * s.v * d.v + m33 * s.r * d.r;
    EXPECT_NEAR(power, 0.0, 1e-9);
}

TEST(Dynamics, SurgeEquilibriumMatchesDragBalance) {
    // Plan validation case B: at u_eq solving T = X_u u + X_|u|u u^2 the surge
    // acceleration vanishes; below u_eq the vessel accelerates.
    ModelParams p = default_params();
    const double thrust = 40.0;
    const double a = p.quad_damping_u;
    const double b = p.lin_damping_u;
    const double u_eq = (-b + std::sqrt(b * b + 4.0 * a * thrust)) / (2.0 * a);

    State at_eq;
    at_eq.u = u_eq;
    const State d_eq = derivative(at_eq, Control{thrust}, Environment{}, p);
    EXPECT_NEAR(d_eq.u, 0.0, 1e-9);

    State below;
    below.u = 0.5 * u_eq;
    const State d_below = derivative(below, Control{thrust}, Environment{}, p);
    EXPECT_GT(d_below.u, 0.0);
}

TEST(Dynamics, SteadyYawRateBalancesMoment) {
    // Plan validation case C (partial): at the yaw rate balancing the applied
    // moment against yaw drag (evaluated at v = 0), the yaw acceleration
    // vanishes. The Coriolis coupling then pushes the vessel toward a port
    // sideslip (bow into the turn): the full steady-turn equilibrium requires
    // v < 0 for a clockwise turn.
    ModelParams p = default_params();
    const double moment = 2.0;
    const double r_eq = (-p.lin_damping_r
                         + std::sqrt(p.lin_damping_r * p.lin_damping_r
                                     + 4.0 * p.quad_damping_r * moment))
                        / (2.0 * p.quad_damping_r);
    State s;
    s.u = 1.4;
    s.r = r_eq;
    const State d = derivative(s, Control{.yaw_moment = moment}, Environment{}, p);
    EXPECT_NEAR(d.r, 0.0, 1e-9);
    EXPECT_LT(d.v, 0.0);
}

TEST(Environment, CurrentDragsVesselAlong) {
    // Damping acts on the relative velocity: a vessel at rest in a current is
    // dragged along with it, and a vessel moving exactly with the current
    // feels no hydrodynamic force.
    ModelParams p = default_params();
    Environment env;
    env.current_east = 0.5;  // [m/s]

    const State d = derivative(State{}, Control{}, env, p);
    EXPECT_GT(d.v, 0.0);        // sway acceleration to starboard (East at psi = 0)
    EXPECT_NEAR(d.x, 0.0, kTol);

    State drifting;
    drifting.v = 0.5;  // moving exactly with the current
    const State d_drift = derivative(drifting, Control{}, env, p);
    EXPECT_NEAR(d_drift.u, 0.0, kTol);
    EXPECT_NEAR(d_drift.v, 0.0, kTol);
}

TEST(Environment, TurningBodyTransportsConstantInertialCurrent) {
    // With zero relative translational velocity, hydrodynamic loads vanish.
    // The absolute body velocity must still rotate as the body turns:
    // v_c_dot^b = -S(r) v_c^b = [r v_c, -r u_c].
    ModelParams p = default_params();
    p.lin_damping_r = 0.0;
    p.quad_damping_r = 0.0;

    Environment env;
    env.current_north = 0.4;
    env.current_east = -0.2;

    State s;
    s.psi = 0.7;
    s.r = 0.3;
    const Eigen::Vector2d current_body =
        rotation_matrix(s.psi).transpose()
        * Eigen::Vector2d(env.current_north, env.current_east);
    s.u = current_body.x();
    s.v = current_body.y();

    const State d = derivative(s, Control{}, env, p);
    EXPECT_NEAR(d.u, s.r * current_body.y(), 1e-12);
    EXPECT_NEAR(d.v, -s.r * current_body.x(), 1e-12);
    EXPECT_NEAR(d.r, 0.0, 1e-12);
}

TEST(Environment, WindForceActsInInertialDirection) {
    // Wind is an inertial-frame force: the same eastward wind pushes sway at
    // psi = 0 and surge at psi = pi/2.
    ModelParams p = default_params();
    Environment env;
    env.wind_east = 10.0;  // [N]

    State s;
    const State d_sway = derivative(s, Control{}, env, p);
    EXPECT_NEAR(d_sway.v * (p.mass + p.added_mass_y), 10.0, 1e-9);

    s.psi = M_PI_2;
    const State d_surge = derivative(s, Control{}, env, p);
    EXPECT_NEAR(d_surge.u * (p.mass + p.added_mass_x), 10.0, 1e-9);
}

TEST(Control, ClampRespectsSaturation) {
    ModelParams p = default_params();
    const Control over{1e3, -1e3};
    const Control clamped = clamp_control(over, p);
    EXPECT_DOUBLE_EQ(clamped.thrust, p.thrust_max);
    EXPECT_DOUBLE_EQ(clamped.yaw_moment, p.moment_min);

    const Control within{5.0, 0.5};
    const Control unchanged = clamp_control(within, p);
    EXPECT_DOUBLE_EQ(unchanged.thrust, 5.0);
    EXPECT_DOUBLE_EQ(unchanged.yaw_moment, 0.5);
}
