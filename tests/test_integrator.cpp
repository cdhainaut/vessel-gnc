// Integration tests: RK4 behaviour and the plan validation cases A-D at the
// trajectory level. See docs/model.md §8.

#include <gtest/gtest.h>

#include <array>
#include <cmath>

#include <Eigen/Core>

#include "vessel_gnc/dynamics.hpp"
#include "vessel_gnc/integrator.hpp"

using namespace vessel_gnc;

TEST(RK4, ZeroForceZeroDampingInertialMotion) {
    // Plan case A through integration: 10 s of unforced, undamped pure-surge
    // motion keeps the velocity constant and moves in a straight line
    // (the Coriolis term vanishes for v = r = 0).
    ModelParams p = default_params();
    p.lin_damping_u = 0.0;
    p.lin_damping_v = 0.0;
    p.lin_damping_r = 0.0;
    p.quad_damping_u = 0.0;
    p.quad_damping_v = 0.0;
    p.quad_damping_r = 0.0;

    State s;
    s.psi = 0.4;
    s.u = 1.2;
    constexpr double dt = 0.01;
    for (int k = 0; k < 1000; ++k) {
        s = rk4_step(s, Control{}, Environment{}, p, dt);
    }
    EXPECT_NEAR(s.u, 1.2, 1e-12);
    EXPECT_NEAR(s.v, 0.0, 1e-12);
    EXPECT_NEAR(s.r, 0.0, 1e-12);
    EXPECT_NEAR(s.psi, 0.4, 1e-12);
    EXPECT_NEAR(s.x, 1.2 * std::cos(0.4) * 10.0, 1e-9);
    EXPECT_NEAR(s.y, 1.2 * std::sin(0.4) * 10.0, 1e-9);
}

TEST(RK4, ZeroForceCoriolisConservesEnergy) {
    // With mixed velocities the undamped, unforced system exchanges energy
    // between axes through C(nu), but the kinetic energy is conserved:
    // nu^T M nu_dot = -nu^T C(nu) nu = 0 (validated at integration level).
    ModelParams p = default_params();
    p.lin_damping_u = 0.0;
    p.lin_damping_v = 0.0;
    p.lin_damping_r = 0.0;
    p.quad_damping_u = 0.0;
    p.quad_damping_v = 0.0;
    p.quad_damping_r = 0.0;
    const double m11 = p.mass + p.added_mass_x;
    const double m22 = p.mass + p.added_mass_y;
    const double m33 = p.inertia_z + p.added_inertia_z;

    State s;
    s.u = 1.2;
    s.v = -0.4;
    s.r = 0.3;
    const double ke0 = 0.5 * (m11 * s.u * s.u + m22 * s.v * s.v + m33 * s.r * s.r);
    constexpr double dt = 0.01;
    for (int k = 0; k < 1000; ++k) {
        s = rk4_step(s, Control{}, Environment{}, p, dt);
    }
    const double ke = 0.5 * (m11 * s.u * s.u + m22 * s.v * s.v + m33 * s.r * s.r);
    EXPECT_NEAR(ke, ke0, 1e-6);
    EXPECT_TRUE(std::isfinite(s.u));
    EXPECT_TRUE(std::isfinite(s.v));
    EXPECT_TRUE(std::isfinite(s.r));
}

TEST(RK4, ConvergenceRateAgainstAnalyticalSurge) {
    // Plan case D: with linear damping only, the surge ODE has a closed-form
    // solution. RK4's global error is O(dt^4): halving dt divides the error
    // by ~16.
    const double thrust = 5.0;
    const double tf = 20.0;
    ModelParams p = default_params();
    p.quad_damping_u = 0.0;
    p.quad_damping_v = 0.0;
    p.quad_damping_r = 0.0;

    const double m11 = p.mass + p.added_mass_x;
    const double a = -p.lin_damping_u / m11;       // u_dot = a (u - u_eq)
    const double u_eq = thrust / p.lin_damping_u;  // [m/s]
    const double u_exact = u_eq * (1.0 - std::exp(a * tf));
    const double x_exact = u_eq * (tf - (std::exp(a * tf) - 1.0) / a);

    const Control control{thrust};
    const std::array<double, 3> dt{0.5, 0.25, 0.125};
    std::array<double, 3> err{};
    for (size_t i = 0; i < dt.size(); ++i) {
        State s;
        const int steps = static_cast<int>(tf / dt[i]);
        for (int k = 0; k < steps; ++k) {
            s = rk4_step(s, control, Environment{}, p, dt[i]);
        }
        err[i] = std::hypot(s.u - u_exact, s.x - x_exact);
    }
    EXPECT_NEAR(err[0] / err[1], 16.0, 4.0);
    EXPECT_NEAR(err[1] / err[2], 16.0, 4.0);
}

TEST(RK4, ReachesSurgeEquilibrium) {
    // Plan case B through integration: constant thrust drives the vessel to
    // the drag-balance speed, then it cruises at constant velocity.
    ModelParams p = default_params();
    const double thrust = 40.0;
    const double a = p.quad_damping_u;
    const double b = p.lin_damping_u;
    const double u_eq = (-b + std::sqrt(b * b + 4.0 * a * thrust)) / (2.0 * a);

    State s;
    const Control control{thrust};
    constexpr double dt = 0.01;
    for (int k = 0; k < 6000; ++k) {
        s = rk4_step(s, control, Environment{}, p, dt);
    }
    EXPECT_NEAR(s.u, u_eq, 0.01);
    EXPECT_NEAR(s.v, 0.0, 1e-6);
    EXPECT_NEAR(s.psi, 0.0, 1e-6);
    // 60 s at ~1.36 m/s minus the start-up transient.
    EXPECT_GT(s.x, 70.0);
    EXPECT_LT(s.x, 90.0);
}

TEST(RK4, SteadyTurnUnderConstantMoment) {
    // Plan case C through integration: constant thrust + yaw moment produce a
    // steady turn. The yaw rate settles near the coupled drag/Munk balance
    // (r ~ 0.072 rad/s, docs/model.md §6) and the vessel develops a port
    // sideslip (v < 0, bow into the turn).
    ModelParams p = default_params();
    State s;
    const Control control{40.0, 2.0};
    constexpr double dt = 0.01;
    double distance = 0.0;
    for (int k = 0; k < 6000; ++k) {
        const State prev = s;
        s = rk4_step(s, control, Environment{}, p, dt);
        distance += std::hypot(s.x - prev.x, s.y - prev.y);
    }
    EXPECT_NEAR(s.r, 0.072, 0.02);
    EXPECT_LT(s.v, -0.01);           // port sideslip, bow into the turn
    EXPECT_GT(s.psi, 3.0);           // more than half a turn in 60 s
    EXPECT_GT(s.y, 10.0);            // turned East
    EXPECT_GT(distance, 70.0);       // ~80 m at cruise speed
}

TEST(RK4, CurrentEquilibrium) {
    // A vessel moving exactly with the ambient current stays in equilibrium.
    ModelParams p = default_params();
    Environment env;
    env.current_east = 0.5;
    State s;
    s.v = 0.5;  // aligned with the current at psi = 0
    const State d = derivative(s, Control{}, env, p);
    EXPECT_NEAR(d.u, 0.0, 1e-9);
    EXPECT_NEAR(d.v, 0.0, 1e-9);
    // ... and RK4 keeps it there.
    for (int k = 0; k < 1000; ++k) {
        s = rk4_step(s, Control{}, env, p, 0.01);
    }
    EXPECT_NEAR(s.u, 0.0, 1e-9);
    EXPECT_NEAR(s.v, 0.5, 1e-9);
}

TEST(RK4, TurningBodyPreservesConstantInertialCurrentDrift) {
    // A co-moving vessel with a constant yaw rate has zero relative
    // translational velocity. Its body components rotate, while its inertial
    // velocity and straight drift remain constant.
    ModelParams p = default_params();
    p.lin_damping_r = 0.0;
    p.quad_damping_r = 0.0;
    Environment env;
    env.current_north = 0.4;
    env.current_east = -0.2;
    State s;
    s.u = env.current_north;
    s.v = env.current_east;
    s.r = 0.2;

    constexpr double dt = 0.01;
    constexpr int n_steps = 500;
    for (int k = 0; k < n_steps; ++k) {
        s = rk4_step(s, Control{}, env, p, dt);
    }

    const double duration = n_steps * dt;
    const Eigen::Vector2d expected_body =
        rotation_matrix(s.psi).transpose()
        * Eigen::Vector2d(env.current_north, env.current_east);
    EXPECT_NEAR(s.x, env.current_north * duration, 1e-9);
    EXPECT_NEAR(s.y, env.current_east * duration, 1e-9);
    EXPECT_NEAR(s.psi, 0.2 * duration, 1e-12);
    EXPECT_NEAR(s.u, expected_body.x(), 1e-9);
    EXPECT_NEAR(s.v, expected_body.y(), 1e-9);
    EXPECT_NEAR(s.r, 0.2, 1e-12);
}

TEST(RK4, StaysFiniteOverLongRun) {
    // No NaN/Inf over a long deterministic run (forced, damped, with current).
    ModelParams p = default_params();
    Environment env;
    env.current_east = 0.3;
    State s;
    const Control control{40.0, 1.0};
    constexpr double dt = 0.01;
    for (int k = 0; k < 30000; ++k) {  // 300 s
        s = rk4_step(s, control, env, p, dt);
    }
    EXPECT_TRUE(std::isfinite(s.x));
    EXPECT_TRUE(std::isfinite(s.y));
    EXPECT_TRUE(std::isfinite(s.psi));
    EXPECT_TRUE(std::isfinite(s.u));
    EXPECT_TRUE(std::isfinite(s.v));
    EXPECT_TRUE(std::isfinite(s.r));
}

TEST(RK4, DeterministicRepetition) {
    // The kernel is a pure function of its inputs: same inputs, bit-identical
    // outputs (the Python layer is responsible for seeding any randomness).
    ModelParams p = default_params();
    Environment env;
    env.current_east = 0.3;
    const Control control{40.0, 1.0};
    const auto run = [&]() {
        State s;
        for (int k = 0; k < 1000; ++k) {
            s = rk4_step(s, control, env, p, 0.01);
        }
        return s;
    };
    const State a = run();
    const State b = run();
    EXPECT_EQ(a.x, b.x);
    EXPECT_EQ(a.y, b.y);
    EXPECT_EQ(a.psi, b.psi);
    EXPECT_EQ(a.u, b.u);
    EXPECT_EQ(a.v, b.v);
    EXPECT_EQ(a.r, b.r);
}
