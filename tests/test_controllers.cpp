// Unit and closed-loop tests for the baseline controllers.
// See docs/control.md for the formulations and the validation record.

#include <gtest/gtest.h>

#include <cmath>

#include "vessel_gnc/controllers.hpp"
#include "vessel_gnc/dynamics.hpp"
#include "vessel_gnc/integrator.hpp"

using namespace vessel_gnc;

namespace {
constexpr double kTol = 1e-9;
}

TEST(WrapToPi, WrapsAcrossBoundaries) {
    EXPECT_NEAR(wrap_to_pi(0.0), 0.0, kTol);
    EXPECT_NEAR(wrap_to_pi(M_PI), M_PI, kTol);
    EXPECT_NEAR(wrap_to_pi(3.0 * M_PI / 2.0), -M_PI / 2.0, kTol);
    EXPECT_NEAR(wrap_to_pi(-3.0 * M_PI / 2.0), M_PI / 2.0, kTol);
    EXPECT_NEAR(wrap_to_pi(2.0 * M_PI + 0.3), 0.3, kTol);
    EXPECT_NEAR(wrap_to_pi(-2.0 * M_PI - 0.3), -0.3, kTol);
    // Any wrapped value is in (-pi, pi].
    for (double a : {10.0, -10.0, 100.0, -100.0}) {
        const double w = wrap_to_pi(a);
        EXPECT_GT(w, -M_PI);
        EXPECT_LE(w, M_PI);
    }
}

TEST(HeadingController, ZeroErrorZeroOutput) {
    HeadingController ctrl(default_heading_gains(), 6.0, 1.0);
    EXPECT_NEAR(ctrl.update(0.5, 0.5, 0.0, 0.1), 0.0, kTol);
    EXPECT_NEAR(ctrl.update(-1.0, -1.0 + 2.0 * M_PI, 0.0, 0.1), 0.0, kTol);  // wrapped
}

TEST(HeadingController, DerivativeActsOnYawRate) {
    // With zero heading error, the output is -kd * yaw_rate.
    HeadingController ctrl(PidGains{12.0, 0.1, 0.5}, 6.0, 1.0);
    const double out = ctrl.update(1.0, 1.0, 0.2, 0.1);
    EXPECT_NEAR(out, -0.5 * 0.2, kTol);
}

TEST(HeadingController, OutputSaturates) {
    HeadingController ctrl(default_heading_gains(), 6.0, 1.0);
    for (int k = 0; k < 10; ++k) {
        const double out = ctrl.update(M_PI, 0.0, 0.0, 0.1);  // 180 deg error
        EXPECT_LE(std::abs(out), 6.0 + 1e-12);
    }
}

TEST(HeadingController, AntiWindupBoundsIntegral) {
    HeadingController ctrl(default_heading_gains(), 6.0, 1.0);
    // Sustained large error for a long time: the output stays at the limit and
    // the integrator cannot wind up beyond its bound.
    for (int k = 0; k < 1000; ++k) {
        const double out = ctrl.update(2.0, 0.0, 0.0, 0.01);
        EXPECT_LE(std::abs(out), 6.0 + 1e-12);
    }
    // Once the error is released, the controller responds immediately
    // (a wound-up integral would cause a large transient).
    const double out = ctrl.update(2.0, 2.0, 0.0, 0.01);
    EXPECT_NEAR(out, 0.0, 1e-9);
}

TEST(SpeedController, AntiWindupBoundsIntegral) {
    SpeedController ctrl(default_speed_gains(), 40.0, 45.0);
    for (int k = 0; k < 1000; ++k) {
        const double out = ctrl.update(1.3, 0.0, 0.01);
        EXPECT_LE(std::abs(out), 40.0 + 1e-12);
    }
    // On release the integral holds the trim value where saturation began
    // (well below the limit: no windup, no large transient).
    const double out = ctrl.update(1.3, 1.3, 0.01);
    EXPECT_LT(std::abs(out), 10.0);
}

TEST(ClosedLoop, HeadingAndSpeedTrackReferences) {
    // Full closed loop over the C++ dynamics: a 90 deg heading step is
    // tracked and the surge speed converges to the reference.
    ModelParams p = default_params();
    HeadingController heading(default_heading_gains(), 6.0, 1.0);
    SpeedController speed(default_speed_gains(), 40.0, 45.0);
    State s;
    const double psi_ref = M_PI / 2.0;
    const double u_ref = 1.3;
    constexpr double dt = 0.01;
    double thrust = 0.0;
    double moment = 0.0;
    for (int k = 0; k < 3000; ++k) {  // 30 s
        if (k % 10 == 0) {            // 10 Hz control
            moment = heading.update(psi_ref, s.psi, s.r, 0.1);
            thrust = speed.update(u_ref, s.u, 0.1);
        }
        s = rk4_step(s, Control{thrust, moment}, Environment{}, p, dt);
    }
    EXPECT_NEAR(wrap_to_pi(s.psi - psi_ref), 0.0, 0.05);
    EXPECT_NEAR(s.u, u_ref, 0.05);
    EXPECT_LT(std::abs(s.r), 0.05);
}

TEST(ClosedLoop, HeadingHoldUnderCrossCurrent) {
    // The heading loop holds the reference heading against a cross-current:
    // steady-state heading error stays small (the vessel crabs).
    ModelParams p = default_params();
    HeadingController heading(default_heading_gains(), 6.0, 1.0);
    SpeedController speed(default_speed_gains(), 40.0, 45.0);
    Environment env;
    env.current_east = 0.3;  // [m/s]
    State s;
    const double psi_ref = 0.4;
    const double u_ref = 1.0;
    constexpr double dt = 0.01;
    double thrust = 0.0;
    double moment = 0.0;
    for (int k = 0; k < 6000; ++k) {  // 60 s
        if (k % 10 == 0) {
            moment = heading.update(psi_ref, s.psi, s.r, 0.1);
            thrust = speed.update(u_ref, s.u, 0.1);
        }
        s = rk4_step(s, Control{thrust, moment}, env, p, dt);
    }
    EXPECT_NEAR(wrap_to_pi(s.psi - psi_ref), 0.0, 0.02);
    EXPECT_NEAR(s.u, u_ref, 0.05);
}
