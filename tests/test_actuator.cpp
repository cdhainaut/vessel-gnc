// Unit tests for the actuator model: first-order response lag, rate limits
// and saturation (docs/model.md §5).

#include <gtest/gtest.h>

#include <cmath>

#include "vessel_gnc/actuator.hpp"
#include "vessel_gnc/dynamics.hpp"

using namespace vessel_gnc;

namespace {
constexpr double kTol = 1e-6;
}

TEST(Actuator, RestStateStaysAtRest) {
    const ModelParams params = default_params();
    ActuatorState actuator;
    for (int k = 0; k < 1000; ++k) {
        actuator = actuator_step(actuator, Control{}, params, 0.01);
    }
    EXPECT_NEAR(actuator.thrust, 0.0, kTol);
    EXPECT_NEAR(actuator.yaw_moment, 0.0, kTol);
}

TEST(Actuator, MatchesAnalyticFirstOrderLag) {
    // Without active rate limits the response is the exponential approach
    // T(t) = T_cmd (1 - exp(-t / tau)); RK4 matches it closely.
    ModelParams params = default_params();
    params.thrust_rate_limit = 1e6;  // never active
    params.moment_rate_limit = 1e6;
    const double tau_t = params.thrust_time_constant;
    const double tau_m = params.moment_time_constant;
    const Control command{40.0, 2.0};

    ActuatorState actuator;
    constexpr double dt = 0.01;
    for (int k = 0; k < 3000; ++k) {  // 30 s
        actuator = actuator_step(actuator, command, params, dt);
        const double t = (k + 1) * dt;
        EXPECT_NEAR(actuator.thrust, 40.0 * (1.0 - std::exp(-t / tau_t)), 1e-6);
        EXPECT_NEAR(actuator.yaw_moment, 2.0 * (1.0 - std::exp(-t / tau_m)), 1e-6);
    }
}

TEST(Actuator, SmoothRateLimitProducesSaturatingResponse) {
    // A large step command saturates the smooth rate limit: the response is
    // the exact solution of dT/dt = L tanh((T_cmd - T) / (tau L)), namely
    //     T(t) = T_cmd - tau L asinh(sinh((T_cmd - T0)/(tau L)) exp(-t/tau))
    // and the achieved rate never exceeds the limit.
    ModelParams params = default_params();
    const double rate = params.thrust_rate_limit;  // 80 N/s
    const double tau = params.thrust_time_constant;
    constexpr double dt = 0.001;
    const double target = 60.0;
    const double u0 = target / (tau * rate);  // 2.5

    ActuatorState actuator;
    double previous_thrust = 0.0;
    for (int k = 0; k < 15000; ++k) {  // 15 s
        actuator = actuator_step(actuator, Control{target, 0.0}, params, dt);
        const double t = (k + 1) * dt;
        const double expected =
            target - tau * rate * std::asinh(std::sinh(u0) * std::exp(-t / tau));
        EXPECT_NEAR(actuator.thrust, expected, 1e-6);
        // Achieved rate stays within the limit.
        EXPECT_LE((actuator.thrust - previous_thrust) / dt, rate + 1e-6);
        previous_thrust = actuator.thrust;
    }
    EXPECT_NEAR(actuator.thrust, target, 1e-3);
}

TEST(Actuator, SaturationBoundsTheState) {
    // Commands beyond the limits: the applied force approaches the bound and
    // never exceeds it.
    ModelParams params = default_params();
    ActuatorState actuator;
    const Control over{1000.0, -1000.0};
    for (int k = 0; k < 20000; ++k) {  // 200 s
        actuator = actuator_step(actuator, over, params, 0.01);
        EXPECT_LE(actuator.thrust, params.thrust_max + 1e-12);
        EXPECT_GE(actuator.yaw_moment, params.moment_min - 1e-12);
    }
    EXPECT_GT(actuator.thrust, params.thrust_max - 1e-3);
    EXPECT_LT(actuator.yaw_moment, params.moment_min + 1e-3);
}

TEST(Actuator, DeterministicRepetition) {
    const ModelParams params = default_params();
    const Control command{50.0, 3.0};
    const auto run = [&]() {
        ActuatorState actuator;
        for (int k = 0; k < 1000; ++k) {
            actuator = actuator_step(actuator, command, params, 0.01);
        }
        return actuator;
    };
    const ActuatorState a = run();
    const ActuatorState b = run();
    EXPECT_EQ(a.thrust, b.thrust);
    EXPECT_EQ(a.yaw_moment, b.yaw_moment);
}

TEST(TruthParams, PerturbedButStable) {
    // The truth parameter set differs from the nominal set...
    const ModelParams truth = truth_params();
    const ModelParams nominal = default_params();
    EXPECT_NE(truth.mass, nominal.mass);
    EXPECT_NE(truth.lin_damping_r, nominal.lin_damping_r);
    // ... and keeps the vessel directionally stable at cruise: the effective
    // yaw damping must exceed the Munk coupling (docs/model.md §6).
    const double m11 = truth.mass + truth.added_mass_x;
    const double m22 = truth.mass + truth.added_mass_y;
    const double u_cruise = 1.3;  // [m/s]
    const double munk_coupling = (m22 - m11) * u_cruise * (m11 * u_cruise / truth.lin_damping_v);
    EXPECT_LT(munk_coupling, truth.lin_damping_r);
}
