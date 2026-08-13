#include "vessel_gnc/integrator.hpp"

#include "vessel_gnc/dynamics.hpp"

namespace vessel_gnc {

namespace {

// state + a * increment, component-wise.
State add_scaled(const State& state, const State& increment, double a) {
    return {state.x + a * increment.x, state.y + a * increment.y,
            state.psi + a * increment.psi, state.u + a * increment.u,
            state.v + a * increment.v, state.r + a * increment.r};
}

}  // namespace

State rk4_step(const State& state,
               const Control& control,
               const Environment& environment,
               const ModelParams& params,
               double dt) {
    const State k1 = derivative(state, control, environment, params);
    const State k2 = derivative(add_scaled(state, k1, dt / 2.0), control, environment, params);
    const State k3 = derivative(add_scaled(state, k2, dt / 2.0), control, environment, params);
    const State k4 = derivative(add_scaled(state, k3, dt), control, environment, params);

    State next;
    next.x = state.x + dt / 6.0 * (k1.x + 2.0 * k2.x + 2.0 * k3.x + k4.x);
    next.y = state.y + dt / 6.0 * (k1.y + 2.0 * k2.y + 2.0 * k3.y + k4.y);
    next.psi = state.psi + dt / 6.0 * (k1.psi + 2.0 * k2.psi + 2.0 * k3.psi + k4.psi);
    next.u = state.u + dt / 6.0 * (k1.u + 2.0 * k2.u + 2.0 * k3.u + k4.u);
    next.v = state.v + dt / 6.0 * (k1.v + 2.0 * k2.v + 2.0 * k3.v + k4.v);
    next.r = state.r + dt / 6.0 * (k1.r + 2.0 * k2.r + 2.0 * k3.r + k4.r);
    return next;
}

}  // namespace vessel_gnc
