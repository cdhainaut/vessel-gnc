#include "vessel_gnc/dynamics.hpp"

#include <algorithm>
#include <cmath>

#include <Eigen/Core>

namespace vessel_gnc {

Eigen::Matrix2d rotation_matrix(double psi) {
    const double c = std::cos(psi);
    const double s = std::sin(psi);
    Eigen::Matrix2d R;
    R << c, -s,
         s,  c;
    return R;
}

State derivative(const State& state,
                 const Control& control,
                 const Environment& environment,
                 const ModelParams& params) {
    const Eigen::Matrix2d R = rotation_matrix(state.psi);

    // Relative velocity in the body frame: subtract the ambient current
    // (assumed irrotational, so the yaw rate is unaffected).
    const Eigen::Vector2d current_body =
        R.transpose() * Eigen::Vector2d(environment.current_north, environment.current_east);
    const double u_rel = state.u - current_body.x();
    const double v_rel = state.v - current_body.y();
    const double r_rel = state.r;

    // Diagonal mass matrix M = diag(m11, m22, m33).
    const double m11 = params.mass + params.added_mass_x;
    const double m22 = params.mass + params.added_mass_y;
    const double m33 = params.inertia_z + params.added_inertia_z;

    // Coriolis/centripetal term for diagonal M (skew-symmetric):
    //     C(nu_rel) nu_rel = (-m22 v_rel r, m11 u_rel r, (m22-m11) u_rel v_rel)^T
    // The yaw component is the (destabilizing) Munk coupling; see docs/model.md §3.
    const double cx = -m22 * v_rel * r_rel;
    const double cy = m11 * u_rel * r_rel;
    const double cr = (m22 - m11) * u_rel * v_rel;

    // Damping, linear + quadratic, acting on the relative velocity.
    const double du =
        params.lin_damping_u * u_rel + params.quad_damping_u * std::abs(u_rel) * u_rel;
    const double dv =
        params.lin_damping_v * v_rel + params.quad_damping_v * std::abs(v_rel) * v_rel;
    const double dr =
        params.lin_damping_r * r_rel + params.quad_damping_r * std::abs(r_rel) * r_rel;

    // Actuator force (body frame) and wind force (inertial -> body frame).
    const double tau_x = control.thrust;
    const double tau_r = control.yaw_moment;
    const Eigen::Vector2d wind_body =
        R.transpose() * Eigen::Vector2d(environment.wind_north, environment.wind_east);

    // eta_dot = R(psi) nu
    State d;
    d.x = R(0, 0) * state.u + R(0, 1) * state.v;  // x_dot = u cos(psi) - v sin(psi)
    d.y = R(1, 0) * state.u + R(1, 1) * state.v;  // y_dot = u sin(psi) + v cos(psi)
    d.psi = state.r;

    // M nu_dot = tau + tau_env - C(nu_rel) nu_rel - D(nu_rel) nu_rel
    d.u = (tau_x + wind_body.x() - cx - du) / m11;
    d.v = (wind_body.y() - cy - dv) / m22;
    d.r = (tau_r - cr - dr) / m33;
    return d;
}

Control clamp_control(const Control& control, const ModelParams& params) {
    Control out;
    out.thrust = std::clamp(control.thrust, params.thrust_min, params.thrust_max);
    out.yaw_moment = std::clamp(control.yaw_moment, params.moment_min, params.moment_max);
    return out;
}

ModelParams default_params() {
    // Illustrative values for a small (~1.5 m, 30 kg) USV. Order-of-magnitude
    // only; not identified from a specific hull. See docs/model.md §6.
    return ModelParams{};
}

}  // namespace vessel_gnc
