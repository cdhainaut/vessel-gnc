#pragma once

#include <Eigen/Core>

#include "vessel_gnc/state.hpp"

namespace vessel_gnc {

// Body-to-inertial rotation matrix (NED convention, psi clockwise from North):
//     eta_dot = R(psi) nu.
Eigen::Matrix2d rotation_matrix(double psi);

// Time derivative of the full vessel state for the 3-DOF model
// (docs/model.md §2–§4):
//     eta_dot = R(psi) nu
//     M nu_dot = tau + tau_env - C(nu_rel) nu_rel - D(nu_rel) nu_rel
// Assumes finite inputs; no NaN/Inf checking inside the kernel.
State derivative(const State& state,
                 const Control& control,
                 const Environment& environment,
                 const ModelParams& params);

// Clamp a control command to the actuator saturation bounds.
Control clamp_control(const Control& control, const ModelParams& params);

// Illustrative small-USV parameter set (docs/model.md §6).
ModelParams default_params();

// Perturbed plant parameters for the model-mismatch scenario (portfolio plan
// Phase C, docs/model.md §6): the controller/filter/NMPC keep
// `default_params()`, the plant uses this set. Multipliers are documented
// and the set keeps the vessel directionally stable at cruise.
ModelParams truth_params();

}  // namespace vessel_gnc
