#include <format>
#include <string>

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>

#include "vessel_gnc/actuator.hpp"
#include "vessel_gnc/controllers.hpp"
#include "vessel_gnc/dynamics.hpp"
#include "vessel_gnc/integrator.hpp"

namespace py = pybind11;
using namespace vessel_gnc;

namespace {

std::string repr_state(const State& s) {
    return std::format(
        "State(x={:.3f}, y={:.3f}, psi={:.3f}, u={:.3f}, v={:.3f}, r={:.3f})",
        s.x, s.y, s.psi, s.u, s.v, s.r);
}

std::string repr_control(const Control& c) {
    return std::format("Control(thrust={:.3f}, yaw_moment={:.3f})", c.thrust, c.yaw_moment);
}

std::string repr_environment(const Environment& e) {
    return std::format(
        "Environment(current_north={:.3f}, current_east={:.3f}, "
        "wind_north={:.3f}, wind_east={:.3f})",
        e.current_north, e.current_east, e.wind_north, e.wind_east);
}

}  // namespace

PYBIND11_MODULE(_core, m) {
    m.doc() = "C++ core of vessel-gnc: 3-DOF vessel dynamics and RK4 integration.";

    py::class_<State>(m, "State")
        .def(py::init([](double x, double y, double psi, double u, double v, double r) {
            return State{x, y, psi, u, v, r};
        }),
             py::arg("x") = 0.0, py::arg("y") = 0.0, py::arg("psi") = 0.0,
             py::arg("u") = 0.0, py::arg("v") = 0.0, py::arg("r") = 0.0)
        .def_readwrite("x", &State::x)
        .def_readwrite("y", &State::y)
        .def_readwrite("psi", &State::psi)
        .def_readwrite("u", &State::u)
        .def_readwrite("v", &State::v)
        .def_readwrite("r", &State::r)
        .def("__repr__", &repr_state);

    py::class_<Control>(m, "Control")
        .def(py::init([](double thrust, double yaw_moment) {
            return Control{thrust, yaw_moment};
        }),
             py::arg("thrust") = 0.0, py::arg("yaw_moment") = 0.0)
        .def_readwrite("thrust", &Control::thrust)
        .def_readwrite("yaw_moment", &Control::yaw_moment)
        .def("__repr__", &repr_control);

    py::class_<Environment>(m, "Environment")
        .def(py::init([](double current_north, double current_east,
                         double wind_north, double wind_east) {
            return Environment{current_north, current_east, wind_north, wind_east};
        }),
             py::arg("current_north") = 0.0, py::arg("current_east") = 0.0,
             py::arg("wind_north") = 0.0, py::arg("wind_east") = 0.0)
        .def_readwrite("current_north", &Environment::current_north)
        .def_readwrite("current_east", &Environment::current_east)
        .def_readwrite("wind_north", &Environment::wind_north)
        .def_readwrite("wind_east", &Environment::wind_east)
        .def("__repr__", &repr_environment);

    py::class_<ModelParams>(m, "ModelParams")
        .def(py::init([](double mass, double inertia_z, double added_mass_x,
                         double added_mass_y, double added_inertia_z,
                         double lin_damping_u, double lin_damping_v, double lin_damping_r,
                         double quad_damping_u, double quad_damping_v, double quad_damping_r,
                         double thrust_min, double thrust_max,
                         double moment_min, double moment_max,
                         double thrust_time_constant, double moment_time_constant,
                         double thrust_rate_limit, double moment_rate_limit) {
            ModelParams p;
            p.mass = mass;
            p.inertia_z = inertia_z;
            p.added_mass_x = added_mass_x;
            p.added_mass_y = added_mass_y;
            p.added_inertia_z = added_inertia_z;
            p.lin_damping_u = lin_damping_u;
            p.lin_damping_v = lin_damping_v;
            p.lin_damping_r = lin_damping_r;
            p.quad_damping_u = quad_damping_u;
            p.quad_damping_v = quad_damping_v;
            p.quad_damping_r = quad_damping_r;
            p.thrust_min = thrust_min;
            p.thrust_max = thrust_max;
            p.moment_min = moment_min;
            p.moment_max = moment_max;
            p.thrust_time_constant = thrust_time_constant;
            p.moment_time_constant = moment_time_constant;
            p.thrust_rate_limit = thrust_rate_limit;
            p.moment_rate_limit = moment_rate_limit;
            return p;
        }),
             py::arg("mass") = 30.0, py::arg("inertia_z") = 4.0,
             py::arg("added_mass_x") = 5.0, py::arg("added_mass_y") = 20.0,
             py::arg("added_inertia_z") = 2.0,
             py::arg("lin_damping_u") = 2.0, py::arg("lin_damping_v") = 150.0,
             py::arg("lin_damping_r") = 30.0,
             py::arg("quad_damping_u") = 20.0, py::arg("quad_damping_v") = 250.0,
             py::arg("quad_damping_r") = 60.0,
             py::arg("thrust_min") = -20.0, py::arg("thrust_max") = 60.0,
             py::arg("moment_min") = -6.0, py::arg("moment_max") = 6.0,
             py::arg("thrust_time_constant") = 1.0,
             py::arg("moment_time_constant") = 0.6,
             py::arg("thrust_rate_limit") = 50.0,
             py::arg("moment_rate_limit") = 8.0)
        .def_readwrite("mass", &ModelParams::mass)
        .def_readwrite("inertia_z", &ModelParams::inertia_z)
        .def_readwrite("added_mass_x", &ModelParams::added_mass_x)
        .def_readwrite("added_mass_y", &ModelParams::added_mass_y)
        .def_readwrite("added_inertia_z", &ModelParams::added_inertia_z)
        .def_readwrite("lin_damping_u", &ModelParams::lin_damping_u)
        .def_readwrite("lin_damping_v", &ModelParams::lin_damping_v)
        .def_readwrite("lin_damping_r", &ModelParams::lin_damping_r)
        .def_readwrite("quad_damping_u", &ModelParams::quad_damping_u)
        .def_readwrite("quad_damping_v", &ModelParams::quad_damping_v)
        .def_readwrite("quad_damping_r", &ModelParams::quad_damping_r)
        .def_readwrite("thrust_min", &ModelParams::thrust_min)
        .def_readwrite("thrust_max", &ModelParams::thrust_max)
        .def_readwrite("moment_min", &ModelParams::moment_min)
        .def_readwrite("moment_max", &ModelParams::moment_max)
        .def_readwrite("thrust_time_constant", &ModelParams::thrust_time_constant)
        .def_readwrite("moment_time_constant", &ModelParams::moment_time_constant)
        .def_readwrite("thrust_rate_limit", &ModelParams::thrust_rate_limit)
        .def_readwrite("moment_rate_limit", &ModelParams::moment_rate_limit);

    m.def("default_params", &default_params, "Illustrative small-USV parameter set.");
    m.def("rotation_matrix", &rotation_matrix, py::arg("psi"),
          "Body-to-inertial rotation matrix R(psi).");
    m.def("derivative", &derivative, py::arg("state"), py::arg("control"),
          py::arg("environment"), py::arg("params"),
          "Time derivative of the vessel state (docs/model.md §2-§4).");
    m.def("rk4_step", &rk4_step, py::arg("state"), py::arg("control"),
          py::arg("environment"), py::arg("params"), py::arg("dt"),
          "One fixed-step RK4 integration step (zero-order hold on control).");
    m.def("clamp_control", &clamp_control, py::arg("control"), py::arg("params"),
          "Clamp a control command to the actuator saturation bounds.");

    py::class_<ActuatorState>(m, "ActuatorState")
        .def(py::init([](double thrust, double yaw_moment) {
            return ActuatorState{thrust, yaw_moment};
        }),
             py::arg("thrust") = 0.0, py::arg("yaw_moment") = 0.0)
        .def_readwrite("thrust", &ActuatorState::thrust)
        .def_readwrite("yaw_moment", &ActuatorState::yaw_moment)
        .def("__repr__", [](const ActuatorState& a) {
            return std::format("ActuatorState(thrust={:.3f}, yaw_moment={:.3f})",
                               a.thrust, a.yaw_moment);
        });

    m.def("actuator_step", &actuator_step, py::arg("actuator"), py::arg("command"),
          py::arg("params"), py::arg("dt"),
          "One RK4 step of the rate-limited first-order actuator response.");
    m.def("truth_params", &truth_params,
          "Perturbed plant parameters for the model-mismatch scenario.");

    py::class_<PidGains>(m, "PidGains")
        .def(py::init([](double kp, double ki, double kd) {
            return PidGains{kp, ki, kd};
        }),
             py::arg("kp") = 0.0, py::arg("ki") = 0.0, py::arg("kd") = 0.0)
        .def_readwrite("kp", &PidGains::kp)
        .def_readwrite("ki", &PidGains::ki)
        .def_readwrite("kd", &PidGains::kd);

    py::class_<HeadingController>(m, "HeadingController")
        .def(py::init<const PidGains&, double, double>(), py::arg("gains"),
             py::arg("moment_limit") = 6.0, py::arg("integrator_limit") = 1.0,
             "PID heading controller (docs/control.md §2).")
        .def("reset", &HeadingController::reset)
        .def("update", &HeadingController::update, py::arg("heading_ref"),
             py::arg("heading"), py::arg("yaw_rate"), py::arg("dt"),
             "One control step; returns the yaw moment [N m] (clamped).");

    py::class_<SpeedController>(m, "SpeedController")
        .def(py::init<const PidGains&, double, double>(), py::arg("gains"),
             py::arg("thrust_limit") = 40.0, py::arg("integrator_limit") = 45.0,
             "PI surge-speed controller (docs/control.md §2).")
        .def("reset", &SpeedController::reset)
        .def("update", &SpeedController::update, py::arg("speed_ref"),
             py::arg("surge_speed"), py::arg("dt"),
             "One control step; returns the surge thrust [N] (clamped).");

    m.def("wrap_to_pi", &wrap_to_pi, py::arg("angle"),
          "Wrap an angle to (-pi, pi].");
    m.def("default_heading_gains", &default_heading_gains,
          "Heading gains tuned for the default vessel.");
    m.def("default_speed_gains", &default_speed_gains,
          "Speed gains tuned for the default vessel.");
}
