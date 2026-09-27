"""Focused tests for the shared CasADi vessel prediction model."""

import casadi as ca
import numpy as np
import pytest
from vessel_gnc import _core
from vessel_gnc.nmpc import VesselNmpc
from vessel_gnc.prediction import build_prediction_step, environment_vector

PARAMS = _core.default_params()
DT = 0.4
SUBSTEPS = 2
ENVIRONMENT = _core.Environment(
    current_north=0.12,
    current_east=-0.08,
    wind_north=2.0,
    wind_east=-1.0,
)
DISTURBANCE = np.array([0.12, -0.08, 2.0, -1.0])


def _cpp_prediction_step(
    state_vector: np.ndarray,
    command_vector: np.ndarray,
) -> np.ndarray:
    """Propagate with the independent C++ actuator and vessel kernels."""
    vessel_state = _core.State(
        x=state_vector[0],
        y=state_vector[1],
        psi=state_vector[2],
        u=state_vector[3],
        v=state_vector[4],
        r=state_vector[5],
    )
    actuator_state = _core.ActuatorState(
        thrust=state_vector[6],
        yaw_moment=state_vector[7],
    )
    command = _core.Control(
        thrust=command_vector[0],
        yaw_moment=command_vector[1],
    )
    substep_dt = DT / SUBSTEPS
    for _ in range(SUBSTEPS):
        actuator_state = _core.actuator_step(
            actuator_state,
            command,
            PARAMS,
            substep_dt,
        )
        vessel_state = _core.rk4_step(
            vessel_state,
            _core.Control(
                thrust=actuator_state.thrust,
                yaw_moment=actuator_state.yaw_moment,
            ),
            ENVIRONMENT,
            PARAMS,
            substep_dt,
        )
    return np.array(
        [
            vessel_state.x,
            vessel_state.y,
            vessel_state.psi,
            vessel_state.u,
            vessel_state.v,
            vessel_state.r,
            actuator_state.thrust,
            actuator_state.yaw_moment,
        ]
    )


def _shared_prediction_step(
    state_vector: np.ndarray,
    command_vector: np.ndarray,
) -> np.ndarray:
    prediction_step = build_prediction_step(
        params=PARAMS,
        dt=DT,
        substeps=SUBSTEPS,
    )
    return np.array(
        prediction_step(
            ca.DM(state_vector),
            ca.DM(command_vector),
            ca.DM(DISTURBANCE),
        )
    ).ravel()


def test_environment_vector_preserves_order_and_exact_zero_case():
    zero = environment_vector(None)
    np.testing.assert_array_equal(zero, np.zeros(4))
    np.testing.assert_array_equal(environment_vector(_core.Environment()), zero)

    environment = _core.Environment(
        current_north=0.12,
        current_east=-0.08,
        wind_north=2.0,
        wind_east=-1.0,
    )
    np.testing.assert_array_equal(
        environment_vector(environment),
        [0.12, -0.08, 2.0, -1.0],
    )


@pytest.mark.parametrize(
    "component",
    ["current_north", "current_east", "wind_north", "wind_east"],
)
@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf])
def test_environment_vector_rejects_nonfinite_components(component, value):
    with pytest.raises(ValueError, match="finite"):
        environment_vector(_core.Environment(**{component: value}))


@pytest.mark.parametrize("dt", [0.0, -0.1, np.nan, np.inf, -np.inf])
def test_prediction_step_rejects_nonpositive_or_nonfinite_dt(dt):
    with pytest.raises(ValueError, match="finite and positive"):
        build_prediction_step(params=PARAMS, dt=dt, substeps=SUBSTEPS)


@pytest.mark.parametrize("substeps", [0, -1, 1.5, True])
def test_prediction_step_rejects_invalid_substeps(substeps):
    with pytest.raises(ValueError, match="integer"):
        build_prediction_step(params=PARAMS, dt=DT, substeps=substeps)


@pytest.mark.parametrize(
    "actuator_components",
    [
        (90.0, -10.0),
        (-50.0, 10.0),
    ],
)
def test_prediction_matches_cpp_for_initially_out_of_bound_actuator_state(
    actuator_components,
):
    state = np.array([2.0, -1.0, 0.4, 1.1, -0.2, 0.08, *actuator_components])
    command = np.array([30.0, -0.5])

    cpp_step = _cpp_prediction_step(state, command)
    casadi_step = _shared_prediction_step(state, command)

    assert PARAMS.thrust_min <= cpp_step[6] <= PARAMS.thrust_max
    assert PARAMS.moment_min <= cpp_step[7] <= PARAMS.moment_max
    np.testing.assert_allclose(casadi_step, cpp_step, rtol=0.0, atol=1e-10)


@pytest.mark.parametrize(
    "command",
    [
        np.array([1000.0, -1000.0]),
        np.array([-1000.0, 1000.0]),
    ],
)
def test_prediction_matches_cpp_for_out_of_range_commands(command):
    state = np.array([2.0, -1.0, 0.4, 1.1, -0.2, 0.08, 25.0, 0.3])

    cpp_step = _cpp_prediction_step(state, command)
    casadi_step = _shared_prediction_step(state, command)

    np.testing.assert_allclose(casadi_step, cpp_step, rtol=0.0, atol=1e-10)


def test_vessel_nmpc_delegates_model_step_to_shared_prediction():
    nmpc = VesselNmpc(PARAMS)
    prediction_step = build_prediction_step(
        params=PARAMS,
        dt=nmpc.config.dt,
        substeps=nmpc.config.substeps,
    )
    state = np.array([2.0, -1.0, 0.4, 1.1, -0.2, 0.08, 25.0, 0.3])
    command = np.array([30.0, -0.5])
    environments = (
        None,
        _core.Environment(
            current_north=0.12,
            current_east=-0.08,
            wind_north=2.0,
            wind_east=-1.0,
        ),
    )

    for environment in environments:
        disturbance = environment_vector(environment)
        shared_step = np.array(
            prediction_step(
                ca.DM(state),
                ca.DM(command),
                ca.DM(disturbance),
            )
        ).ravel()
        np.testing.assert_array_equal(
            nmpc.model_step(state, command, environment),
            shared_step,
        )
