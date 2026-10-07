from __future__ import annotations

import math

import numpy as np
import pytest

from parcacik_hareketi_simulatoru import analytic
from parcacik_hareketi_simulatoru.fields import uniform, zero
from parcacik_hareketi_simulatoru.integrators import boris_integrate, rk4_integrate
from parcacik_hareketi_simulatoru.particles import (
    C,
    ELECTRON,
    PROTON,
    gamma_from_u,
    gyrofrequency_rad_s,
    gyroradius_m,
)

B = 1.0
U0 = (0.4 * C, 0.0, 0.3 * C)  # relativistic: gamma ~ 1.12
OMEGA = abs(gyrofrequency_rad_s(ELECTRON, B, gamma_from_u(*U0)))
PERIOD = 2.0 * math.pi / OMEGA
RADIUS = gyroradius_m(ELECTRON, U0[0], B)


def _run(integrator, steps_per_period, periods, record_every=1, particle=ELECTRON, u0=U0):
    return integrator(
        particle, zero(), uniform(0, 0, B), (0.0, 0.0, 0.0), u0, PERIOD / steps_per_period,
        periods * steps_per_period, record_every=record_every,
    )


def _max_position_error(integrator, steps_per_period, periods=10):
    tr = _run(integrator, steps_per_period, periods)
    exact, _ = analytic.cyclotron_exact(ELECTRON, B, (0, 0, 0), U0, tr.t)
    return float(np.max(np.linalg.norm(tr.x - exact, axis=1))) / RADIUS


def test_exact_solution_traces_a_circle_of_the_analytic_gyroradius():
    t = np.linspace(0.0, PERIOD, 400, endpoint=False)  # uniform in time = uniform in angle
    x, _ = analytic.cyclotron_exact(ELECTRON, B, (0, 0, 0), U0, t)
    center = x[:, :2].mean(axis=0)
    radii = np.linalg.norm(x[:, :2] - center, axis=1)
    assert np.allclose(radii, RADIUS, rtol=1e-3)
    assert math.isclose(x[-1, 2], U0[2] / gamma_from_u(*U0) * t[-1], rel_tol=1e-12)  # drift along B uses v = u/gamma


def test_exact_solution_satisfies_the_equation_of_motion():
    # finite-difference check of the closed form itself against du/dt = (q/m) v x B
    t = np.array([0.3 * PERIOD])
    h = 1e-6 * PERIOD
    _, u_plus = analytic.cyclotron_exact(ELECTRON, B, (0, 0, 0), U0, t + h)
    _, u_minus = analytic.cyclotron_exact(ELECTRON, B, (0, 0, 0), U0, t - h)
    _, u_mid = analytic.cyclotron_exact(ELECTRON, B, (0, 0, 0), U0, t)
    dudt = (u_plus[0] - u_minus[0]) / (2 * h)
    v = u_mid[0] / gamma_from_u(*u_mid[0])
    expected = ELECTRON.q_over_m * np.cross(v, [0, 0, B])
    assert np.allclose(dudt, expected, rtol=1e-6)


def test_boris_conserves_speed_to_roundoff_over_many_periods():
    tr = _run(boris_integrate, 20, 500, record_every=20)  # coarse 20 steps/period on purpose
    speed = np.linalg.norm(tr.u, axis=1)
    assert np.max(np.abs(speed / speed[0] - 1.0)) < 1e-12


def test_rk4_loses_speed_steadily_at_the_same_step_size():
    tr = _run(rk4_integrate, 50, 300, record_every=50)
    speed = np.linalg.norm(tr.u, axis=1)
    rel = speed / speed[0] - 1.0
    assert rel[-1] < -1e-5
    assert np.all(np.diff(rel) < 0)  # monotone decay, not noise


def test_boris_gyro_angle_per_step_is_two_atan_of_half_omega_dt():
    for steps in (10, 20, 50):
        tr = _run(boris_integrate, steps, 1)
        angles = np.unwrap(np.arctan2(tr.u[:, 1], tr.u[:, 0]))
        measured = abs(float(np.mean(np.diff(angles))))
        assert math.isclose(measured, analytic.boris_rotation_angle(OMEGA * PERIOD / steps), rel_tol=1e-12)


def test_boris_orbit_radius_matches_closed_form():
    steps = 30
    tr = _run(boris_integrate, steps, 1)
    planar = np.column_stack([tr.x[:3, :2], np.zeros(3)])
    radius = analytic.circumradius(planar[0], planar[1], planar[2])
    assert math.isclose(radius, RADIUS * analytic.boris_orbit_radius_factor(2.0 * math.pi / steps), rel_tol=1e-9)


def test_boris_is_second_order_and_rk4_fourth_order():
    steps = [25, 50, 100, 200]
    dts = [1.0 / s for s in steps]
    boris_errors = [_max_position_error(boris_integrate, s) for s in steps]
    rk4_errors = [_max_position_error(rk4_integrate, s) for s in steps]
    assert 1.9 < analytic.observed_order(dts, boris_errors) < 2.1
    assert 3.5 < analytic.observed_order(dts, rk4_errors) < 5.5
    assert all(r < b for r, b in zip(rk4_errors, boris_errors))  # RK4 follows the phase far better


def test_boris_phase_error_is_the_predicted_frequency_shift():
    """Numerical angular frequency = 2*atan(omega*dt/2)/dt, so after N periods
    the angular lag is N*2*pi*(1 - [2*atan(a)/(2a)]) with a = omega*dt/2."""
    steps, periods = 40, 50
    tr = _run(boris_integrate, steps, periods, record_every=steps)
    a = 0.5 * OMEGA * PERIOD / steps
    lag_per_period = 2.0 * math.pi * (1.0 - math.atan(a) / a)
    # sampling once per nominal period, the unwrapped angle accumulates only the lag
    measured_angle = np.unwrap(np.arctan2(tr.u[:, 1], tr.u[:, 0]))
    measured_lag = abs(measured_angle[-1] - measured_angle[0])
    assert math.isclose(measured_lag, periods * lag_per_period, rel_tol=1e-6)


def test_helix_drift_along_b_is_unaffected_by_the_field():
    tr = _run(boris_integrate, 40, 5)
    v_z = U0[2] / gamma_from_u(*U0)
    assert np.allclose(tr.x[:, 2], v_z * tr.t, rtol=1e-9)
    assert np.allclose(tr.u[:, 2], U0[2], rtol=1e-12)


def test_positive_and_negative_charges_gyrate_in_opposite_senses():
    """After a quarter gyro-period starting along +x in B = +z, the proton has
    turned toward -y and the electron toward +y (each at its own time scale)."""
    u0 = (1.0e6, 0.0, 0.0)
    ends = {}
    for particle in (PROTON, ELECTRON):
        period = 2.0 * math.pi / abs(gyrofrequency_rad_s(particle, B))
        tr = boris_integrate(particle, zero(), uniform(0, 0, B), (0, 0, 0), u0, period / 100, 25)
        ends[particle.name] = tr.x[-1]
    radius_p = gyroradius_m(PROTON, 1.0e6, B)
    radius_e = gyroradius_m(ELECTRON, 1.0e6, B)
    assert ends["proton"][1] < 0 < ends["elektron"][1]
    assert math.isclose(abs(ends["proton"][1]), radius_p, rel_tol=2e-2)
    assert math.isclose(abs(ends["elektron"][1]), radius_e, rel_tol=2e-2)


@pytest.mark.parametrize("integrator", [boris_integrate, rk4_integrate])
def test_zero_fields_give_straight_line_motion(integrator):
    u0 = (1e6, -2e6, 3e5)
    tr = integrator(ELECTRON, zero(), zero(), (1.0, 2.0, 3.0), u0, 1e-9, 100)
    v = np.array(u0) / gamma_from_u(*u0)
    assert np.allclose(tr.x[-1], np.array([1.0, 2.0, 3.0]) + v * tr.t[-1], rtol=1e-12)
