from __future__ import annotations

import math

import numpy as np
import pytest

from parcacik_hareketi_simulatoru import analytic
from parcacik_hareketi_simulatoru.fields import uniform
from parcacik_hareketi_simulatoru.integrators import boris_integrate
from parcacik_hareketi_simulatoru.particles import C, ELECTRON, PROTON, gamma_from_u, gyrofrequency_rad_s

B = 1.0


def _run(particle, e_over_cb, u0, steps_per_period=100, periods=30, record_every=None):
    e_mag = e_over_cb * C * B
    omega = abs(gyrofrequency_rad_s(particle, B, gamma_from_u(*u0)))
    period = 2.0 * math.pi / omega
    return boris_integrate(
        particle, uniform(e_mag, 0, 0), uniform(0, 0, B), (0, 0, 0), u0, period / steps_per_period,
        periods * steps_per_period, record_every=record_every or max(1, steps_per_period // 10),
    ), np.array([e_mag, 0.0, 0.0]), np.array([0.0, 0.0, B])


def test_drift_velocity_direction_and_magnitude():
    v = analytic.drift_velocity([2.0, 0.0, 0.0], [0.0, 0.0, 4.0])
    assert np.allclose(v, [0.0, -0.5, 0.0])  # E x B / B^2 with x_hat cross z_hat = -y_hat


def test_drift_is_independent_of_charge_sign_and_mass():
    e, b = [3.0e5, 0.0, 0.0], [0.0, 0.0, 2.0]
    assert np.allclose(analytic.drift_velocity(e, b), analytic.drift_velocity(e, b))
    for particle in (PROTON, ELECTRON):
        tr, *_ = _run(particle, 1e-3, (1.0e5, 0.0, 0.0), periods=20)
        slope_y = np.polyfit(tr.t, tr.x[:, 1], 1)[0]
        assert slope_y < 0  # same -y direction for both charges
        assert math.isclose(abs(slope_y), 1e-3 * C, rel_tol=0.02)


def test_nonrelativistic_cycloid_matches_closed_form():
    """Starting at rest with E/(cB) = 1e-3 the motion is the textbook cycloid
    x = (E/(B*Om)) (1 - cos), y = (E/(B*Om)) (sin - Om*t); relativistic
    corrections are O((E/cB)^2) = 1e-6."""
    e_mag = 1e-3 * C * B
    omega = PROTON.charge_c * B / PROTON.mass_kg
    period = 2.0 * math.pi / omega
    scale = e_mag / (B * omega)
    errors = []
    for steps in (100, 200, 400):
        tr = boris_integrate(
            PROTON, uniform(e_mag, 0, 0), uniform(0, 0, B), (0, 0, 0), (0, 0, 0), period / steps, 5 * steps
        )
        x = scale * (1.0 - np.cos(omega * tr.t))
        y = scale * (np.sin(omega * tr.t) - omega * tr.t)
        errors.append(float(np.max(np.hypot(tr.x[:, 0] - x, tr.x[:, 1] - y))) / scale)
    assert errors[-1] < 3e-3
    assert 1.9 < analytic.observed_order([1 / 100, 1 / 200, 1 / 400], errors) < 2.1


@pytest.mark.parametrize("e_over_cb", [0.0, 0.3, 0.6, 0.9])
def test_drift_frame_invariants_are_conserved(e_over_cb):
    u0 = (0.1 * C, 0.05 * C, 0.02 * C)
    tr, e, b = _run(PROTON, e_over_cb, u0, steps_per_period=200)
    energy, p_mag = analytic.drift_frame_invariants(PROTON, tr.u, e, b)
    assert np.ptp(energy) / np.mean(energy) < 2e-4
    assert np.ptp(p_mag) / np.mean(p_mag) < 2e-4


def test_invariant_drift_shrinks_quadratically_with_step_size():
    u0 = (0.1 * C, 0.05 * C, 0.02 * C)
    variations = []
    for steps in (50, 100, 200):
        tr, e, b = _run(PROTON, 0.6, u0, steps_per_period=steps)
        energy, _ = analytic.drift_frame_invariants(PROTON, tr.u, e, b)
        variations.append(float(np.ptp(energy) / np.mean(energy)))
    assert math.isclose(variations[0] / variations[1], 4.0, rel_tol=0.1)
    assert math.isclose(variations[1] / variations[2], 4.0, rel_tol=0.1)


@pytest.mark.parametrize("e_over_cb", [0.3, 0.6, 0.9])
def test_relativistic_mean_drift_speed_is_e_over_b(e_over_cb):
    tr, e, b = _run(PROTON, e_over_cb, (0.1 * C, 0.0, 0.0), steps_per_period=100, periods=200, record_every=100)
    v_d = analytic.drift_velocity(e, b)
    along = tr.x @ (v_d / np.linalg.norm(v_d))
    slope = float(np.polyfit(tr.t, along, 1)[0])
    assert math.isclose(slope, np.linalg.norm(v_d), rel_tol=0.01)


def test_invariants_were_actually_changing_in_the_lab_frame():
    """Guard against a vacuous test: in the lab frame the particle's energy is
    NOT constant (the electric field does work); only the drift-frame energy is."""
    u0 = (0.1 * C, 0.05 * C, 0.02 * C)
    tr, e, b = _run(PROTON, 0.6, u0, steps_per_period=200)
    gamma = np.sqrt(1.0 + np.sum(tr.u**2, axis=1) / C**2)
    lab_variation = np.ptp(gamma) / np.mean(gamma)
    energy, _ = analytic.drift_frame_invariants(PROTON, tr.u, e, b)
    assert lab_variation > 100 * (np.ptp(energy) / np.mean(energy))


def test_superluminal_drift_is_rejected():
    with pytest.raises(ValueError):
        analytic.drift_frame_invariants(PROTON, [[1.0, 0.0, 0.0]], [1.1 * C, 0.0, 0.0], [0.0, 0.0, 1.0])
