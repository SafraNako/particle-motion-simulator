from __future__ import annotations

import math

import numpy as np

from parcacik_hareketi_simulatoru import analytic
from parcacik_hareketi_simulatoru.fields import mirror_b_field, zero
from parcacik_hareketi_simulatoru.integrators import boris_integrate
from parcacik_hareketi_simulatoru.particles import (
    ELECTRON,
    PROTON,
    gamma_from_u,
    gyrofrequency_rad_s,
    gyroradius_m,
    proper_velocity_from_kinetic,
)


def _bounce(particle, kinetic_mev, pitch_deg, b0, length, bounces=2.5, steps_per_period=40):
    alpha = math.radians(pitch_deg)
    u_perp, u_par = proper_velocity_from_kinetic(kinetic_mev, particle, alpha)
    u0 = (u_perp, 0.0, u_par)
    gamma = gamma_from_u(*u0)
    dt = 2.0 * math.pi / abs(gyrofrequency_rad_s(particle, b0, gamma)) / steps_per_period
    z_turn = analytic.mirror_turning_point_m(length, alpha)
    n = int(bounces * 4.0 * z_turn / (u_par / gamma) / dt)
    field = mirror_b_field(b0, length)
    tr = boris_integrate(particle, zero(), field, (0, 0, 0), u0, dt, n, record_every=steps_per_period // 4)
    b_along = np.array([field(*pos, 0.0) for pos in tr.x])
    rho = gyroradius_m(particle, u_perp, b0)
    return tr, b_along, z_turn, rho


def test_turning_point_formula_is_the_field_value_that_stops_parallel_motion():
    # |p| conserved and p_perp^2/B conserved => p_perp = |p| where B = B0/sin^2(alpha)
    alpha = math.radians(40.0)
    z = analytic.mirror_turning_point_m(1.0, alpha)
    assert math.isclose(1.0 + z * z, 1.0 / math.sin(alpha) ** 2)


def test_proton_turning_point_matches_prediction():
    tr, _, z_turn, rho = _bounce(PROTON, 0.005, 40.0, 1.0, 1.0)
    assert rho / 1.0 < 0.01
    z_max, z_min = tr.x[:, 2].max(), tr.x[:, 2].min()
    assert math.isclose(z_max, z_turn, rel_tol=5e-4)
    assert math.isclose(z_min, -z_turn, rel_tol=5e-4)


def test_speed_is_conserved_exactly_in_a_pure_magnetic_field():
    tr, *_ = _bounce(PROTON, 0.005, 40.0, 1.0, 1.0)
    speed = np.linalg.norm(tr.u, axis=1)
    assert np.max(np.abs(speed / speed[0] - 1.0)) < 1e-12


def test_adiabatic_invariant_is_nearly_constant_and_the_orbit_actually_bounces():
    tr, b_along, _, _ = _bounce(PROTON, 0.005, 40.0, 1.0, 1.0)
    mu = analytic.adiabatic_invariant(PROTON, tr.u, b_along)
    assert np.max(np.abs(mu / mu[0] - 1.0)) < 1e-3
    # not vacuous: the pitch changes a lot along the orbit, and B varies by a large factor
    assert b_along[:, 2].max() / b_along[:, 2].min() > 2.0
    # reflected at +z_turn, at -z_turn and again at +z_turn. (The nominal bounce period used to choose
    # the run length assumes the midplane parallel speed, so it underestimates the real one.)
    assert np.sum(np.diff(np.sign(tr.u[:, 2])) != 0) >= 3


def test_turning_point_error_and_invariant_drift_scale_with_gyroradius_over_length_squared():
    """Halving rho/L (same particle, doubled L) should cut both errors by ~4."""
    errors, mu_drift = [], []
    for length in (1.0, 2.0):
        tr, b_along, z_turn, _ = _bounce(PROTON, 0.005, 40.0, 1.0, length)
        errors.append(abs(tr.x[:, 2].max() - z_turn) / z_turn)
        mu = analytic.adiabatic_invariant(PROTON, tr.u, b_along)
        mu_drift.append(float(np.max(np.abs(mu / mu[0] - 1.0))))
    assert 2.5 < errors[0] / errors[1] < 6.0
    assert 2.5 < mu_drift[0] / mu_drift[1] < 6.0


def test_relativistic_electron_obeys_the_same_turning_point_law():
    tr, _, z_turn, rho = _bounce(ELECTRON, 0.5, 40.0, 0.2, 2.0)
    assert rho / 2.0 < 0.01
    assert math.isclose(tr.x[:, 2].max(), z_turn, rel_tol=1e-3)
    assert gamma_from_u(*tr.u[0]) > 1.9  # genuinely relativistic


def test_turning_point_depends_on_pitch_and_length_but_not_on_mass_or_charge():
    """An electron with the proton's speed in a field scaled by m_e/m_p has the
    same gyroradius and gyro-period (opposite sense), so the two orbits are
    directly comparable; the turning point z = L*cot(alpha_0) must agree."""
    mass_ratio = ELECTRON.mass_kg / PROTON.mass_kg
    proton, *_ = _bounce(PROTON, 0.005, 40.0, 1.0, 1.0)
    electron, *_ = _bounce(ELECTRON, 0.005 * mass_ratio, 40.0, 1.0 * mass_ratio, 1.0)
    assert math.isclose(proton.x[:, 2].max(), electron.x[:, 2].max(), rel_tol=1e-3)
    assert math.isclose(proton.x[:, 2].min(), electron.x[:, 2].min(), rel_tol=1e-3)


def test_larger_pitch_angle_reflects_closer_to_the_center():
    turn_30 = _bounce(PROTON, 0.005, 30.0, 1.0, 1.0)[0].x[:, 2].max()
    turn_60 = _bounce(PROTON, 0.005, 60.0, 1.0, 1.0)[0].x[:, 2].max()
    assert turn_30 > turn_60
    assert math.isclose(turn_60, analytic.mirror_turning_point_m(1.0, math.radians(60.0)), rel_tol=1e-3)


def test_purely_perpendicular_particle_never_leaves_the_midplane():
    u0 = (1.0e6, 0.0, 0.0)
    tr = boris_integrate(PROTON, zero(), mirror_b_field(1.0, 1.0), (0, 0, 0), u0, 1e-9, 2000)
    assert np.max(np.abs(tr.x[:, 2])) < 1e-12


def test_particle_moving_along_the_axis_feels_no_force():
    u0 = (0.0, 0.0, 1.0e6)
    tr = boris_integrate(PROTON, zero(), mirror_b_field(1.0, 1.0), (0, 0, 0), u0, 1e-8, 500)
    assert np.allclose(tr.x[:, :2], 0.0, atol=1e-12)
    assert np.allclose(tr.u[:, 2], 1.0e6, rtol=1e-12)
