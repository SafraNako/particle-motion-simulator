from __future__ import annotations

import math

import pytest

from parcacik_hareketi_simulatoru.fields import mirror_b_field, uniform
from parcacik_hareketi_simulatoru.particles import (
    C,
    ELECTRON,
    PROTON,
    gamma_from_u,
    gyrofrequency_rad_s,
    gyroradius_m,
    kinetic_energy_j,
    proper_velocity_from_kinetic,
)


def test_masses_and_charges_match_codata():
    assert math.isclose(ELECTRON.mass_kg, 9.1093837e-31, rel_tol=1e-6)
    assert math.isclose(PROTON.mass_kg, 1.67262192e-27, rel_tol=1e-6)
    assert ELECTRON.charge_c < 0 < PROTON.charge_c
    assert math.isclose(abs(ELECTRON.charge_c), PROTON.charge_c)


def test_rest_energies_in_mev():
    assert math.isclose(ELECTRON.rest_energy_mev, 0.51099895, rel_tol=1e-6)
    assert math.isclose(PROTON.rest_energy_mev, 938.27209, rel_tol=1e-6)


def test_gamma_of_zero_velocity_is_one_and_grows_without_bound():
    assert gamma_from_u(0.0, 0.0, 0.0) == 1.0
    assert math.isclose(gamma_from_u(C, 0.0, 0.0), math.sqrt(2.0))  # u = c  <=>  v = c/sqrt(2)
    assert gamma_from_u(1000 * C, 0, 0) > 999


def test_proper_velocity_reproduces_requested_kinetic_energy():
    u_perp, u_par = proper_velocity_from_kinetic(2.5, ELECTRON, math.radians(35.0))
    ke_mev = kinetic_energy_j(u_perp, 0.0, u_par, ELECTRON) / (1.602176634e-13)
    assert math.isclose(ke_mev, 2.5, rel_tol=1e-9)
    assert math.isclose(math.atan2(u_perp, u_par), math.radians(35.0))


def test_gyrofrequency_sign_follows_charge_and_scales_with_gamma():
    assert gyrofrequency_rad_s(PROTON, 1.0) > 0 > gyrofrequency_rad_s(ELECTRON, 1.0)
    assert math.isclose(gyrofrequency_rad_s(PROTON, 1.0, 2.0), 0.5 * gyrofrequency_rad_s(PROTON, 1.0))
    assert math.isclose(gyrofrequency_rad_s(PROTON, 1.0), 9.5788e7, rel_tol=1e-4)  # textbook proton value at 1 T


def test_gyroradius_is_momentum_over_charge_times_field():
    u = 3.0e6
    assert math.isclose(gyroradius_m(PROTON, u, 2.0), PROTON.mass_kg * u / (PROTON.charge_c * 2.0))


def test_uniform_field_returns_constant_vector():
    f = uniform(1.0, 2.0, 3.0)
    assert f(5.0, -2.0, 0.3, 7.0) == (1.0, 2.0, 3.0)


def test_mirror_field_is_divergence_free_and_has_expected_axis_profile():
    b = mirror_b_field(1.5, 0.5)
    assert b(0.0, 0.0, 0.0, 0.0) == (0.0, 0.0, 1.5)
    assert math.isclose(b(0.0, 0.0, 0.5, 0.0)[2], 3.0)  # B0*(1 + 1) at z = L
    h = 1e-6
    x0 = (0.013, -0.02, 0.21)

    def comp(i, d):
        p = list(x0)
        p[i] += d
        return b(*p, 0.0)[i]

    div = sum((comp(i, h) - comp(i, -h)) / (2 * h) for i in range(3))
    assert abs(div) < 1e-8  # compared with the individual terms of order B0/L^2 * z ~ 1
    # the radial component is what cancels the axial gradient: it must be nonzero off axis
    assert b(0.1, 0.0, 0.3, 0.0)[0] != 0.0


@pytest.mark.parametrize("name", ["elektron", "proton"])
def test_particle_registry(name):
    from parcacik_hareketi_simulatoru.particles import PARTICLES

    assert PARTICLES[name].name == name
