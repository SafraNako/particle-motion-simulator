from __future__ import annotations

import math

import numpy as np

from parcacik_hareketi_simulatoru import analytic
from parcacik_hareketi_simulatoru.fields import uniform, zero
from parcacik_hareketi_simulatoru.integrators import boris_integrate, rk4_integrate
from parcacik_hareketi_simulatoru.particles import C, ELECTRON, PROTON

E = 1e8
T_C = PROTON.mass_kg * C / (PROTON.charge_c * E)  # characteristic time m c / (q E)


def _run(integrator, steps_per_tc, span_tc=20):
    return integrator(
        PROTON, uniform(E, 0, 0), zero(), (0, 0, 0), (0, 0, 0), T_C / steps_per_tc, int(steps_per_tc * span_tc)
    )


def test_exact_solution_limits():
    t = np.array([1e-3 * T_C, 1e3 * T_C])
    x, ux = analytic.hyperbolic_motion(PROTON, E, t)
    # early times: Newtonian x = a t^2 / 2 and u = a t
    a = PROTON.charge_c * E / PROTON.mass_kg
    assert math.isclose(x[0], 0.5 * a * t[0] ** 2, rel_tol=1e-5)
    assert math.isclose(ux[0], a * t[0], rel_tol=1e-12)
    # late times: moves at c, so x grows like c*t
    assert math.isclose(x[1] / (C * t[1]), 1.0, rel_tol=2e-3)


def test_boris_proper_velocity_is_exact_in_a_uniform_electric_field():
    """The two half-kicks add up to the exact momentum change q*E*dt."""
    tr = _run(boris_integrate, 10)
    _, ux_exact = analytic.hyperbolic_motion(PROTON, E, tr.t)
    assert np.allclose(tr.u[:, 0], ux_exact, rtol=1e-12, atol=1e-3)


def test_boris_position_is_second_order():
    errors, dts = [], []
    for spt in (5, 10, 20, 40):
        tr = _run(boris_integrate, spt)
        x_exact, _ = analytic.hyperbolic_motion(PROTON, E, tr.t)
        errors.append(abs(tr.x[-1, 0] - x_exact[-1]) / abs(x_exact[-1]))
        dts.append(1.0 / spt)
    assert 1.9 < analytic.observed_order(dts, errors) < 2.1


def test_speed_never_reaches_c_whereas_newton_would_exceed_it():
    tr = _run(boris_integrate, 20, span_tc=50)
    gamma = np.sqrt(1.0 + np.sum(tr.u**2, axis=1) / C**2)
    speed = np.linalg.norm(tr.u, axis=1) / gamma
    assert np.all(speed < C)
    assert speed[-1] / C > 0.9997
    newton_speed_over_c = tr.t[-1] / T_C  # a*t/c = t/t_c
    assert newton_speed_over_c > 49


def test_kinetic_energy_gain_equals_work_done_by_the_field():
    tr = _run(boris_integrate, 20)
    gamma = np.sqrt(1.0 + np.sum(tr.u**2, axis=1) / C**2)
    energy_gain = (gamma[-1] - 1.0) * PROTON.mass_kg * C**2
    work = PROTON.charge_c * E * tr.x[-1, 0]
    assert math.isclose(energy_gain, work, rel_tol=2e-4)


def test_rk4_also_follows_the_exact_solution():
    tr = _run(rk4_integrate, 20)
    x_exact, _ = analytic.hyperbolic_motion(PROTON, E, tr.t)
    assert abs(tr.x[-1, 0] - x_exact[-1]) / abs(x_exact[-1]) < 1e-7


def test_electron_accelerates_opposite_to_the_field():
    tr = boris_integrate(ELECTRON, uniform(E, 0, 0), zero(), (0, 0, 0), (0, 0, 0), 1e-14, 3000)
    x_exact, _ = analytic.hyperbolic_motion(ELECTRON, E, tr.t)
    assert tr.x[-1, 0] < 0
    assert math.isclose(tr.x[-1, 0], x_exact[-1], rel_tol=1e-3)
