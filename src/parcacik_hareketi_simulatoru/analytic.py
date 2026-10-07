"""Closed-form results the numerical pushers are validated against.

All formulas are exact (relativistic) unless stated otherwise.

- Uniform B along z: the speed, and hence gamma, is constant, so the motion is
  a circle (helix with a drift along B) at the signed Larmor frequency
  Omega = q*B/(gamma*m).
- Uniform E along x, particle starting at rest: hyperbolic motion,
  gamma(t) = sqrt(1 + (qEt/mc)^2), x(t) = (m c^2 / qE) * (gamma - 1).
- Crossed uniform E and B with E perpendicular to B and |E| < c|B|: the
  guiding center drifts at v_d = E x B / B^2. Stronger than that statement is
  an exact invariant: in the frame moving at v_d the electric field vanishes,
  so the particle's energy and momentum magnitude *in that frame* are constant.
- Magnetic mirror, B_z = B0(1 + z^2/L^2): p_perp^2/B is an adiabatic invariant
  and |p| is exactly conserved, so at the turning point p_perp = |p| and
  B = B0/sin^2(alpha_0), i.e. z_turn = L*cot(alpha_0).
"""

from __future__ import annotations

import math

import numpy as np

from .particles import C, Particle


def cyclotron_exact(particle: Particle, b_tesla: float, x0, u0, t):
    """Exact (x(t), u(t)) in a uniform field B = b_tesla * z_hat."""
    t = np.asarray(t, dtype=np.float64)
    ux0, uy0, uz0 = (float(v) for v in u0)
    gamma = math.sqrt(1.0 + (ux0**2 + uy0**2 + uz0**2) / (C * C))
    omega = particle.charge_c * b_tesla / (gamma * particle.mass_kg)
    c_, s_ = np.cos(omega * t), np.sin(omega * t)
    vx0, vy0, vz0 = ux0 / gamma, uy0 / gamma, uz0 / gamma  # positions integrate v = u/gamma
    x = x0[0] + (vx0 * s_ + vy0 * (1.0 - c_)) / omega
    y = x0[1] + (vx0 * (c_ - 1.0) + vy0 * s_) / omega
    z = x0[2] + vz0 * t
    ux = ux0 * c_ + uy0 * s_
    uy = -ux0 * s_ + uy0 * c_
    uz = np.full_like(t, uz0)
    return np.stack([x, y, z], axis=1), np.stack([ux, uy, uz], axis=1)


def boris_rotation_angle(omega_dt: float) -> float:
    """Gyro-angle per step of the Boris scheme: 2*atan(omega*dt/2) (vs omega*dt)."""
    return 2.0 * math.atan(0.5 * omega_dt)


def boris_orbit_radius_factor(omega_dt: float) -> float:
    """Ratio of the Boris orbit radius to the true gyroradius: sqrt(1 + (omega*dt/2)^2).

    Each step advances the position by v*dt along a chord that subtends the
    Boris gyro-angle 2*atan(omega*dt/2), so the numerical positions lie on a
    circle of radius v*dt / (2*sin(atan(omega*dt/2))) = (v/omega)*sqrt(1+(omega*dt/2)^2).
    """
    return math.sqrt(1.0 + (0.5 * omega_dt) ** 2)


def circumradius(p1, p2, p3) -> float:
    """Radius of the circle through three points (any dimension)."""
    a = np.linalg.norm(np.asarray(p2) - np.asarray(p1))
    b = np.linalg.norm(np.asarray(p3) - np.asarray(p2))
    c = np.linalg.norm(np.asarray(p3) - np.asarray(p1))
    area2 = np.linalg.norm(np.cross(np.asarray(p2) - np.asarray(p1), np.asarray(p3) - np.asarray(p1)))
    return float(a * b * c / (2.0 * area2))


def hyperbolic_motion(particle: Particle, e_x_v_per_m: float, t):
    """Exact (x(t), u_x(t)) for a particle starting at rest at x = 0 in E = E_x x_hat."""
    t = np.asarray(t, dtype=np.float64)
    m, q = particle.mass_kg, particle.charge_c
    p = q * e_x_v_per_m * t
    gamma = np.sqrt(1.0 + (p / (m * C)) ** 2)
    x = m * C * C / (q * e_x_v_per_m) * (gamma - 1.0)
    return x, p / m


def drift_velocity(e, b):
    e, b = np.asarray(e, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.cross(e, b) / float(np.dot(b, b))


def drift_frame_invariants(particle: Particle, u, e, b):
    """(energy', |p'|) of the particle in the E x B drift frame, for each row of u.

    Exactly constant along the true trajectory in uniform crossed fields with
    E perpendicular to B and |E| < c|B|.
    """
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    v_d = drift_velocity(e, b)
    speed_d = float(np.linalg.norm(v_d))
    if speed_d >= C:
        raise ValueError("|E| < c|B| gerekli (sürüklenme hızı c'den küçük olmalı)")
    m = particle.mass_kg
    gamma = np.sqrt(1.0 + np.sum(u * u, axis=1) / (C * C))
    energy = gamma * m * C * C
    p = m * u
    if speed_d == 0.0:
        return energy, np.linalg.norm(p, axis=1)
    n_hat = v_d / speed_d
    beta_d = speed_d / C
    gamma_d = 1.0 / math.sqrt(1.0 - beta_d * beta_d)
    p_par = p @ n_hat
    p_perp = p - np.outer(p_par, n_hat)
    energy_p = gamma_d * (energy - speed_d * p_par)
    p_par_p = gamma_d * (p_par - beta_d * energy / C)
    p_mag = np.sqrt(p_par_p**2 + np.sum(p_perp * p_perp, axis=1))
    return energy_p, p_mag


def mirror_turning_point_m(length_m: float, pitch_angle_rad: float) -> float:
    """z_turn = L*cot(alpha_0) for B_z = B0*(1 + z^2/L^2), alpha_0 the pitch angle at z = 0."""
    return length_m / math.tan(pitch_angle_rad)


def adiabatic_invariant(particle: Particle, u, b):
    """p_perp^2 / (2 m |B|) for each row (p_perp measured against the local B direction)."""
    u = np.atleast_2d(np.asarray(u, dtype=np.float64))
    b = np.atleast_2d(np.asarray(b, dtype=np.float64))
    b_mag = np.linalg.norm(b, axis=1)
    b_hat = b / b_mag[:, None]
    p = particle.mass_kg * u
    p_par = np.sum(p * b_hat, axis=1)
    p_perp_sq = np.sum(p * p, axis=1) - p_par**2
    return p_perp_sq / (2.0 * particle.mass_kg * b_mag)


def observed_order(step_sizes, errors) -> float:
    """Slope of log(error) vs log(step): the observed order of convergence."""
    slope, _ = np.polyfit(np.log(np.asarray(step_sizes)), np.log(np.asarray(errors)), 1)
    return float(slope)
