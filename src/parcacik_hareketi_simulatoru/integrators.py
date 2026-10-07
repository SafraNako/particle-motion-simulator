"""Particle pushers: Boris (leapfrog, volume-preserving) and classical RK4.

Boris: in a pure magnetic field the velocity update is an exact rotation, so
|u| is conserved to round-off for any step size; the price is a phase error:
the numerical gyro-angle per step is 2*atan(omega*dt/2) instead of omega*dt
(second-order accurate). RK4 has a much smaller per-step truncation error but
no such structure: its amplification factor for pure rotation is slightly
below one, so |u| decays steadily over many gyro-periods.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .fields import Field
from .particles import C, Particle

_C2 = C * C


@dataclass(frozen=True, eq=False)
class Trajectory:
    t: np.ndarray  # (n,)
    x: np.ndarray  # (n, 3) positions, m
    u: np.ndarray  # (n, 3) proper velocity gamma*v, m/s, synchronized with t


def boris_push_velocity(u, e, b, q_over_m: float, h: float):
    """One Boris velocity update over time h (h may be negative)."""
    ux, uy, uz = u
    ex, ey, ez = e
    bx, by, bz = b
    hh = 0.5 * h * q_over_m
    mx, my, mz = ux + hh * ex, uy + hh * ey, uz + hh * ez
    gamma = math.sqrt(1.0 + (mx * mx + my * my + mz * mz) / _C2)
    k = hh / gamma
    tx, ty, tz = k * bx, k * by, k * bz
    sfac = 2.0 / (1.0 + tx * tx + ty * ty + tz * tz)
    sx, sy, sz = sfac * tx, sfac * ty, sfac * tz
    px = mx + (my * tz - mz * ty)
    py = my + (mz * tx - mx * tz)
    pz = mz + (mx * ty - my * tx)
    qx = mx + (py * sz - pz * sy)
    qy = my + (pz * sx - px * sz)
    qz = mz + (px * sy - py * sx)
    return qx + hh * ex, qy + hh * ey, qz + hh * ez


def boris_integrate(
    particle: Particle,
    e_field: Field,
    b_field: Field,
    x0,
    u0,
    dt: float,
    n_steps: int,
    record_every: int = 1,
) -> Trajectory:
    """Leapfrog: positions at t_n, velocities at t_{n+1/2}; the recorded
    velocity is synchronized to t_n with a half step so it is directly
    comparable with positions and with analytic solutions."""
    qm = particle.q_over_m
    x, y, z = (float(v) for v in x0)
    u = tuple(float(v) for v in u0)
    t = 0.0
    u_half = boris_push_velocity(u, e_field(x, y, z, t), b_field(x, y, z, t), qm, -0.5 * dt)

    ts, xs, us = [0.0], [(x, y, z)], [u]
    for n in range(n_steps):
        u_half = boris_push_velocity(u_half, e_field(x, y, z, t), b_field(x, y, z, t), qm, dt)
        gamma = math.sqrt(1.0 + (u_half[0] ** 2 + u_half[1] ** 2 + u_half[2] ** 2) / _C2)
        x += dt * u_half[0] / gamma
        y += dt * u_half[1] / gamma
        z += dt * u_half[2] / gamma
        t = (n + 1) * dt
        if (n + 1) % record_every == 0:
            u_sync = boris_push_velocity(u_half, e_field(x, y, z, t), b_field(x, y, z, t), qm, 0.5 * dt)
            ts.append(t)
            xs.append((x, y, z))
            us.append(u_sync)
    return Trajectory(np.array(ts), np.array(xs), np.array(us))


def _rk4_derivative(state, t, qm, e_field, b_field):
    x, y, z, ux, uy, uz = state
    gamma = math.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) / _C2)
    vx, vy, vz = ux / gamma, uy / gamma, uz / gamma
    ex, ey, ez = e_field(x, y, z, t)
    bx, by, bz = b_field(x, y, z, t)
    ax = qm * (ex + vy * bz - vz * by)
    ay = qm * (ey + vz * bx - vx * bz)
    az = qm * (ez + vx * by - vy * bx)
    return (vx, vy, vz, ax, ay, az)


def rk4_integrate(
    particle: Particle,
    e_field: Field,
    b_field: Field,
    x0,
    u0,
    dt: float,
    n_steps: int,
    record_every: int = 1,
) -> Trajectory:
    qm = particle.q_over_m
    state = tuple(float(v) for v in (*x0, *u0))
    ts, rows = [0.0], [state]
    for n in range(n_steps):
        t = n * dt
        k1 = _rk4_derivative(state, t, qm, e_field, b_field)
        s2 = tuple(s + 0.5 * dt * k for s, k in zip(state, k1))
        k2 = _rk4_derivative(s2, t + 0.5 * dt, qm, e_field, b_field)
        s3 = tuple(s + 0.5 * dt * k for s, k in zip(state, k2))
        k3 = _rk4_derivative(s3, t + 0.5 * dt, qm, e_field, b_field)
        s4 = tuple(s + dt * k for s, k in zip(state, k3))
        k4 = _rk4_derivative(s4, t + dt, qm, e_field, b_field)
        state = tuple(s + dt / 6.0 * (a + 2.0 * b + 2.0 * c + d) for s, a, b, c, d in zip(state, k1, k2, k3, k4))
        if (n + 1) % record_every == 0:
            ts.append((n + 1) * dt)
            rows.append(state)
    arr = np.array(rows)
    return Trajectory(np.array(ts), arr[:, :3], arr[:, 3:])
