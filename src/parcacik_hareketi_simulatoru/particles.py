"""Particles and basic relativistic quantities.

Throughout this package a particle's state is (x, u) with u = gamma*v in m/s
(proper velocity). It stays finite as v -> c and makes the equation of motion
du/dt = (q/m) (E + v x B), v = u/gamma, gamma = sqrt(1 + |u|^2/c^2).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from scipy import constants

C = constants.c


@dataclass(frozen=True)
class Particle:
    name: str
    mass_kg: float
    charge_c: float

    @property
    def q_over_m(self) -> float:
        return self.charge_c / self.mass_kg

    @property
    def rest_energy_mev(self) -> float:
        return self.mass_kg * C * C / (constants.e * 1e6)


ELECTRON = Particle("elektron", constants.m_e, -constants.e)
PROTON = Particle("proton", constants.m_p, constants.e)
PARTICLES = {"elektron": ELECTRON, "proton": PROTON}


def gamma_from_u(ux: float, uy: float, uz: float) -> float:
    return math.sqrt(1.0 + (ux * ux + uy * uy + uz * uz) / (C * C))


def kinetic_energy_j(ux: float, uy: float, uz: float, particle: Particle) -> float:
    return (gamma_from_u(ux, uy, uz) - 1.0) * particle.mass_kg * C * C


def gyrofrequency_rad_s(particle: Particle, b_tesla: float, gamma: float = 1.0) -> float:
    """Signed Larmor angular frequency q*B/(gamma*m)."""
    return particle.charge_c * b_tesla / (gamma * particle.mass_kg)


def gyroradius_m(particle: Particle, u_perp_m_s: float, b_tesla: float) -> float:
    """r = p_perp/(|q| B) = m*u_perp/(|q| B), exact relativistically."""
    return particle.mass_kg * u_perp_m_s / (abs(particle.charge_c) * abs(b_tesla))


def proper_velocity_from_kinetic(kinetic_mev: float, particle: Particle, pitch_rad: float):
    """(u_perp, u_parallel) for a given kinetic energy and pitch angle to B."""
    gamma = 1.0 + kinetic_mev / particle.rest_energy_mev
    speed_u = C * math.sqrt(gamma * gamma - 1.0)
    return speed_u * math.sin(pitch_rad), speed_u * math.cos(pitch_rad)
