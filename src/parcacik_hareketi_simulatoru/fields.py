"""Prescribed field callables f(x, y, z, t) -> (fx, fy, fz) in SI units."""

from __future__ import annotations

from typing import Callable, Tuple

Field = Callable[[float, float, float, float], Tuple[float, float, float]]


def uniform(fx: float, fy: float, fz: float) -> Field:
    vec = (float(fx), float(fy), float(fz))
    return lambda x, y, z, t: vec


def zero() -> Field:
    return uniform(0.0, 0.0, 0.0)


def mirror_b_field(b0_tesla: float, length_m: float) -> Field:
    """Paraxial magnetic bottle with B_z(z) = B0*(1 + z^2/L^2).

    The radial part B_r = -(r/2) dB_z/dz = -B0*r*z/L^2 is what makes the
    field divergence-free; in Cartesian components

        B = (-B0*x*z/L^2, -B0*y*z/L^2, B0*(1 + z^2/L^2)).

    This is a paraxial model (it is not curl-free off axis), valid for
    gyroradii much smaller than L.
    """
    inv_l2 = 1.0 / (length_m * length_m)

    def field(x: float, y: float, z: float, t: float):
        return (-b0_tesla * x * z * inv_l2, -b0_tesla * y * z * inv_l2, b0_tesla * (1.0 + z * z * inv_l2))

    return field
