from __future__ import annotations

import argparse
import math
import sys

import numpy as np

from . import analytic
from .fields import mirror_b_field, uniform, zero
from .integrators import boris_integrate, rk4_integrate
from .particles import (
    C,
    PARTICLES,
    gamma_from_u,
    gyrofrequency_rad_s,
    gyroradius_m,
    proper_velocity_from_kinetic,
)
from .render import plot_convergence, plot_crossed_fields, plot_cyclotron, plot_hyperbolic, plot_mirror


def _check_positive(**values: float) -> None:
    for name, value in values.items():
        if value <= 0:
            raise ValueError(f"{name} pozitif olmalı")


def _cmd_siklotron(args: argparse.Namespace) -> int:
    _check_positive(kinetik=args.kinetik, b=args.b, adim_sayisi=args.adim_sayisi, periyot_sayisi=args.periyot_sayisi)
    particle = PARTICLES[args.parcacik]
    u_perp, u_par = proper_velocity_from_kinetic(args.kinetik, particle, math.radians(args.pitch))
    u0 = (u_perp, 0.0, u_par)
    gamma = gamma_from_u(*u0)
    omega = abs(gyrofrequency_rad_s(particle, args.b, gamma))
    period = 2.0 * math.pi / omega
    steps = args.adim_sayisi
    dt = period / steps
    b_field = uniform(0.0, 0.0, args.b)
    radius = gyroradius_m(particle, u_perp, args.b)

    print(f"{particle.name}, T = {args.kinetik} MeV (γ = {gamma:.4f}), B = {args.b} T, pitch = {args.pitch}°")
    print(f"Jirasyon periyodu: {period:.4e} s   jirasyon yarıçapı r = p⊥/(|q|B) = {radius:.4e} m")

    short = boris_integrate(particle, zero(), b_field, (0, 0, 0), u0, dt, 3 * steps)
    x_exact, _ = analytic.cyclotron_exact(particle, args.b, (0, 0, 0), u0, short.t)
    # the in-plane projection of a helix is a circle of radius v_perp/omega
    planar = short.x[:, :2]
    planar3 = np.column_stack([planar, np.zeros(len(planar))])
    numeric_radius = analytic.circumradius(planar3[0], planar3[1], planar3[2])
    print(
        f"Sayısal yörünge yarıçapı (3 noktadan geçen çember): {numeric_radius:.6e} m  "
        f"(öngörülen r·√(1+(ωdt/2)²) = {radius * analytic.boris_orbit_radius_factor(omega * dt):.6e} m)"
    )
    angles = np.unwrap(np.arctan2(short.u[:, 1], short.u[:, 0]))
    measured_angle = abs(float(np.mean(np.diff(angles))))
    print(
        f"Boris adım başına jirasyon açısı: ölçülen {measured_angle:.8f} rad, "
        f"öngörülen 2·atan(ωdt/2) = {analytic.boris_rotation_angle(omega * dt):.8f} rad (saf ωdt = {omega * dt:.8f})"
    )

    n = args.periyot_sayisi * steps
    boris = boris_integrate(particle, zero(), b_field, (0, 0, 0), u0, dt, n, record_every=steps)
    rk4 = rk4_integrate(particle, zero(), b_field, (0, 0, 0), u0, dt, n, record_every=steps)
    speed0 = float(np.linalg.norm(boris.u[0]))
    boris_drift = np.linalg.norm(boris.u, axis=1) / speed0 - 1.0
    rk4_drift = np.linalg.norm(rk4.u, axis=1) / speed0 - 1.0
    x_end, _ = analytic.cyclotron_exact(particle, args.b, (0, 0, 0), u0, boris.t[-1:])
    err_boris = float(np.linalg.norm(boris.x[-1] - x_end[0])) / radius
    err_rk4 = float(np.linalg.norm(rk4.x[-1] - x_end[0])) / radius
    print(f"{args.periyot_sayisi} periyot sonra, {steps} adım/periyot:")
    print(f"  Boris: |u| göreli kayması = {boris_drift[-1]:+.2e}   konum hatası/r = {err_boris:.2e}")
    print(f"  RK4  : |u| göreli kayması = {rk4_drift[-1]:+.2e}   konum hatası/r = {err_rk4:.2e}")
    print("Not: Boris saf manyetik alanda |u|'yu yuvarlama hassasiyetinde korur ama faz hatası verir; RK4 fazı daha iyi izler ama enerjiyi sürekli kaybeder.")

    if args.grafik:
        plot_cyclotron(short.x[:, :2], x_exact[:, :2], np.arange(len(boris_drift)), boris_drift, rk4_drift, args.grafik)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_capraz_alanlar(args: argparse.Namespace) -> int:
    _check_positive(kinetik=args.kinetik, b=args.b, adim_sayisi=args.adim_sayisi, periyot_sayisi=args.periyot_sayisi)
    if not 0.0 <= args.e_orani < 1.0:
        raise ValueError("e-orani E/(cB) 0 ile 1 arasında (1 hariç) olmalı")
    particle = PARTICLES[args.parcacik]
    e_mag = args.e_orani * C * args.b
    e_vec, b_vec = np.array([e_mag, 0.0, 0.0]), np.array([0.0, 0.0, args.b])
    u_perp, _ = proper_velocity_from_kinetic(args.kinetik, particle, math.pi / 2.0)
    u0 = (u_perp, 0.0, 0.0)
    omega = abs(gyrofrequency_rad_s(particle, args.b, gamma_from_u(*u0)))
    period = 2.0 * math.pi / omega
    v_d = analytic.drift_velocity(e_vec, b_vec)
    speed_d = float(np.linalg.norm(v_d))

    print(f"{particle.name}, E/(cB) = {args.e_orani}, B = {args.b} T, T = {args.kinetik} MeV")
    print(f"Öngörülen sürüklenme hızı E×B/B² = {speed_d / C:.4f} c  (yön: {np.round(v_d / speed_d, 3) if speed_d else 'yok'})")

    results = []
    for steps in (args.adim_sayisi, 2 * args.adim_sayisi):
        tr = boris_integrate(
            particle,
            uniform(*e_vec),
            uniform(*b_vec),
            (0, 0, 0),
            u0,
            period / steps,
            args.periyot_sayisi * steps,
            record_every=max(1, steps // 10),
        )
        energy, p_mag = analytic.drift_frame_invariants(particle, tr.u, e_vec, b_vec)
        results.append((steps, tr, float(np.ptp(energy) / np.mean(energy)), float(np.ptp(p_mag) / np.mean(p_mag))))

    print("Sürüklenme çerçevesinde değişmezler (enerji', |p'|) -- tam sabit olmalı:")
    for steps, _, e_var, p_var in results:
        print(f"  {steps:>4} adım/periyot: enerji' göreli değişimi = {e_var:.2e}   |p'| = {p_var:.2e}")
    if results[0][2] > 0 and results[1][2] > 0:
        print(f"  adım yarılanınca enerji' değişimi {results[0][2] / results[1][2]:.2f} kat azalıyor (ikinci mertebe: 4)")

    _, tr, _, _ = results[-1]
    if speed_d > 0:
        n_hat = v_d / speed_d
        along = tr.x @ n_hat
        measured = float(np.polyfit(tr.t, along, 1)[0])
        print(f"Ölçülen ortalama sürüklenme hızı: {measured / C:.4f} c (öngörülen {speed_d / C:.4f} c)")
        # the mean line: through the time-averaged position, moving at the predicted drift velocity
        center = tr.x.mean(axis=0) + v_d * (tr.t - tr.t.mean())[:, None]
        gx, gy = center[:, 0], center[:, 1]
    else:
        gx, gy = np.zeros_like(tr.t), np.zeros_like(tr.t)
    if args.grafik:
        plot_crossed_fields(tr.x[:, 0], tr.x[:, 1], gx, gy, args.grafik)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_hiperbolik_hareket(args: argparse.Namespace) -> int:
    _check_positive(e=args.e, adim_sayisi=args.adim_sayisi, sure_tc=args.sure_tc)
    particle = PARTICLES[args.parcacik]
    t_c = particle.mass_kg * C / (abs(particle.charge_c) * args.e)
    dt = t_c / args.adim_sayisi
    n = int(args.adim_sayisi * args.sure_tc)
    tr = boris_integrate(particle, uniform(args.e, 0.0, 0.0), zero(), (0, 0, 0), (0, 0, 0), dt, n)
    x_exact, ux_exact = analytic.hyperbolic_motion(particle, args.e, tr.t)
    gamma_end = gamma_from_u(*tr.u[-1])
    pos_err = abs(tr.x[-1, 0] - x_exact[-1]) / abs(x_exact[-1])
    u_err = abs(tr.u[-1, 0] - ux_exact[-1]) / abs(ux_exact[-1])
    v_num = np.abs(tr.u[:, 0]) / np.sqrt(1.0 + np.sum(tr.u**2, axis=1) / C**2) / C
    v_exact = np.abs(ux_exact) / np.sqrt(1.0 + (ux_exact / C) ** 2) / C
    t_over_tc = tr.t / t_c
    v_newton = t_over_tc

    print(f"{particle.name}, E = {args.e:.3e} V/m, karakteristik süre t_c = mc/(qE) = {t_c:.4e} s")
    print(f"t = {args.sure_tc} t_c: γ = {gamma_end:.3f}, v/c = {v_num[-1]:.6f}")
    print(f"Newton mekaniği v/c = {v_newton[-1]:.1f} verirdi -- ışık hızını {v_newton[-1]:.1f} kat aşar.")
    print(f"Kesin çözüme göre özgül hız hatası: {u_err:.2e}   konum hatası: {pos_err:.2e}  ({args.adim_sayisi} adım/t_c)")
    if args.grafik:
        plot_hyperbolic(t_over_tc, v_num, v_exact, v_newton, args.grafik)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_manyetik_ayna(args: argparse.Namespace) -> int:
    _check_positive(
        kinetik=args.kinetik, b=args.b, uzunluk=args.uzunluk, adim_sayisi=args.adim_sayisi, sicrama_sayisi=args.sicrama_sayisi
    )
    if not 0.0 < args.pitch < 90.0:
        raise ValueError("pitch açısı 0° ile 90° arasında olmalı")
    particle = PARTICLES[args.parcacik]
    alpha = math.radians(args.pitch)
    u_perp, u_par = proper_velocity_from_kinetic(args.kinetik, particle, alpha)
    u0 = (u_perp, 0.0, u_par)
    gamma = gamma_from_u(*u0)
    z_turn = analytic.mirror_turning_point_m(args.uzunluk, alpha)
    omega = abs(gyrofrequency_rad_s(particle, args.b, gamma))
    dt = 2.0 * math.pi / omega / args.adim_sayisi
    rho = gyroradius_m(particle, u_perp, args.b)
    bounce_time = 4.0 * z_turn / (u_par / gamma)
    n = int(args.sicrama_sayisi * bounce_time / dt)
    b_field = mirror_b_field(args.b, args.uzunluk)
    tr = boris_integrate(particle, zero(), b_field, (0, 0, 0), u0, dt, n, record_every=max(1, args.adim_sayisi // 4))

    b_along = np.array([b_field(*pos, 0.0) for pos in tr.x])
    mu = analytic.adiabatic_invariant(particle, tr.u, b_along)
    mu_rel = mu / mu[0] - 1.0
    speed = np.linalg.norm(tr.u, axis=1)

    print(f"{particle.name}, T = {args.kinetik} MeV, B0 = {args.b} T, L = {args.uzunluk} m, pitch α₀ = {args.pitch}°")
    print(f"Jirasyon yarıçapı / L = {rho / args.uzunluk:.4f}  (adyabatik yaklaşım ρ ≪ L gerektirir)")
    print(f"Öngörülen dönüm noktası z = L·cot(α₀) = {z_turn:.4f} m")
    print(f"Ölçülen azami |z|: {np.abs(tr.x[:, 2]).max():.4f} m   (göreli fark {abs(np.abs(tr.x[:, 2]).max() - z_turn) / z_turn:.2e})")
    print(f"Adyabatik değişmez μ = p⊥²/(2mB) göreli değişimi (azami): {np.abs(mu_rel).max():.2e}")
    print(f"|u| göreli kayması (azami): {np.abs(speed / speed[0] - 1.0).max():.2e}  (saf manyetik alan: tam korunmalı)")
    if args.grafik:
        plot_mirror(tr.t, tr.x[:, 2], z_turn, mu_rel, args.grafik)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def _cmd_yakinsama(args: argparse.Namespace) -> int:
    _check_positive(periyot_sayisi=args.periyot_sayisi)
    particle = PARTICLES[args.parcacik]
    u_perp, u_par = proper_velocity_from_kinetic(args.kinetik, particle, math.radians(60.0))
    u0 = (u_perp, 0.0, u_par)
    omega = abs(gyrofrequency_rad_s(particle, 1.0, gamma_from_u(*u0)))
    period = 2.0 * math.pi / omega
    radius = gyroradius_m(particle, u_perp, 1.0)
    steps_list = [25, 50, 100, 200]
    errors = {"Boris": [], "RK4": []}
    print(f"{particle.name}, T = {args.kinetik} MeV: {args.periyot_sayisi} periyot boyunca kesin siklotron çözümüne karşı azami konum hatası/r")
    print(f"{'adım/periyot':>13} | {'Boris':>12} | {'RK4':>12}")
    for steps in steps_list:
        row = []
        for name, integrate in (("Boris", boris_integrate), ("RK4", rk4_integrate)):
            tr = integrate(particle, zero(), uniform(0, 0, 1.0), (0, 0, 0), u0, period / steps, args.periyot_sayisi * steps)
            exact, _ = analytic.cyclotron_exact(particle, 1.0, (0, 0, 0), u0, tr.t)
            err = float(np.max(np.linalg.norm(tr.x - exact, axis=1))) / radius
            errors[name].append(err)
            row.append(err)
        print(f"{steps:>13} | {row[0]:>12.3e} | {row[1]:>12.3e}")
    dts = [1.0 / s for s in steps_list]
    for name in ("Boris", "RK4"):
        print(f"{name}: gözlenen yakınsama mertebesi = {analytic.observed_order(dts, errors[name]):.2f}")
    if args.grafik:
        plot_convergence(dts, errors["Boris"], errors["RK4"], args.grafik)
        print(f"Grafik kaydedildi: {args.grafik}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="parcacik-hareketi-simulatoru", description="Göreli yüklü parçacık hareketi simülatörü")
    sub = parser.add_subparsers(dest="komut", required=True)
    particles = list(PARTICLES)

    p1 = sub.add_parser("siklotron", help="Düzgün B alanında siklotron hareketi: Boris vs RK4")
    p1.add_argument("--parcacik", default="elektron", choices=particles)
    p1.add_argument("--kinetik", type=float, default=1.0, help="MeV")
    p1.add_argument("--b", type=float, default=1.0, help="manyetik alan (T)")
    p1.add_argument("--pitch", type=float, default=60.0, help="hız ile B arasındaki açı (derece)")
    p1.add_argument("--adim-sayisi", type=int, default=50, help="jirasyon periyodu başına adım")
    p1.add_argument("--periyot-sayisi", type=int, default=300)
    p1.add_argument("--grafik", default=None)
    p1.set_defaults(func=_cmd_siklotron)

    p2 = sub.add_parser("capraz-alanlar", help="Çapraz E ve B alanları: E×B sürüklenmesi ve göreli değişmez")
    p2.add_argument("--parcacik", default="proton", choices=particles)
    p2.add_argument("--kinetik", type=float, default=0.01, help="MeV")
    p2.add_argument("--b", type=float, default=1.0, help="T")
    p2.add_argument("--e-orani", type=float, default=0.6, help="E/(cB), [0, 1)")
    p2.add_argument("--adim-sayisi", type=int, default=50)
    p2.add_argument("--periyot-sayisi", type=int, default=30)
    p2.add_argument("--grafik", default=None)
    p2.set_defaults(func=_cmd_capraz_alanlar)

    p3 = sub.add_parser("hiperbolik-hareket", help="Düzgün E alanında göreli (hiperbolik) hızlanma")
    p3.add_argument("--parcacik", default="proton", choices=particles)
    p3.add_argument("--e", type=float, default=1e8, help="elektrik alan (V/m)")
    p3.add_argument("--adim-sayisi", type=int, default=20, help="t_c başına adım")
    p3.add_argument("--sure-tc", type=float, default=20.0, help="t_c = mc/(qE) cinsinden süre")
    p3.add_argument("--grafik", default=None)
    p3.set_defaults(func=_cmd_hiperbolik_hareket)

    p4 = sub.add_parser("manyetik-ayna", help="Manyetik ayna: dönüm noktası ve adyabatik değişmez")
    p4.add_argument("--parcacik", default="proton", choices=particles)
    p4.add_argument("--kinetik", type=float, default=0.005, help="MeV")
    p4.add_argument("--b", type=float, default=1.0, help="B0 (T)")
    p4.add_argument("--uzunluk", type=float, default=1.0, help="L (m), B_z = B0(1 + z²/L²)")
    p4.add_argument("--pitch", type=float, default=40.0, help="z=0'daki pitch açısı α₀ (derece)")
    p4.add_argument("--adim-sayisi", type=int, default=40)
    p4.add_argument("--sicrama-sayisi", type=float, default=2.5, help="tahmini sıçrama sayısı (süre, orta düzlem paralel hızından hesaplanır; gerçek sıçrama biraz daha uzun sürer)")
    p4.add_argument("--grafik", default=None)
    p4.set_defaults(func=_cmd_manyetik_ayna)

    p5 = sub.add_parser("yakinsama", help="Boris ve RK4 yakınsama mertebesi (kesin siklotron çözümüne karşı)")
    p5.add_argument("--parcacik", default="elektron", choices=particles)
    p5.add_argument("--kinetik", type=float, default=1.0, help="MeV")
    p5.add_argument("--periyot-sayisi", type=int, default=10)
    p5.add_argument("--grafik", default=None)
    p5.set_defaults(func=_cmd_yakinsama)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ValueError as exc:
        print(f"Hata: {exc}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
