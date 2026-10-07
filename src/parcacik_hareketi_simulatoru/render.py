"""Plotting helpers. Each function writes a PNG and returns nothing."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_cyclotron(boris_xy, exact_xy, periods, boris_drift, rk4_drift, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].plot(exact_xy[:, 0] * 1e3, exact_xy[:, 1] * 1e3, "k", lw=2.5, alpha=0.35, label="kesin")
    axes[0].plot(boris_xy[:, 0] * 1e3, boris_xy[:, 1] * 1e3, "r", lw=0.8, label="Boris")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("x (mm)")
    axes[0].set_ylabel("y (mm)")
    axes[0].set_title("Siklotron yörüngesi (ilk periyotlar)")
    axes[0].legend()
    axes[1].plot(periods, np.abs(boris_drift) + 1e-18, label="Boris")
    axes[1].plot(periods, np.abs(rk4_drift) + 1e-18, label="RK4")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("jirasyon periyodu sayısı")
    axes[1].set_ylabel("|Δ|u|/|u|₀|")
    axes[1].set_title("Enerji kayması")
    axes[1].grid(True, which="both", alpha=0.3)
    axes[1].legend()
    _save(fig, path)


def plot_crossed_fields(x, y, drift_x, drift_y, path: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(x, y, lw=0.9, label="yörünge")
    ax.plot(drift_x, drift_y, "r--", lw=1.5, label="E×B sürüklenmesi (öngörülen kılavuz merkez)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Çapraz E ve B alanlarında hareket")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    _save(fig, path)


def plot_hyperbolic(t_over_tc, v_numeric, v_exact, v_newton, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(t_over_tc, v_newton, "--", label="Newton (v = at)")
    ax.plot(t_over_tc, v_exact, label="göreli, kesin")
    ax.plot(t_over_tc, v_numeric, "o", ms=3, label="Boris")
    ax.axhline(1.0, color="red", ls=":", lw=1)
    ax.set_ylim(0, 1.6)
    ax.set_xlabel("t / t_c,  t_c = mc/(qE)")
    ax.set_ylabel("v / c")
    ax.set_title("Düzgün elektrik alanda hızlanma")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save(fig, path)


def plot_mirror(t, z, z_turn, mu_rel, path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(t * 1e6, z, lw=0.9)
    axes[0].axhline(z_turn, color="r", ls=":", label="öngörülen dönüm noktası ±L·cot(α₀)")
    axes[0].axhline(-z_turn, color="r", ls=":")
    axes[0].set_xlabel("t (µs)")
    axes[0].set_ylabel("z (m)")
    axes[0].set_title("Manyetik ayna içinde sıçrama")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(t * 1e6, mu_rel, lw=0.9)
    axes[1].set_xlabel("t (µs)")
    axes[1].set_ylabel("μ / μ₀ - 1")
    axes[1].set_title("Adyabatik değişmez p⊥²/(2mB)")
    axes[1].grid(True, alpha=0.3)
    _save(fig, path)


def plot_convergence(step_sizes, boris_errors, rk4_errors, path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.loglog(step_sizes, boris_errors, "o-", label="Boris")
    ax.loglog(step_sizes, rk4_errors, "s-", label="RK4")
    ax.set_xlabel("adım dt (jirasyon periyodu cinsinden)")
    ax.set_ylabel("azami konum hatası / jirasyon yarıçapı")
    ax.set_title("Yakınsama: kesin siklotron çözümüne karşı")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    _save(fig, path)
