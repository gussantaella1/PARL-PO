#!/usr/bin/env python3
"""Plot the HCW and Elliptic LTV chief orbits used by the evaluations."""

from __future__ import annotations

import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from config_rl import COMMON


EARTH_RADIUS_M = 6_371_000.0


def main() -> None:
    cache_dir = Path(tempfile.gettempdir()) / "research_repo_matplotlib_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_dir))

    import matplotlib.pyplot as plt

    out_dir = REPO_ROOT / "LaTeX_Tables" / "Merged_Thesis"
    out_dir.mkdir(parents=True, exist_ok=True)

    hcw = COMMON["hcw"]
    orbit = COMMON["chief_orbit"]

    r_hcw = float(hcw["r0"])
    a = float(orbit["a"])
    e = float(orbit["e"])
    mu = float(orbit["mu"])

    theta = np.linspace(0.0, 2.0 * np.pi, 1200)
    x_hcw = r_hcw * np.cos(theta)
    y_hcw = r_hcw * np.sin(theta)

    p = a * (1.0 - e * e)
    r_ell = p / (1.0 + e * np.cos(theta))
    x_ell = r_ell * np.cos(theta)
    y_ell = r_ell * np.sin(theta)

    rp = a * (1.0 - e)
    ra = a * (1.0 + e)
    h_hcw = r_hcw - EARTH_RADIUS_M
    h_perigee = rp - EARTH_RADIUS_M
    h_apogee = ra - EARTH_RADIUS_M
    n_hcw = math.sqrt(float(hcw["mu"]) / r_hcw**3)
    t_hcw = 2.0 * math.pi / n_hcw
    t_ell = 2.0 * math.pi * math.sqrt(a**3 / mu)

    km = 1.0 / 1000.0
    fig, ax = plt.subplots(figsize=(8.2, 7.2))

    earth = plt.Circle(
        (0.0, 0.0),
        EARTH_RADIUS_M * km,
        facecolor="#d9eef8",
        edgecolor="#4f7f98",
        linewidth=1.1,
        zorder=1,
    )
    ax.add_patch(earth)

    ax.plot(x_hcw * km, y_hcw * km, color="#2f6fbb", linewidth=2.0, label="HCW circular chief orbit")
    ax.plot(x_ell * km, y_ell * km, color="#b8463a", linewidth=2.0, label="Elliptic LTV chief orbit")

    ax.scatter([rp * km], [0.0], color="#b8463a", s=36, zorder=4)
    ax.scatter([-ra * km], [0.0], color="#b8463a", s=36, zorder=4)
    ax.scatter([r_hcw * km], [0.0], color="#2f6fbb", s=36, zorder=4)

    ax.annotate(
        f"Elliptic perigee\n{h_perigee * km:.1f} km alt.",
        xy=(rp * km, 0.0),
        xytext=(rp * km + 330.0, 550.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#555555"},
        fontsize=9,
        ha="left",
    )
    ax.annotate(
        f"Elliptic apogee\n{h_apogee * km:.1f} km alt.",
        xy=(-ra * km, 0.0),
        xytext=(-ra * km - 1750.0, -720.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#555555"},
        fontsize=9,
        ha="left",
    )
    ax.annotate(
        f"HCW circular\n{h_hcw * km:.1f} km alt.",
        xy=(r_hcw * km, 0.0),
        xytext=(r_hcw * km + 290.0, -720.0),
        arrowprops={"arrowstyle": "->", "linewidth": 0.9, "color": "#555555"},
        fontsize=9,
        ha="left",
    )

    info = (
        "Parameters used in config_rl.py\n"
        f"HCW: r0 = {r_hcw * km:.0f} km, T = {t_hcw / 60.0:.2f} min\n"
        f"Elliptic LTV: a = {a * km:.0f} km, e = {e:.4f}, T = {t_ell / 60.0:.2f} min"
    )
    ax.text(
        0.02,
        0.02,
        info,
        transform=ax.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )

    lim = max(ra, r_hcw) * km * 1.17
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Inertial x (km)")
    ax.set_ylabel("Inertial y (km)")
    ax.set_title("Chief Orbit Geometry Used for HCW and Elliptic LTV Evaluations", pad=12)
    ax.grid(True, color="#dddddd", linewidth=0.8)
    ax.legend(loc="upper right", frameon=False)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    png_path = out_dir / "chief_orbit_hcw_vs_elliptic_ltv.png"
    pdf_path = out_dir / "chief_orbit_hcw_vs_elliptic_ltv.pdf"
    fig.tight_layout()
    fig.savefig(png_path, dpi=240)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"Wrote: {png_path}")
    print(f"Wrote: {pdf_path}")


if __name__ == "__main__":
    main()
