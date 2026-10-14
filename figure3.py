#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Figure 3 of the CHASE magnetic-levitation manuscript.

Figure 3:
    Representative in-orbit focal-plane pointing residuals during
    steady-state observation and raster scanning.

Required input files
--------------------
Preferred filenames:
    data/Data_S1_Steady_State_Pointing_Telemetry.xlsx
    data/Data_S2_Raster_Scanning_Pointing_Telemetry.xlsx

Legacy filenames are also recognized:
    data/SteadyState.xlsx
    data/Maneuvering.xlsx

Processing definition
---------------------
Steady-state observation:
    - attitude X/Y/Z: subtract temporal mean
    - angular rate X/Y/Z: subtract temporal mean

Raster scanning:
    - attitude X: remove a linear baseline
      because the commanded scan motion is about the payload X axis
    - attitude Y/Z: subtract temporal mean
    - angular rate X/Y/Z: subtract temporal mean

Robust dispersion:
    sigma_rob = 1.4826 * median(|x - median(x)|)
    reported metric = 3 * sigma_rob

The two-dimensional 3-sigma contours are descriptive robust covariance
ellipses constructed from the MAD-scaled X/Y dispersions and their
sample correlation coefficient.

Outputs
-------
    results/Figure3.png
    results/Figure3.pdf

Dependencies
------------
    numpy
    pandas
    scipy
    matplotlib
    openpyxl
"""

from __future__ import annotations

from pathlib import Path
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from scipy.stats import gaussian_kde


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Plot settings
# ============================================================================

COLOR_STEADY = "#00458A"
COLOR_RASTER = "#A3142F"
COLOR_ZERO = "0.65"

FONT_SIZE_AXIS = 9.5
FONT_SIZE_LABEL = 10.5
FONT_SIZE_LEGEND = 8.5


# ============================================================================
# Utilities
# ============================================================================

def first_existing_file(candidates: list[str]) -> Path:
    """Return the first existing file among candidate paths."""
    for name in candidates:
        path = DATA_DIR / name
        if path.exists():
            return path

    names = "\n  ".join(candidates)
    raise FileNotFoundError(
        "Could not locate the required source-data file.\n"
        f"Checked:\n  {names}"
    )


def find_column(df: pd.DataFrame, key: str) -> pd.Series:
    """
    Return the unique dataframe column whose name contains `key`.
    """
    matches = [col for col in df.columns if key in str(col)]

    if len(matches) == 0:
        raise KeyError(f'No column contains "{key}".')

    if len(matches) > 1:
        raise KeyError(
            f'Multiple columns contain "{key}": {matches}'
        )

    return pd.to_numeric(df[matches[0]], errors="coerce")


def mean_center(x: np.ndarray) -> np.ndarray:
    """Subtract the finite-sample temporal mean."""
    x = np.asarray(x, dtype=float)
    return x - np.nanmean(x)


def linear_detrend(x: np.ndarray) -> np.ndarray:
    """
    Remove an ordinary least-squares linear baseline from a 1-D record.
    Equivalent in intent to MATLAB detrend(x, 'linear').
    """
    x = np.asarray(x, dtype=float)
    t = np.arange(x.size, dtype=float)

    good = np.isfinite(x)

    if good.sum() < 2:
        raise ValueError("Too few finite samples for linear detrending.")

    p = np.polyfit(t[good], x[good], 1)
    baseline = np.polyval(p, t)

    return x - baseline


def robust_sigma_mad(x: np.ndarray) -> float:
    """
    MAD-based robust standard deviation:
        sigma_rob = 1.4826 * median(|x - median(x)|)
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]

    if x.size < 2:
        return np.nan

    med = np.median(x)
    mad = np.median(np.abs(x - med))

    return 1.4826 * mad


def robust_3sigma(x: np.ndarray) -> float:
    """Three times the MAD-based robust standard deviation."""
    return 3.0 * robust_sigma_mad(x)


def robust_ellipse(
    x: np.ndarray,
    y: np.ndarray,
    sigma_multiplier: float = 3.0,
    n_points: int = 400,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct the descriptive robust covariance ellipse used in Fig. 3.

    The diagonal dispersions are MAD-based robust standard deviations,
    while the off-diagonal term uses the ordinary sample correlation.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    sx = robust_sigma_mad(x)
    sy = robust_sigma_mad(y)

    if x.size < 2:
        raise ValueError("Too few samples for ellipse construction.")

    rho = np.corrcoef(x, y)[0, 1]

    covariance = np.array(
        [
            [sx**2, rho * sx * sy],
            [rho * sx * sy, sy**2],
        ],
        dtype=float,
    )

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    eigenvalues = np.maximum(eigenvalues, 0.0)

    theta = np.linspace(0.0, 2.0 * np.pi, n_points)

    unit_circle = np.vstack(
        [np.cos(theta), np.sin(theta)]
    )

    transform = (
        eigenvectors
        @ np.diag(
            sigma_multiplier * np.sqrt(eigenvalues)
        )
    )

    ellipse = transform @ unit_circle

    return ellipse[0], ellipse[1]


def safe_kde(
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Evaluate a 1-D Gaussian KDE with a safe degenerate fallback."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size < 2 or np.nanstd(values) == 0:
        return np.zeros_like(grid)

    kde = gaussian_kde(values)
    return kde(grid)


# ============================================================================
# Data loading and processing
# ============================================================================

def load_source_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    steady_file = first_existing_file(
        [
            "Data_S1_Steady_State_Pointing_Telemetry.xlsx",
            "Data_S1_Steady.xlsx",
            "SteadyState.xlsx",
        ]
    )

    raster_file = first_existing_file(
        [
            "Data_S2_Raster_Scanning_Pointing_Telemetry.xlsx",
            "Data_S2_Raster.xlsx",
            "Maneuvering.xlsx",
        ]
    )

    print(f"[Figure 3] Steady-state source: {steady_file.name}")
    print(f"[Figure 3] Raster source:       {raster_file.name}")

    steady = pd.read_excel(steady_file)
    raster = pd.read_excel(raster_file)

    return steady, raster


def process_data(
    steady: pd.DataFrame,
    raster: pd.DataFrame,
) -> dict[str, dict[str, np.ndarray]]:

    # ------------------------------------------------------------------
    # Absolute PM attitude [deg]
    # ------------------------------------------------------------------

    att_s = {
        "x": find_column(steady, "TMF0910").to_numpy(float),
        "y": find_column(steady, "TMF0911").to_numpy(float),
        "z": find_column(steady, "TMF0912").to_numpy(float),
    }

    att_r = {
        "x": find_column(raster, "TMF0910").to_numpy(float),
        "y": find_column(raster, "TMF0911").to_numpy(float),
        "z": find_column(raster, "TMF0912").to_numpy(float),
    }

    # ------------------------------------------------------------------
    # PM angular rate [deg/s]
    # ------------------------------------------------------------------

    rate_s = {
        "x": find_column(steady, "TMF0907").to_numpy(float),
        "y": find_column(steady, "TMF0908").to_numpy(float),
        "z": find_column(steady, "TMF0909").to_numpy(float),
    }

    rate_r = {
        "x": find_column(raster, "TMF0907").to_numpy(float),
        "y": find_column(raster, "TMF0908").to_numpy(float),
        "z": find_column(raster, "TMF0909").to_numpy(float),
    }

    # ------------------------------------------------------------------
    # Residual definitions
    # ------------------------------------------------------------------

    # Steady-state:
    # mean removal for all attitude and angular-rate channels.
    att_s_res = {
        axis: mean_center(att_s[axis])
        for axis in ("x", "y", "z")
    }

    rate_s_res = {
        axis: mean_center(rate_s[axis])
        for axis in ("x", "y", "z")
    }

    # Raster scanning:
    # commanded macroscopic attitude motion is about X.
    # Therefore only X attitude receives linear detrending.
    att_r_res = {
        "x": linear_detrend(att_r["x"]),
        "y": mean_center(att_r["y"]),
        "z": mean_center(att_r["z"]),
    }

    # Constant scan-rate baseline:
    # mean removal for all angular-rate channels.
    rate_r_res = {
        axis: mean_center(rate_r[axis])
        for axis in ("x", "y", "z")
    }

    return {
        "attitude": {
            "steady_x": att_s_res["x"],
            "steady_y": att_s_res["y"],
            "steady_z": att_s_res["z"],
            "raster_x": att_r_res["x"],
            "raster_y": att_r_res["y"],
            "raster_z": att_r_res["z"],
        },
        "rate": {
            "steady_x": rate_s_res["x"],
            "steady_y": rate_s_res["y"],
            "steady_z": rate_s_res["z"],
            "raster_x": rate_r_res["x"],
            "raster_y": rate_r_res["y"],
            "raster_z": rate_r_res["z"],
        },
    }


# ============================================================================
# Numerical summary
# ============================================================================

def print_summary(
    processed: dict[str, dict[str, np.ndarray]],
) -> None:

    def row(group: str, mode: str, scale: float) -> tuple[float, ...]:
        d = processed[group]

        x = robust_3sigma(d[f"{mode}_x"]) * scale
        y = robust_3sigma(d[f"{mode}_y"]) * scale
        z = robust_3sigma(d[f"{mode}_z"]) * scale

        xy = np.sqrt(x**2 + y**2)

        return x, y, xy, z

    # deg -> arcsec
    att_s = row("attitude", "steady", 3600.0)
    att_r = row("attitude", "raster", 3600.0)

    # deg/s -> arcsec/s
    rate_s = row("rate", "steady", 3600.0)
    rate_r = row("rate", "raster", 3600.0)

    print("\nFigure 3 / Table 1 robust 3-sigma summary")
    print("-------------------------------------------------------------")
    print("Attitude residuals [arcsec]")
    print(
        "  steady : "
        f"X={att_s[0]:.3f}, "
        f"Y={att_s[1]:.3f}, "
        f"X-Y={att_s[2]:.3f}, "
        f"Z={att_s[3]:.3f}"
    )
    print(
        "  raster : "
        f"X={att_r[0]:.3f}, "
        f"Y={att_r[1]:.3f}, "
        f"X-Y={att_r[2]:.3f}, "
        f"Z={att_r[3]:.3f}"
    )

    print("\nAngular-rate residuals [arcsec/s]")
    print(
        "  steady : "
        f"X={rate_s[0]:.3f}, "
        f"Y={rate_s[1]:.3f}, "
        f"X-Y={rate_s[2]:.3f}, "
        f"Z={rate_s[3]:.3f}"
    )
    print(
        "  raster : "
        f"X={rate_r[0]:.3f}, "
        f"Y={rate_r[1]:.3f}, "
        f"X-Y={rate_r[2]:.3f}, "
        f"Z={rate_r[3]:.3f}"
    )
    print("-------------------------------------------------------------")


# ============================================================================
# Plotting
# ============================================================================

def draw_residual_panel(
    fig: plt.Figure,
    subplot_spec,
    x_steady: np.ndarray,
    y_steady: np.ndarray,
    x_raster: np.ndarray,
    y_raster: np.ndarray,
    xlabel: str,
    ylabel: str,
    panel_label: str,
    show_ellipse: bool = True,
) -> None:

    # Display residuals in 10^-5 deg or 10^-5 deg/s.
    scale = 1e5

    xs = x_steady * scale
    ys = y_steady * scale

    xr = x_raster * scale
    yr = y_raster * scale

    # Robust 3-sigma ellipses are always calculated so that the
    # axis scaling remains consistent, but they are displayed only
    # when show_ellipse=True.
    exs, eys = robust_ellipse(
        xs,
        ys,
        sigma_multiplier=3.0,
    )

    exr, eyr = robust_ellipse(
        xr,
        yr,
        sigma_multiplier=3.0,
    )

    all_extent = np.concatenate(
        [
            np.abs(xs),
            np.abs(ys),
            np.abs(xr),
            np.abs(yr),
            np.abs(exs),
            np.abs(eys),
            np.abs(exr),
            np.abs(eyr),
        ]
    )

    lim = 1.12 * np.nanmax(all_extent)

    inner = GridSpecFromSubplotSpec(
        2,
        2,
        subplot_spec=subplot_spec,
        width_ratios=[4.7, 1.15],
        height_ratios=[1.15, 4.7],
        wspace=0.035,
        hspace=0.035,
    )

    ax_top = fig.add_subplot(inner[0, 0])
    ax_scatter = fig.add_subplot(inner[1, 0])
    ax_right = fig.add_subplot(inner[1, 1])

    # ------------------------------------------------------------------
    # Main scatter
    # ------------------------------------------------------------------

    raster_scatter = ax_scatter.scatter(
        xr,
        yr,
        s=8,
        c=COLOR_RASTER,
        alpha=0.25,
        linewidths=0,
        label="Raster scanning",
        zorder=2,
    )

    steady_scatter = ax_scatter.scatter(
        xs,
        ys,
        s=8,
        c=COLOR_STEADY,
        alpha=0.48,
        linewidths=0,
        label="Steady-state",
        zorder=3,
    )

    # Ellipses are shown only in the angular-rate panel.
    if show_ellipse:

        raster_ellipse, = ax_scatter.plot(
            exr,
            eyr,
            "--",
            color=COLOR_RASTER,
            linewidth=1.2,
            label=r"Raster-scanning $3\sigma$",
            zorder=4,
        )

        steady_ellipse, = ax_scatter.plot(
            exs,
            eys,
            "--",
            color=COLOR_STEADY,
            linewidth=1.2,
            label=r"Steady $3\sigma$",
            zorder=5,
        )

    # Zero-reference lines
    ax_scatter.axvline(
        0,
        linestyle=":",
        color=COLOR_ZERO,
        linewidth=0.9,
        zorder=1,
    )

    ax_scatter.axhline(
        0,
        linestyle=":",
        color=COLOR_ZERO,
        linewidth=0.9,
        zorder=1,
    )

    ax_scatter.set_xlim(-lim, lim)
    ax_scatter.set_ylim(-lim, lim)
    ax_scatter.set_aspect("equal", adjustable="box")

    ax_scatter.set_xlabel(
        xlabel,
        fontsize=FONT_SIZE_LABEL,
    )

    ax_scatter.set_ylabel(
        ylabel,
        fontsize=FONT_SIZE_LABEL,
    )

    ax_scatter.tick_params(
        direction="out",
        labelsize=FONT_SIZE_AXIS,
    )

    ax_scatter.grid(
        True,
        color="0.88",
        linewidth=0.6,
        alpha=0.7,
    )

    ax_scatter.spines["top"].set_visible(False)
    ax_scatter.spines["right"].set_visible(False)

    # ------------------------------------------------------------------
    # Top KDE
    # ------------------------------------------------------------------

    grid = np.linspace(-lim, lim, 400)

    kde_xr = safe_kde(xr, grid)
    kde_xs = safe_kde(xs, grid)

    ax_top.fill_between(
        grid,
        kde_xr,
        color=COLOR_RASTER,
        alpha=0.22,
    )

    ax_top.plot(
        grid,
        kde_xr,
        color=COLOR_RASTER,
        linewidth=1.1,
    )

    ax_top.fill_between(
        grid,
        kde_xs,
        color=COLOR_STEADY,
        alpha=0.38,
    )

    ax_top.plot(
        grid,
        kde_xs,
        color=COLOR_STEADY,
        linewidth=1.1,
    )

    ax_top.set_xlim(-lim, lim)
    ax_top.set_xticks([])
    ax_top.set_yticks([])

    for spine in ax_top.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------------
    # Right KDE
    # ------------------------------------------------------------------

    kde_yr = safe_kde(yr, grid)
    kde_ys = safe_kde(ys, grid)

    ax_right.fill_betweenx(
        grid,
        0,
        kde_yr,
        color=COLOR_RASTER,
        alpha=0.22,
    )

    ax_right.plot(
        kde_yr,
        grid,
        color=COLOR_RASTER,
        linewidth=1.1,
    )

    ax_right.fill_betweenx(
        grid,
        0,
        kde_ys,
        color=COLOR_STEADY,
        alpha=0.38,
    )

    ax_right.plot(
        kde_ys,
        grid,
        color=COLOR_STEADY,
        linewidth=1.1,
    )

    ax_right.set_ylim(-lim, lim)
    ax_right.set_xticks([])
    ax_right.set_yticks([])

    for spine in ax_right.spines.values():
        spine.set_visible(False)

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    if show_ellipse:

        handles = [
            steady_scatter,
            raster_scatter,
            steady_ellipse,
            raster_ellipse,
        ]

        labels = [
            "Steady-state",
            "Raster scanning",
            r"Steady $3\sigma$",
            r"Raster-scanning $3\sigma$",
        ]

        # Push the legend as far as practical toward the bottom-right
        # corner to reduce overlap with the point cloud.
        ax_scatter.legend(
            handles,
            labels,
            loc="lower right",
            bbox_to_anchor=(1.08, -0.01),
            borderaxespad=0.15,
            frameon=False,
            fontsize=7.8,
            handlelength=1.7,
            handletextpad=0.45,
            labelspacing=0.25,
        )

    else:

        # Panel (a): only the two operating modes are shown.
        ax_scatter.legend(
            [steady_scatter, raster_scatter],
            ["Steady-state", "Raster scanning"],
            loc="lower right",
            bbox_to_anchor=(0.995, 0.01),
            borderaxespad=0.15,
            frameon=False,
            fontsize=8.5,
            handlelength=1.2,
            handletextpad=0.45,
            labelspacing=0.25,
        )

    # ------------------------------------------------------------------
    # Panel label
    # ------------------------------------------------------------------

    ax_top.text(
        -0.105,
        1.02,
        panel_label,
        transform=ax_top.transAxes,
        fontsize=12,
        fontweight="bold",
        va="bottom",
        ha="left",
    )

def make_figure(
    processed: dict[str, dict[str, np.ndarray]],
) -> plt.Figure:

    fig = plt.figure(
        figsize=(11.0, 5.2),
        facecolor="white",
    )

    # Reduced spacing between panels (a) and (b).
    outer = GridSpec(
        1,
        2,
        figure=fig,
        width_ratios=[1, 1],
        wspace=0.21,
        left=0.065,
        right=0.985,
        bottom=0.13,
        top=0.94,
    )

    # ------------------------------------------------------------------
    # Panel (a): attitude residuals
    # No 3-sigma ellipses, matching the manuscript figure.
    # ------------------------------------------------------------------

    att = processed["attitude"]

    draw_residual_panel(
        fig=fig,
        subplot_spec=outer[0],
        x_steady=att["steady_x"],
        y_steady=att["steady_y"],
        x_raster=att["raster_x"],
        y_raster=att["raster_y"],
        xlabel=r"Scan-axis attitude residual, $\Delta\theta_x$ ($10^{-5}$ deg)",
        ylabel=r"Cross-scan attitude residual, $\Delta\theta_y$ ($10^{-5}$ deg)",
        panel_label="a",
        show_ellipse=False,
    )

    # ------------------------------------------------------------------
    # Panel (b): angular-rate residuals
    # Retain robust 3-sigma ellipses.
    # ------------------------------------------------------------------

    rate = processed["rate"]

    draw_residual_panel(
        fig=fig,
        subplot_spec=outer[1],
        x_steady=rate["steady_x"],
        y_steady=rate["steady_y"],
        x_raster=rate["raster_x"],
        y_raster=rate["raster_y"],
        xlabel=r"Scan-axis rate residual, $\omega_x$ ($10^{-5}$ deg/s)",
        ylabel=r"Cross-scan rate residual, $\omega_y$ ($10^{-5}$ deg/s)",
        panel_label="b",
        show_ellipse=True,
    )

    return fig


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    steady, raster = load_source_data()

    processed = process_data(
        steady=steady,
        raster=raster,
    )

    print_summary(processed)

    fig = make_figure(processed)

    png_file = RESULTS_DIR / "Figure3.png"
    pdf_file = RESULTS_DIR / "Figure3.pdf"

    fig.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
    )

    plt.close(fig)

    print("\nSaved:")
    print(f"  {png_file}")
    print(f"  {pdf_file}")


if __name__ == "__main__":
    main()