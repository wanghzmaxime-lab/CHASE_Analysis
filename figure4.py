#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Figure 4 of the CHASE magnetic-levitation manuscript.

Figure 4:
    Spectral separation of Payload Module (PM) and Support Module (SM)
    three-axis absolute-attitude motion during:
        (a) steady-state observation
        (b) raster scanning

Required input files
--------------------
Preferred filenames:
    data/Data_S1_Steady_State_Pointing_Telemetry.xlsx
    data/Data_S2_Raster_Scanning_Pointing_Telemetry.xlsx

Legacy filenames are also recognized:
    data/SteadyState.xlsx
    data/Maneuvering.xlsx

Scientific processing
---------------------
1. PM absolute attitude is taken directly from telemetry.

2. The measured relative attitude describes the Support Module (SM)
   with respect to the Payload Module (PM).

3. SM absolute attitude is reconstructed using finite rotation matrices:

       T_i^sb = (T_sb^pb)^T @ T_i^pb

   where:
       T_i^pb   : inertial frame -> PM body frame
       T_sb^pb  : SM body frame -> PM body frame
       T_i^sb   : inertial frame -> SM body frame

   The same passive Z-Y-X Euler convention as the SI Appendix is used:

       T_i^b = R_x(phi) @ R_y(theta) @ R_z(psi)

4. A linear trend is removed independently from every PM and
   reconstructed-SM attitude component before PSD estimation.

5. Axis-resolved PSDs are estimated using Welch's method:
       sampling frequency = 8 Hz
       FFT length         = 256 points

   The segmentation/window settings reproduce the MATLAB call

       pwelch(x, [], [], 256, 8)

   as closely as possible:
       Hamming window
       approximately 8 segments
       50% overlap

6. Three-axis total attitude PSD:

       P_total = P_x + P_y + P_z

7. Zero frequency is excluded.

8. Below 0.5 Hz, direct Welch estimates are retained.
   At and above 0.5 Hz, log10(PSD) is smoothed using an
   11-point quadratic Savitzky-Golay filter.

9. Spectral separation is calculated directly from the processed
   PSD traces:

       DeltaP = 10 log10(P_SM / P_PM)

   Positive values indicate greater spectral power in the SM.

Interpretation
--------------
DeltaP is a descriptive spectral-separation metric. It is not an
input-output transfer function, transmissibility, or causal
attenuation ratio.

Outputs
-------
    results/Figure4.png
    results/Figure4.pdf

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

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Non-GUI backend: avoids Qt/Tk dependencies.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from scipy.signal import (
    detrend,
    savgol_filter,
    welch,
)


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}\n"
        "Please place the source-data files in a 'data' folder "
        "next to figure4.py."
    )

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Analysis parameters
# ============================================================================

FS = 8.0
NFFT = 256

SMOOTH_BOUNDARY_HZ = 0.5
SMOOTH_WINDOW = 11
SMOOTH_POLYORDER = 2

PLOT_MAX_HZ = 4.0
SUMMARY_MIN_HZ = 0.1
SUMMARY_MAX_HZ = 4.0


# ============================================================================
# Plot settings
# ============================================================================

COLOR_SM = "#808080"
COLOR_PM = "#00458A"
COLOR_SEPARATION = "#B3262E"
COLOR_FILL = "#73A9D8"

FONT_SIZE_AXIS = 9.5
FONT_SIZE_LABEL = 10.5
FONT_SIZE_TITLE = 11.0
FONT_SIZE_LEGEND = 8.5


# ============================================================================
# File and column utilities
# ============================================================================

def first_existing_file(candidates: list[str]) -> Path:
    """Return the first existing file among candidate filenames."""
    for name in candidates:
        path = DATA_DIR / name

        if path.exists():
            return path

    names = "\n  ".join(candidates)

    raise FileNotFoundError(
        "Could not locate the required source-data file.\n"
        f"Checked:\n  {names}"
    )


def find_column(df: pd.DataFrame, key: str) -> np.ndarray:
    """
    Return the unique numeric dataframe column whose name contains `key`.
    """
    matches = [
        col for col in df.columns
        if key in str(col)
    ]

    if len(matches) == 0:
        raise KeyError(
            f'No column contains "{key}".'
        )

    if len(matches) > 1:
        raise KeyError(
            f'Multiple columns contain "{key}": {matches}'
        )

    values = pd.to_numeric(
        df[matches[0]],
        errors="coerce",
    ).to_numpy(dtype=float)

    if not np.all(np.isfinite(values)):
        raise ValueError(
            f'Column "{matches[0]}" contains non-finite values.'
        )

    return values


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

    print(f"[Figure 4] Steady-state source: {steady_file.name}")
    print(f"[Figure 4] Raster source:       {raster_file.name}")

    steady = pd.read_excel(steady_file)
    raster = pd.read_excel(raster_file)

    return steady, raster


# ============================================================================
# Passive Z-Y-X rotation convention used in the SI Appendix
# ============================================================================

def rotation_x(phi: float) -> np.ndarray:
    """
    Passive coordinate transformation for rotation about X.

    This convention is consistent with

        T_i^b = R_x(phi) R_y(theta) R_z(psi)

    and the Euler extraction formulas used in the SI.
    """
    c = np.cos(phi)
    s = np.sin(phi)

    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, c,   s],
            [0.0, -s,  c],
        ]
    )


def rotation_y(theta: float) -> np.ndarray:
    """Passive coordinate transformation for rotation about Y."""
    c = np.cos(theta)
    s = np.sin(theta)

    return np.array(
        [
            [c,   0.0, -s],
            [0.0, 1.0, 0.0],
            [s,   0.0, c],
        ]
    )


def rotation_z(psi: float) -> np.ndarray:
    """Passive coordinate transformation for rotation about Z."""
    c = np.cos(psi)
    s = np.sin(psi)

    return np.array(
        [
            [c,   s,   0.0],
            [-s,  c,   0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def euler_zyx_to_transform(
    phi: float,
    theta: float,
    psi: float,
) -> np.ndarray:
    """
    Convert Z-Y-X Euler angles to the passive transformation matrix

        T = R_x(phi) @ R_y(theta) @ R_z(psi)

    Angles are in radians.
    """
    return (
        rotation_x(phi)
        @ rotation_y(theta)
        @ rotation_z(psi)
    )


def transform_to_euler_zyx(
    T: np.ndarray,
) -> np.ndarray:
    """
    Extract Z-Y-X Euler angles from the passive transformation matrix.

    The formulas follow the convention used in the SI:

        phi   = atan2(T23, T33)
        theta = -asin(T13)
        psi   = atan2(T12, T11)

    Returns
    -------
    ndarray
        [phi, theta, psi] in radians.
    """

    # Numerical protection against a tiny floating-point excursion
    # outside [-1, 1].
    s_theta = np.clip(
        T[0, 2],
        -1.0,
        1.0,
    )

    theta = -np.arcsin(s_theta)

    phi = np.arctan2(
        T[1, 2],
        T[2, 2],
    )

    psi = np.arctan2(
        T[0, 1],
        T[0, 0],
    )

    return np.array(
        [phi, theta, psi],
        dtype=float,
    )


# ============================================================================
# SM absolute-attitude reconstruction
# ============================================================================

def reconstruct_sm_attitude(
    pm_euler_deg: np.ndarray,
    relative_euler_deg: np.ndarray,
) -> np.ndarray:
    """
    Reconstruct SM absolute attitude from PM absolute attitude and
    SM-with-respect-to-PM relative attitude.

    Parameters
    ----------
    pm_euler_deg
        N x 3 array:
            [phi_PM, theta_PM, psi_PM] in degrees.

    relative_euler_deg
        N x 3 array describing SM with respect to PM:
            [phi_s/p, theta_s/p, psi_s/p] in degrees.

    Returns
    -------
    N x 3 ndarray
        Reconstructed SM absolute Z-Y-X Euler angles in degrees.

    Relation
    --------
        T_i^sb = (T_sb^pb)^T @ T_i^pb
    """

    pm_euler_deg = np.asarray(
        pm_euler_deg,
        dtype=float,
    )

    relative_euler_deg = np.asarray(
        relative_euler_deg,
        dtype=float,
    )

    if pm_euler_deg.shape != relative_euler_deg.shape:
        raise ValueError(
            "PM and relative attitude arrays must have the same shape."
        )

    if (
        pm_euler_deg.ndim != 2
        or pm_euler_deg.shape[1] != 3
    ):
        raise ValueError(
            "Attitude arrays must have shape (N, 3)."
        )

    n_samples = pm_euler_deg.shape[0]

    sm_euler_rad = np.empty(
        (n_samples, 3),
        dtype=float,
    )

    pm_rad = np.deg2rad(pm_euler_deg)
    rel_rad = np.deg2rad(relative_euler_deg)

    for k in range(n_samples):

        phi_p, theta_p, psi_p = pm_rad[k]
        phi_rel, theta_rel, psi_rel = rel_rad[k]

        # IF -> PM-BF
        T_i_pb = euler_zyx_to_transform(
            phi_p,
            theta_p,
            psi_p,
        )

        # SM-BF -> PM-BF
        T_sb_pb = euler_zyx_to_transform(
            phi_rel,
            theta_rel,
            psi_rel,
        )

        # IF -> SM-BF
        T_i_sb = T_sb_pb.T @ T_i_pb

        sm_euler_rad[k] = transform_to_euler_zyx(
            T_i_sb
        )

    # Remove possible ±pi representation jumps before spectral processing.
    sm_euler_rad = np.unwrap(
        sm_euler_rad,
        axis=0,
    )

    return np.rad2deg(sm_euler_rad)


# ============================================================================
# Extract attitude channels
# ============================================================================

def extract_attitude_arrays(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract PM absolute attitude and SM-with-respect-to-PM relative attitude.

    Returns
    -------
    pm : N x 3
        PM absolute attitude [X, Y, Z] in degrees.

    rel : N x 3
        Relative attitude of SM with respect to PM [X, Y, Z] in degrees.
    """

    pm = np.column_stack(
        [
            find_column(df, "TMF0910"),
            find_column(df, "TMF0911"),
            find_column(df, "TMF0912"),
        ]
    )

    rel = np.column_stack(
        [
            find_column(df, "TMF0913"),
            find_column(df, "TMF0914"),
            find_column(df, "TMF0915"),
        ]
    )

    return pm, rel


# ============================================================================
# MATLAB-compatible Welch PSD
# ============================================================================

def matlab_pwelch_psd(
    x: np.ndarray,
    fs: float = FS,
    nfft: int = NFFT,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Reproduce the processing intent of MATLAB:

        pwelch(detrend(x), [], [], nfft, fs)

    MATLAB's default empty-window behavior divides the signal into
    approximately eight Hamming-windowed sections with 50% overlap.

    For a record of length N:
        nperseg = floor(N / 4.5)
        noverlap = floor(nperseg / 2)

    The global linear trend is removed before Welch estimation.
    """

    x = np.asarray(
        x,
        dtype=float,
    )

    if not np.all(np.isfinite(x)):
        raise ValueError(
            "PSD input contains non-finite samples."
        )

    # MATLAB detrend(x): remove best-fit straight line.
    x_detrended = detrend(
        x,
        type="linear",
    )

    n = x_detrended.size

    nperseg = int(
        np.floor(n / 4.5)
    )

    if nperseg < 3:
        raise ValueError(
            "Telemetry record is too short for Welch estimation."
        )

    noverlap = int(
        np.floor(nperseg / 2)
    )

    # MATLAB hamming(n) is symmetric.
    window = np.hamming(
        nperseg
    )

    f, pxx = welch(
        x_detrended,
        fs=fs,
        window=window,
        nperseg=nperseg,
        noverlap=noverlap,
        nfft=nfft,
        detrend=False,
        return_onesided=True,
        scaling="density",
        average="mean",
    )

    return f, pxx


def total_three_axis_psd(
    attitude_deg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the summed X/Y/Z absolute-attitude PSD:

        P_total = P_X + P_Y + P_Z
    """

    attitude_deg = np.asarray(
        attitude_deg,
        dtype=float,
    )

    spectra = []

    f_reference = None

    for axis in range(3):

        f, pxx = matlab_pwelch_psd(
            attitude_deg[:, axis]
        )

        if f_reference is None:
            f_reference = f
        else:
            if not np.allclose(
                f,
                f_reference,
            ):
                raise RuntimeError(
                    "Welch frequency grids are inconsistent."
                )

        spectra.append(
            pxx
        )

    total = (
        spectra[0]
        + spectra[1]
        + spectra[2]
    )

    return f_reference, total


# ============================================================================
# PSD smoothing
# ============================================================================

def smooth_total_psd(
    f: np.ndarray,
    psd: np.ndarray,
) -> np.ndarray:
    """
    Apply the manuscript-defined piecewise smoothing.

    f < 0.5 Hz:
        retain direct Welch PSD

    f >= 0.5 Hz:
        smooth log10(PSD) with an 11-point quadratic
        Savitzky-Golay filter
    """

    f = np.asarray(
        f,
        dtype=float,
    )

    psd = np.asarray(
        psd,
        dtype=float,
    )

    if np.any(psd <= 0):
        raise ValueError(
            "PSD contains non-positive values; log-domain smoothing "
            "cannot be applied."
        )

    log_psd = np.log10(
        psd
    )

    smooth_log_psd = savgol_filter(
        log_psd,
        window_length=SMOOTH_WINDOW,
        polyorder=SMOOTH_POLYORDER,
        mode="interp",
    )

    result = psd.copy()

    high_frequency = (
        f >= SMOOTH_BOUNDARY_HZ
    )

    result[high_frequency] = (
        10.0
        ** smooth_log_psd[high_frequency]
    )

    return result


# ============================================================================
# Process one operating mode
# ============================================================================

def process_mode(
    df: pd.DataFrame,
) -> dict[str, np.ndarray]:

    pm_attitude, relative_attitude = (
        extract_attitude_arrays(df)
    )

    sm_attitude = reconstruct_sm_attitude(
        pm_euler_deg=pm_attitude,
        relative_euler_deg=relative_attitude,
    )

    # PM telemetry can also be unwrapped in angle space before detrending.
    pm_attitude_unwrapped = np.rad2deg(
        np.unwrap(
            np.deg2rad(pm_attitude),
            axis=0,
        )
    )

    f_pm, pm_psd = total_three_axis_psd(
        pm_attitude_unwrapped
    )

    f_sm, sm_psd = total_three_axis_psd(
        sm_attitude
    )

    if not np.allclose(
        f_pm,
        f_sm,
    ):
        raise RuntimeError(
            "PM and SM Welch frequency grids are inconsistent."
        )

    # Remove zero frequency.
    nonzero = (
        f_pm > 0
    )

    f = f_pm[nonzero]
    pm_psd = pm_psd[nonzero]
    sm_psd = sm_psd[nonzero]

    # Piecewise SG smoothing.
    pm_psd_processed = smooth_total_psd(
        f,
        pm_psd,
    )

    sm_psd_processed = smooth_total_psd(
        f,
        sm_psd,
    )

    # No additional smoothing is applied to this curve.
    separation_db = (
        10.0
        * np.log10(
            sm_psd_processed
            / pm_psd_processed
        )
    )

    return {
        "frequency": f,
        "pm_psd": pm_psd_processed,
        "sm_psd": sm_psd_processed,
        "separation_db": separation_db,
    }


# ============================================================================
# Numerical summary
# ============================================================================

def print_mode_summary(
    name: str,
    result: dict[str, np.ndarray],
) -> None:

    f = result["frequency"]
    sep = result["separation_db"]

    band = (
        (f >= SUMMARY_MIN_HZ)
        & (f <= SUMMARY_MAX_HZ)
    )

    fb = f[band]
    sb = sep[band]

    print(f"\n{name}, {SUMMARY_MIN_HZ:.1f}-{SUMMARY_MAX_HZ:.1f} Hz")
    print(
        f"  Median spectral separation : "
        f"{np.median(sb):.2f} dB"
    )
    print(
        f"  10th-90th percentile       : "
        f"{np.percentile(sb, 10):.2f} to "
        f"{np.percentile(sb, 90):.2f} dB"
    )

    idx = np.argmax(sb)

    print(
        f"  Maximum spectral separation: "
        f"{sb[idx]:.2f} dB at {fb[idx]:.3f} Hz"
    )


# ============================================================================
# Plotting
# ============================================================================

def make_figure(
    steady: dict[str, np.ndarray],
    raster: dict[str, np.ndarray],
) -> plt.Figure:

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.35),
        facecolor="white",
    )

    plt.subplots_adjust(
        left=0.080,
        right=0.925,
        bottom=0.145,
        top=0.900,
        wspace=0.29,
    )

    results = [
        steady,
        raster,
    ]

    titles = [
        "Steady-state observation",
        "Raster scanning",
    ]

    # ------------------------------------------------------------------
    # Common left-axis PSD range
    # ------------------------------------------------------------------

    all_psd = np.concatenate(
        [
            steady["pm_psd"],
            steady["sm_psd"],
            raster["pm_psd"],
            raster["sm_psd"],
        ]
    )

    positive_psd = all_psd[
        np.isfinite(all_psd)
        & (all_psd > 0)
    ]

    psd_min = positive_psd.min()
    psd_max = positive_psd.max()

    # Give modest logarithmic padding.
    log_min = np.log10(psd_min)
    log_max = np.log10(psd_max)

    psd_lower = 10.0 ** (
        log_min - 0.08 * (log_max - log_min)
    )

    psd_upper = 10.0 ** (
        log_max + 0.08 * (log_max - log_min)
    )

    # ------------------------------------------------------------------
    # Common right-axis separation range
    # ------------------------------------------------------------------

    all_sep = np.concatenate(
        [
            steady["separation_db"],
            raster["separation_db"],
        ]
    )

    sep_min = np.nanmin(all_sep)
    sep_max = np.nanmax(all_sep)

    sep_span = sep_max - sep_min

    if sep_span <= 0:
        sep_span = 1.0

    sep_lower = sep_min - 0.08 * sep_span
    sep_upper = sep_max + 0.08 * sep_span

    # Always include zero for physical interpretation.
    sep_lower = min(
        sep_lower,
        0.0,
    )

    sep_upper = max(
        sep_upper,
        0.0,
    )

    # ------------------------------------------------------------------
    # Draw panels
    # ------------------------------------------------------------------

    for panel_index, (
        ax,
        result,
        title,
    ) in enumerate(
        zip(
            axes,
            results,
            titles,
        )
    ):

        f = result["frequency"]
        p_pm = result["pm_psd"]
        p_sm = result["sm_psd"]
        sep = result["separation_db"]

        # PSD separation region.
        ax.fill_between(
            f,
            p_pm,
            p_sm,
            color=COLOR_FILL,
            alpha=0.16,
            linewidth=0,
            zorder=1,
        )

        sm_line, = ax.plot(
            f,
            p_sm,
            color=COLOR_SM,
            linewidth=1.35,
            label="Support Module",
            zorder=3,
        )

        pm_line, = ax.plot(
            f,
            p_pm,
            color=COLOR_PM,
            linewidth=1.9,
            label="Payload Module",
            zorder=4,
        )

        ax.set_xscale(
            "log"
        )

        ax.set_yscale(
            "log"
        )

        ax.set_xlim(
            f.min(),
            PLOT_MAX_HZ,
        )

        ax.set_ylim(
            psd_lower,
            psd_upper,
        )

        ax.set_xlabel(
            "Frequency (Hz)",
            fontsize=FONT_SIZE_LABEL,
        )

        ax.set_ylabel(
            r"Three-axis total attitude PSD (deg$^2$/Hz)",
            fontsize=FONT_SIZE_LABEL,
        )

        ax.set_title(
            title,
            fontsize=FONT_SIZE_TITLE,
            fontweight="normal",
            pad=7,
        )

        ax.tick_params(
            axis="both",
            which="both",
            direction="in",
            labelsize=FONT_SIZE_AXIS,
        )

        ax.grid(
            True,
            which="major",
            color="0.85",
            linewidth=0.6,
            alpha=0.65,
        )

        ax.grid(
            True,
            which="minor",
            color="0.92",
            linewidth=0.4,
            alpha=0.40,
        )

        # --------------------------------------------------------------
        # Right axis: spectral separation
        # --------------------------------------------------------------

        ax_sep = ax.twinx()

        sep_line, = ax_sep.plot(
            f,
            sep,
            "--",
            color=COLOR_SEPARATION,
            linewidth=1.45,
            label="Spectral separation",
            zorder=5,
        )

        ax_sep.set_xscale(
            "log"
        )

        ax_sep.set_ylim(
            sep_lower,
            sep_upper,
        )

        ax_sep.set_ylabel(
            r"Spectral separation, "
            r"$10\log_{10}(P_{\rm SM}/P_{\rm PM})$ (dB)",
            fontsize=FONT_SIZE_LABEL,
            color=COLOR_SEPARATION,
        )

        ax_sep.tick_params(
            axis="y",
            direction="in",
            labelsize=FONT_SIZE_AXIS,
            colors=COLOR_SEPARATION,
        )

        ax_sep.spines["right"].set_color(
            COLOR_SEPARATION
        )

        # Zero-separation reference.
        if sep_lower < 0 < sep_upper:
            ax_sep.axhline(
                0.0,
                color=COLOR_SEPARATION,
                linewidth=0.65,
                linestyle=":",
                alpha=0.45,
                zorder=2,
            )

        # --------------------------------------------------------------
        # Legend only in the left panel, matching the manuscript style.
        # --------------------------------------------------------------

        if panel_index == 0:

            ax.legend(
                handles=[
                    sm_line,
                    pm_line,
                    sep_line,
                ],
                labels=[
                    "Support Module",
                    "Payload Module",
                    "Spectral separation",
                ],
                loc="lower left",
                frameon=False,
                fontsize=FONT_SIZE_LEGEND,
                handlelength=2.1,
                labelspacing=0.30,
            )

        # Panel labels.
        ax.text(
            -0.16,
            1.04,
            "a" if panel_index == 0 else "b",
            transform=ax.transAxes,
            fontsize=12,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

    return fig


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    steady_df, raster_df = (
        load_source_data()
    )

    print(
        "\n[Figure 4] Reconstructing SM absolute attitude "
        "using rotation matrices..."
    )

    steady_result = process_mode(
        steady_df
    )

    raster_result = process_mode(
        raster_df
    )

    print_mode_summary(
        "Steady-state observation",
        steady_result,
    )

    print_mode_summary(
        "Raster scanning",
        raster_result,
    )

    fig = make_figure(
        steady=steady_result,
        raster=raster_result,
    )

    png_file = (
        RESULTS_DIR
        / "Figure4.png"
    )

    pdf_file = (
        RESULTS_DIR
        / "Figure4.pdf"
    )

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