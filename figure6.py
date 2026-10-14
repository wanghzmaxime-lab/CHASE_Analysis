#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Figure 6 of the CHASE magnetic-levitation manuscript.

Figure 6:
    Long-term spectroscopic consistency during fully mechanically
    untethered operation.

Required input
--------------
Preferred filename:
    data/Data_S4_Halpha_Raster_Scan_202212_202607.xlsx

The workbook must contain a sheet named:

    ScanMetrics

with, at minimum:

    observation_utc
    robust_3sigma_mA

or the preferred machine-readable equivalent:

    robust_3sigma_mAngstrom

If a MonthlySummary sheet is present, it is used only for an integrity
check. The plotted monthly statistics are recalculated directly from
ScanMetrics.

Scientific processing
---------------------
1. Use scan-level observations from December 2022 through July 2026.

2. December 2022 is the first complete calendar month following cable
   shedding on 7 November 2022.

3. Monthly statistics are calculated directly from the selected
   scan-level robust 3-sigma values:
       - number of scans
       - median
       - 25th percentile
       - 75th percentile

4. The descriptive linear trend is fitted by ordinary least squares
   to the 44 MONTHLY MEDIANS only.

5. Monthly time coordinates are assigned to the 15th day of each month.

6. Time is expressed in Julian years using:
       365.2425 days/year

7. The reported slope confidence interval is the 95% Student-t
   confidence interval for the fitted slope.

8. The shaded region in Figure 6 is the 95% confidence band of the
   FITTED MEAN, not a prediction interval.

Interpretation
--------------
The linear trend is a descriptive summary of the longitudinal dataset.
It is not interpreted as a causal model of spacecraft performance.

The scan-level H-alpha metric is an instrument-level spectroscopic
consistency metric. It is not a direct spacecraft LOS-attitude
measurement.

Outputs
-------
    results/Figure6.png
    results/Figure6.pdf

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
import matplotlib.dates as mdates

from scipy.stats import t as student_t


# ============================================================================
# Paths
# ============================================================================

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

if not DATA_DIR.exists():
    raise FileNotFoundError(
        f"Data directory not found: {DATA_DIR}\n"
        "Please place Supplementary Data S4 in a 'data' folder "
        "next to figure6.py."
    )

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Analysis interval
# ============================================================================

START_DATE = pd.Timestamp("2022-12-01")
END_DATE_EXCLUSIVE = pd.Timestamp("2026-08-01")

DAYS_PER_YEAR = 365.2425


# ============================================================================
# Plot settings
# ============================================================================

# Individual scans: light and visually subordinate.
SCAN_COLOR = "#A9CEE8"

# Monthly statistics: primary visual layer.
MONTH_COLOR = "#1B5E93"

# Trend: neutral gray so it is not visually interpreted as a performance
# requirement or model prediction.
TREND_COLOR = "#565656"

# 95% confidence band of fitted mean.
BAND_COLOR = "#BDBDBD"

Y_LIMITS = (4.0, 7.0)


# ============================================================================
# File utilities
# ============================================================================

def first_existing_file(
    candidates: list[str],
) -> Path:
    """Return the first existing candidate file."""

    for name in candidates:

        path = DATA_DIR / name

        if path.exists():
            return path

    names = "\n  ".join(candidates)

    raise FileNotFoundError(
        "Could not locate Supplementary Data S4.\n"
        f"Checked:\n  {names}"
    )


def find_data_file() -> Path:
    """
    Locate Supplementary Data S4.

    ASCII-only repository filename is preferred, but legacy manuscript
    filenames are also supported.
    """

    return first_existing_file(
        [
            "Data_S4_Halpha_Raster_Scan_202212_202607.xlsx",
            "Data_S4_RSM_Halpha_Full_Disk_Scan_202212_202607.xlsx",
            "Data S4. RSM_Hα_Full-Disk_Scan_202212_202607.xlsx",
            "Data S4. RSM_Hα_Full-Disk_Scan_202212_202607(1).xlsx",
            "Data S4. RSM_Hα_Full-Disk_Scan_202212_202607(2).xlsx",
        ]
    )


def choose_metric_column(
    df: pd.DataFrame,
) -> str:
    """
    Find the scan-level robust 3-sigma metric.

    The repository-preferred name is mAngstrom; the existing manuscript
    source workbook may still use mA for milli-Angstrom.
    """

    candidates = [
        "robust_3sigma_mAngstrom",
        "robust_3sigma_mA",
    ]

    for column in candidates:

        if column in df.columns:
            return column

    raise KeyError(
        "Could not find the scan-level robust 3-sigma metric column.\n"
        "Expected one of:\n"
        "  robust_3sigma_mAngstrom\n"
        "  robust_3sigma_mA"
    )


# ============================================================================
# Load source data
# ============================================================================

def load_scan_metrics(
    data_file: Path,
) -> tuple[pd.DataFrame, str]:

    print(
        f"[Figure 6] Source workbook: {data_file.name}"
    )

    scan = pd.read_excel(
        data_file,
        sheet_name="ScanMetrics",
    )

    required = {
        "observation_utc",
    }

    missing = (
        required
        - set(scan.columns)
    )

    if missing:
        raise KeyError(
            "Missing required ScanMetrics columns: "
            f"{sorted(missing)}"
        )

    metric_column = choose_metric_column(
        scan
    )

    # ------------------------------------------------------------------
    # UTC observation time
    # ------------------------------------------------------------------

    observation_time = pd.to_datetime(
        scan["observation_utc"],
        utc=True,
        errors="coerce",
    )

    # Remove timezone information for Matplotlib.
    observation_time = (
        observation_time
        .dt.tz_convert(None)
    )

    metric = pd.to_numeric(
        scan[metric_column],
        errors="coerce",
    )

    valid = (
        observation_time.notna()
        & np.isfinite(metric)
        & (observation_time >= START_DATE)
        & (observation_time < END_DATE_EXCLUSIVE)
    )

    df = pd.DataFrame(
        {
            "observation_utc": observation_time[valid],
            "robust_3sigma_mA": metric[valid].to_numpy(dtype=float),
        }
    )

    df = (
        df
        .sort_values("observation_utc")
        .reset_index(drop=True)
    )

    if df.empty:
        raise ValueError(
            "No valid Figure 6 observations were found."
        )

    # Each observation belongs to a calendar month.
    df["month"] = (
        df["observation_utc"]
        .dt.to_period("M")
    )

    return df, metric_column


# ============================================================================
# Recalculate monthly statistics from ScanMetrics
# ============================================================================

def build_monthly_summary(
    scan_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Recalculate the monthly statistics directly from the scan-level data.

    Linear quantile interpolation is used, matching the current source
    workbook.
    """

    grouped = (
        scan_df
        .groupby(
            "month",
            sort=True,
        )["robust_3sigma_mA"]
    )

    monthly = pd.DataFrame(
        {
            "n": grouped.size(),
            "median": grouped.median(),
            "q25": grouped.quantile(
                0.25,
                interpolation="linear",
            ),
            "q75": grouped.quantile(
                0.75,
                interpolation="linear",
            ),
        }
    ).reset_index()

    # Plot monthly statistics at the 15th day of each month.
    monthly["month_dt"] = (
        monthly["month"]
        .dt.to_timestamp()
        + pd.Timedelta(days=14)
    )

    return monthly


# ============================================================================
# Optional integrity check against MonthlySummary sheet
# ============================================================================

def validate_monthly_summary_sheet(
    data_file: Path,
    calculated: pd.DataFrame,
) -> None:
    """
    Compare recalculated monthly statistics with the workbook's
    MonthlySummary sheet, if present.

    This check does not affect the plotted values.
    """

    xls = pd.ExcelFile(
        data_file
    )

    if "MonthlySummary" not in xls.sheet_names:

        print(
            "[Figure 6] MonthlySummary sheet not found; "
            "skipping workbook consistency check."
        )

        return

    supplied = pd.read_excel(
        data_file,
        sheet_name="MonthlySummary",
    )

    required_columns = {
        "month",
        "n_valid_scans",
    }

    if not required_columns.issubset(
        supplied.columns
    ):

        print(
            "[Figure 6] MonthlySummary sheet uses an unexpected schema; "
            "skipping consistency check."
        )

        return

    # Allow both old and repository-preferred units in column names.
    possible_sets = [
        (
            "median_robust_3sigma_mAngstrom",
            "q25_robust_3sigma_mAngstrom",
            "q75_robust_3sigma_mAngstrom",
        ),
        (
            "median_robust_3sigma_mA",
            "q25_robust_3sigma_mA",
            "q75_robust_3sigma_mA",
        ),
    ]

    summary_columns = None

    for candidate in possible_sets:

        if all(
            col in supplied.columns
            for col in candidate
        ):
            summary_columns = candidate
            break

    if summary_columns is None:

        print(
            "[Figure 6] MonthlySummary metric columns not recognized; "
            "skipping consistency check."
        )

        return

    med_col, q25_col, q75_col = summary_columns

    supplied = supplied.copy()

    supplied["month"] = (
        supplied["month"]
        .astype(str)
    )

    calculated_check = calculated.copy()

    calculated_check["month"] = (
        calculated_check["month"]
        .astype(str)
    )

    merged = calculated_check.merge(
        supplied,
        on="month",
        how="inner",
    )

    if len(merged) != len(calculated):

        raise ValueError(
            "MonthlySummary does not contain the same monthly bins "
            "as the recalculated ScanMetrics summary."
        )

    numerical_checks = [
        np.allclose(
            merged["n"].to_numpy(float),
            merged["n_valid_scans"].to_numpy(float),
            rtol=0,
            atol=0,
        ),
        np.allclose(
            merged["median"].to_numpy(float),
            merged[med_col].to_numpy(float),
            rtol=0,
            atol=1e-10,
        ),
        np.allclose(
            merged["q25"].to_numpy(float),
            merged[q25_col].to_numpy(float),
            rtol=0,
            atol=1e-10,
        ),
        np.allclose(
            merged["q75"].to_numpy(float),
            merged[q75_col].to_numpy(float),
            rtol=0,
            atol=1e-10,
        ),
    ]

    if not all(
        numerical_checks
    ):

        raise ValueError(
            "The MonthlySummary sheet is inconsistent with the "
            "recalculated scan-level statistics."
        )

    print(
        "[Figure 6] MonthlySummary integrity check: PASSED"
    )


# ============================================================================
# Descriptive OLS trend
# ============================================================================

def fit_monthly_median_trend(
    monthly: pd.DataFrame,
) -> dict[str, object]:
    """
    Fit ordinary least squares to the monthly medians.

    Returns:
        intercept
        slope
        slope 95% CI
        fitted-mean 95% confidence band
    """

    month_dt = pd.DatetimeIndex(
        monthly["month_dt"]
    )

    y = (
        monthly["median"]
        .to_numpy(dtype=float)
    )

    n = len(
        y
    )

    if n < 3:
        raise ValueError(
            "At least three monthly bins are required for OLS."
        )

    # ------------------------------------------------------------------
    # Time coordinate in Julian years.
    # ------------------------------------------------------------------

    t0 = (
        month_dt[0]
    )

    x = (
        (month_dt - t0)
        .total_seconds()
        .to_numpy()
        / (
            DAYS_PER_YEAR
            * 86400.0
        )
    )

    # ------------------------------------------------------------------
    # Ordinary least squares:
    #
    #     y = beta0 + beta1 x
    # ------------------------------------------------------------------

    X = np.column_stack(
        [
            np.ones(n),
            x,
        ]
    )

    beta, _, _, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    intercept = float(
        beta[0]
    )

    slope = float(
        beta[1]
    )

    fitted = (
        X @ beta
    )

    residuals = (
        y - fitted
    )

    dof = (
        n - 2
    )

    SSE = np.sum(
        residuals**2
    )

    residual_variance = (
        SSE / dof
    )

    residual_sd = np.sqrt(
        residual_variance
    )

    # ------------------------------------------------------------------
    # Slope standard error and 95% Student-t CI.
    # ------------------------------------------------------------------

    x_mean = np.mean(
        x
    )

    Sxx = np.sum(
        (x - x_mean) ** 2
    )

    slope_se = (
        residual_sd
        / np.sqrt(Sxx)
    )

    tcrit = student_t.ppf(
        0.975,
        dof,
    )

    slope_ci = np.array(
        [
            slope - tcrit * slope_se,
            slope + tcrit * slope_se,
        ]
    )

    # ------------------------------------------------------------------
    # Smooth time grid covering the manuscript analysis window.
    # ------------------------------------------------------------------

    grid_dt = pd.date_range(
        START_DATE,
        END_DATE_EXCLUSIVE
        - pd.Timedelta(days=1),
        periods=400,
    )

    grid_x = (
        (grid_dt - t0)
        .total_seconds()
        .to_numpy()
        / (
            DAYS_PER_YEAR
            * 86400.0
        )
    )

    fitted_grid = (
        intercept
        + slope * grid_x
    )

    # ------------------------------------------------------------------
    # 95% confidence interval of the fitted MEAN:
    #
    # SE_mean =
    #     s * sqrt(
    #         1/n
    #         + (x - xbar)^2 / Sxx
    #     )
    #
    # This is not a prediction interval.
    # ------------------------------------------------------------------

    se_mean = (
        residual_sd
        * np.sqrt(
            1.0 / n
            + (
                (grid_x - x_mean) ** 2
                / Sxx
            )
        )
    )

    ci_low = (
        fitted_grid
        - tcrit * se_mean
    )

    ci_high = (
        fitted_grid
        + tcrit * se_mean
    )

    return {
        "intercept": intercept,
        "slope": slope,
        "slope_ci": slope_ci,
        "grid_dt": grid_dt,
        "fitted_grid": fitted_grid,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "dof": dof,
        "residual_sd": residual_sd,
    }


# ============================================================================
# Console summary
# ============================================================================

def print_summary(
    scan_df: pd.DataFrame,
    monthly: pd.DataFrame,
    trend: dict[str, object],
) -> None:

    slope = float(
        trend["slope"]
    )

    slope_ci = np.asarray(
        trend["slope_ci"],
        dtype=float,
    )

    print(
        "\nFigure 6 longitudinal-analysis summary"
    )

    print(
        "-------------------------------------------------------------"
    )

    print(
        f"Analysis interval : "
        f"{START_DATE.date()} to "
        f"{(END_DATE_EXCLUSIVE - pd.Timedelta(days=1)).date()}"
    )

    print(
        f"Raster scans      : {len(scan_df)}"
    )

    print(
        f"Monthly bins      : {len(monthly)}"
    )

    print(
        f"First observation : "
        f"{scan_df['observation_utc'].iloc[0]}"
    )

    print(
        f"Last observation  : "
        f"{scan_df['observation_utc'].iloc[-1]}"
    )

    print(
        "\nMonthly-median OLS trend"
    )

    print(
        f"  slope           : "
        f"{slope:.6f} mÅ/year"
    )

    print(
        f"  95% slope CI    : "
        f"{slope_ci[0]:.6f} to "
        f"{slope_ci[1]:.6f} mÅ/year"
    )

    print(
        "-------------------------------------------------------------"
    )


# ============================================================================
# Figure construction
# ============================================================================

def make_figure(
    scan_df: pd.DataFrame,
    monthly: pd.DataFrame,
    trend: dict[str, object],
) -> plt.Figure:

    slope = float(
        trend["slope"]
    )

    slope_ci = np.asarray(
        trend["slope_ci"],
        dtype=float,
    )

    grid_dt = trend[
        "grid_dt"
    ]

    fitted_grid = np.asarray(
        trend["fitted_grid"],
        dtype=float,
    )

    ci_low = np.asarray(
        trend["ci_low"],
        dtype=float,
    )

    ci_high = np.asarray(
        trend["ci_high"],
        dtype=float,
    )

    # ------------------------------------------------------------------
    # Use a publication-style sans-serif font.
    # Arial will be used where installed; otherwise Matplotlib falls back
    # to a compatible sans-serif font.
    # ------------------------------------------------------------------

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Arial",
                "DejaVu Sans",
            ],
            "mathtext.fontset": "dejavusans",
            "axes.unicode_minus": True,
        }
    )

    fig, ax = plt.subplots(
        figsize=(9.2, 4.0),
        facecolor="white",
    )

    # ------------------------------------------------------------------
    # 95% confidence band of fitted mean.
    # Draw first so it remains behind the observations.
    # ------------------------------------------------------------------

    ax.fill_between(
        grid_dt,
        ci_low,
        ci_high,
        color=BAND_COLOR,
        alpha=0.28,
        linewidth=0,
        zorder=1,
    )

    # ------------------------------------------------------------------
    # Individual raster scans.
    # ------------------------------------------------------------------

    h_scan = ax.scatter(
        scan_df["observation_utc"],
        scan_df["robust_3sigma_mA"],
        s=16,
        color=SCAN_COLOR,
        alpha=0.34,
        edgecolors="none",
        zorder=2,
        label="Individual raster scans",
    )

    # ------------------------------------------------------------------
    # Descriptive trend.
    # ------------------------------------------------------------------

    h_trend, = ax.plot(
        grid_dt,
        fitted_grid,
        "--",
        color=TREND_COLOR,
        linewidth=1.35,
        zorder=3,
        label="Descriptive linear trend",
    )

    # ------------------------------------------------------------------
    # Monthly median and IQR.
    # ------------------------------------------------------------------

    median = (
        monthly["median"]
        .to_numpy(dtype=float)
    )

    q25 = (
        monthly["q25"]
        .to_numpy(dtype=float)
    )

    q75 = (
        monthly["q75"]
        .to_numpy(dtype=float)
    )

    lower_error = (
        median - q25
    )

    upper_error = (
        q75 - median
    )

    h_month = ax.errorbar(
        monthly["month_dt"],
        median,
        yerr=np.vstack(
            [
                lower_error,
                upper_error,
            ]
        ),
        fmt="o",
        linestyle="none",
        markersize=4.3,
        color=MONTH_COLOR,
        ecolor=MONTH_COLOR,
        elinewidth=1.0,
        capsize=2.8,
        capthick=0.9,
        markerfacecolor=MONTH_COLOR,
        markeredgecolor=MONTH_COLOR,
        zorder=4,
        label="Monthly median (IQR)",
    )

    # ------------------------------------------------------------------
    # Axes
    # ------------------------------------------------------------------

    ax.set_xlim(
        pd.Timestamp("2022-12-01"),
        pd.Timestamp("2026-07-31"),
    )

    ax.set_ylim(
        *Y_LIMITS
    )

    # Six-month labeling, beginning with January 2023.
    ticks = pd.date_range(
        "2023-01-01",
        "2026-07-01",
        freq="6MS",
    )

    ax.set_xticks(
        ticks
    )

    ax.xaxis.set_major_formatter(
        mdates.DateFormatter(
            "%Y-%m"
        )
    )

    plt.setp(
        ax.get_xticklabels(),
        rotation=30,
        ha="right",
    )

    ax.set_xlabel(
        "Observation date (UTC)",
        fontsize=10.5,
    )

    ax.set_ylabel(
        r"Robust $3\sigma$ within-scan H$\alpha$ "
        r"line-center residual (m$\AA$)",
        fontsize=10.5,
    )

    ax.tick_params(
        axis="both",
        direction="out",
        labelsize=9.2,
    )

    # Horizontal grid only.
    ax.grid(
        False
    )

    ax.yaxis.grid(
        True,
        color="#C8C8C8",
        linewidth=0.65,
        alpha=0.42,
    )

    ax.xaxis.grid(
        False
    )

    ax.spines[
        "top"
    ].set_visible(
        False
    )

    ax.spines[
        "right"
    ].set_visible(
        False
    )

    ax.spines[
        "left"
    ].set_linewidth(
        0.8
    )

    ax.spines[
        "bottom"
    ].set_linewidth(
        0.8
    )

    # ------------------------------------------------------------------
    # Trend annotation
    # ------------------------------------------------------------------

    trend_text = (
        "Monthly-median trend: "
        rf"{slope:+.2f} m$\AA$ yr$^{{-1}}$ "
        rf"(95% CI {slope_ci[0]:+.2f} to "
        rf"{slope_ci[1]:+.2f})"
    )

    ax.text(
        0.012,
        0.965,
        trend_text,
        transform=ax.transAxes,
        horizontalalignment="left",
        verticalalignment="top",
        fontsize=9.2,
        color="#383838",
    )

    # ------------------------------------------------------------------
    # Legend
    # ------------------------------------------------------------------

    ax.legend(
        handles=[
            h_scan,
            h_month,
            h_trend,
        ],
        labels=[
            "Individual raster scans",
            "Monthly median (IQR)",
            "Descriptive linear trend",
        ],
        loc="upper right",
        frameon=False,
        fontsize=8.7,
        borderpad=0.15,
        labelspacing=0.30,
        handletextpad=0.55,
    )

    fig.tight_layout(
        pad=0.7
    )

    return fig


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    data_file = find_data_file()

    scan_df, metric_column = load_scan_metrics(
        data_file
    )

    print(
        f"[Figure 6] Metric column: {metric_column}"
    )

    # ------------------------------------------------------------------
    # Recalculate monthly statistics directly from ScanMetrics.
    # ------------------------------------------------------------------

    monthly = build_monthly_summary(
        scan_df
    )

    # ------------------------------------------------------------------
    # Verify workbook MonthlySummary, if present.
    # ------------------------------------------------------------------

    validate_monthly_summary_sheet(
        data_file,
        monthly,
    )

    # ------------------------------------------------------------------
    # Descriptive monthly-median trend.
    # ------------------------------------------------------------------

    trend = fit_monthly_median_trend(
        monthly
    )

    print_summary(
        scan_df,
        monthly,
        trend,
    )

    # ------------------------------------------------------------------
    # Generate figure.
    # ------------------------------------------------------------------

    fig = make_figure(
        scan_df,
        monthly,
        trend,
    )

    png_file = (
        RESULTS_DIR
        / "Figure6.png"
    )

    pdf_file = (
        RESULTS_DIR
        / "Figure6.pdf"
    )

    fig.savefig(
        png_file,
        dpi=600,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        pdf_file,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(
        fig
    )

    print(
        "\nSaved:"
    )

    print(
        f"  {png_file}"
    )

    print(
        f"  {pdf_file}"
    )


if __name__ == "__main__":
    main()