#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Reproduce Figure 5 of the CHASE magnetic-levitation manuscript.

Figure 5:
    Six-degree-of-freedom confinement of the inter-module relative state
    during in-orbit operation.

Required input file
-------------------
Preferred filename:
    data/Data_S3_6DoF_Relative_State_Telemetry.xlsx

Legacy filenames are also recognized:
    data/Data_S3_6DOF.xlsx
    data/ManeuveringNew.xlsx

State convention
----------------
The relative state describes the Support Module (SM) with respect to
the Payload Module (PM), consistent with the manuscript and SI Appendix.

Relative attitude:
    TMF0913, TMF0914, TMF0915   [deg]

Relative translation:
    TMF0916, TMF0917, TMF0918   [mm]

Scientific processing
---------------------
No baseline removal, mean centering, or recentering is applied.

The measured relative state itself is compared directly with the
prescribed operational envelope:

    translation: ±5.0 mm
    rotation:    ±0.3 deg

Peak boundary utilization:
    U = 100 * max_j( max_t |q_j(t)| / q_limit )

Minimum remaining margin:
    M = min_j( q_limit - max_t |q_j(t)| )

Visualization
-------------
- Raw 3-D state samples are colored by normalized local sample density.
- Samples are projected onto the three coordinate planes.
- Gray dashed polygons denote convex hulls of projected samples.
- Muted-red dashed cube edges denote the prescribed operational bounds.
- Density is normalized independently in the translational and
  rotational panels and is used only for visualization.

Outputs
-------
    results/Figure5.png
    results/Figure5.pdf
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Use non-GUI backend.
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from matplotlib.colors import (
    LinearSegmentedColormap,
    Normalize,
)
from matplotlib.cm import ScalarMappable

from scipy.spatial import (
    cKDTree,
    ConvexHull,
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
        "next to figure5.py."
    )

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================================
# Operational limits
# ============================================================================

TRANSLATION_LIMIT_MM = 5.0
ROTATION_LIMIT_DEG = 0.3

# Plot slightly beyond the operational boundary.
TRANSLATION_AXIS_LIMIT_MM = 5.5
ROTATION_AXIS_LIMIT_DEG = 0.33


# ============================================================================
# Local-density parameter
# ============================================================================

DENSITY_NEIGHBORS = 40


# ============================================================================
# Academic / restrained color scheme
# ============================================================================

# Muted red for the operational boundary.
BOUNDARY_COLOR = "#B04A4A"

# Neutral gray for convex hulls.
HULL_COLOR = "#6A6A6A"

# Projection points should remain secondary.
PROJECTION_ALPHA = 0.13

# Main 3-D trajectory should be clearly visible.
MAIN_POINT_ALPHA = 0.90

# Sequential blue palette:
# pale blue -> medium blue -> royal blue -> navy.
DENSITY_CMAP = LinearSegmentedColormap.from_list(
    "chase_density_academic",
    [
        "#CFE2F3",
        "#8DB8D9",
        "#4F8AB8",
        "#1F5F96",
        "#083A67",
    ],
    N=256,
)


# ============================================================================
# File utilities
# ============================================================================

def first_existing_file(
    candidates: list[str],
) -> Path:
    """Return the first existing source-data file."""

    for name in candidates:
        path = DATA_DIR / name

        if path.exists():
            return path

    names = "\n  ".join(candidates)

    raise FileNotFoundError(
        "Could not locate the required source-data file.\n"
        f"Checked:\n  {names}"
    )


def find_column(
    df: pd.DataFrame,
    key: str,
) -> np.ndarray:
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


# ============================================================================
# Load source telemetry
# ============================================================================

def load_relative_state() -> tuple[np.ndarray, np.ndarray]:

    data_file = first_existing_file(
        [
            "Data_S3_6DoF_Relative_State_Telemetry.xlsx",
            "Data_S3_6DOF.xlsx",
            "Data S3. Raster_Scanning_6DoF_Relative_Metrology.xlsx",
            "ManeuveringNew.xlsx",
        ]
    )

    print(
        f"[Figure 5] Relative-state source: {data_file.name}"
    )

    df = pd.read_excel(
        data_file
    )

    # ---------------------------------------------------------------
    # Relative attitude of SM with respect to PM [deg]
    # ---------------------------------------------------------------

    rotation = np.column_stack(
        [
            find_column(df, "TMF0913"),
            find_column(df, "TMF0914"),
            find_column(df, "TMF0915"),
        ]
    )

    # ---------------------------------------------------------------
    # Relative translation of SM with respect to PM [mm]
    # ---------------------------------------------------------------

    translation = np.column_stack(
        [
            find_column(df, "TMF0916"),
            find_column(df, "TMF0917"),
            find_column(df, "TMF0918"),
        ]
    )

    print(
        f"[Figure 5] Number of samples: {len(df)}"
    )

    return translation, rotation


# ============================================================================
# Confinement metrics
# ============================================================================

def confinement_metrics(
    state: np.ndarray,
    limit: float,
) -> tuple[np.ndarray, float, float]:
    """
    Compute component-wise peak excursion, peak boundary utilization,
    and minimum remaining margin.

    IMPORTANT:
        The raw measured state is used directly.
        No mean-centering or baseline removal is performed.
    """

    state = np.asarray(
        state,
        dtype=float,
    )

    max_component = np.max(
        np.abs(state),
        axis=0,
    )

    utilization = (
        100.0
        * np.max(
            max_component / limit
        )
    )

    margin = np.min(
        limit - max_component
    )

    return (
        max_component,
        utilization,
        margin,
    )


# ============================================================================
# Local sample density
# ============================================================================

def local_density(
    points: np.ndarray,
    k: int = DENSITY_NEIGHBORS,
) -> np.ndarray:
    """
    Estimate normalized local sample density using k-nearest neighbors.

    The local density is estimated as

        rho ~ 1 / r_k^3

    where r_k is the distance to the k-th nearest neighbor in the
    standardized three-dimensional state space.

    For visualization, the raw density is transformed to log10 space
    and robustly normalized between its 5th and 95th percentiles.
    Values outside this interval are clipped to [0, 1].

    This robust normalization prevents a small number of extremely
    dense samples from compressing most points into the low end of the
    colormap.

    Higher returned values correspond to higher local sample density.
    """

    points = np.asarray(
        points,
        dtype=float,
    )

    n = points.shape[0]

    if n < 3:
        return np.zeros(
            n,
            dtype=float,
        )

    # ------------------------------------------------------------------
    # Standardize the three coordinates only for density estimation.
    # This prevents one coordinate with a larger numerical spread from
    # dominating the Euclidean nearest-neighbor distance.
    # ------------------------------------------------------------------

    scale = np.std(
        points,
        axis=0,
        ddof=1,
    )

    scale[
        scale == 0
    ] = 1.0

    normalized_points = (
        points / scale
    )

    # ------------------------------------------------------------------
    # k-nearest-neighbor search
    # ------------------------------------------------------------------

    tree = cKDTree(
        normalized_points
    )

    k_eff = min(
        k + 1,
        n,
    )

    distances, _ = tree.query(
        normalized_points,
        k=k_eff,
    )

    # Column 0 corresponds to the point itself.
    kth_distance = distances[:, -1]

    eps = np.finfo(float).eps

    # ------------------------------------------------------------------
    # Local density:
    #
    #     rho ~ 1 / r_k^3
    #
    # because the state space is three-dimensional.
    # ------------------------------------------------------------------

    raw_density = (
        1.0
        / np.maximum(
            kth_distance,
            eps,
        ) ** 3
    )

    # ------------------------------------------------------------------
    # Log compression
    #
    # kNN density can span several orders of magnitude. Working in
    # log-density space makes the visual color scale much more useful.
    # ------------------------------------------------------------------

    log_density = np.log10(
        raw_density
    )

    # ------------------------------------------------------------------
    # Robust visualization range
    #
    # The extreme tails are clipped so that a very small number of
    # unusually dense/sparse samples do not determine the entire
    # colormap.
    # ------------------------------------------------------------------

    lower = np.percentile(
        log_density,
        5.0,
    )

    upper = np.percentile(
        log_density,
        95.0,
    )

    if upper <= lower:
        return np.zeros(
            n,
            dtype=float,
        )

    density = (
        log_density - lower
    ) / (
        upper - lower
    )

    density = np.clip(
        density,
        0.0,
        1.0,
    )

    return density


# ============================================================================
# Operational-boundary geometry
# ============================================================================

def cube_edges(
    limit: float,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """
    Return the 12 edges of a cube spanning [-limit, +limit]^3.
    """

    vertices = np.array(
        [
            [-limit, -limit, -limit],
            [-limit, -limit, +limit],
            [-limit, +limit, -limit],
            [-limit, +limit, +limit],
            [+limit, -limit, -limit],
            [+limit, -limit, +limit],
            [+limit, +limit, -limit],
            [+limit, +limit, +limit],
        ],
        dtype=float,
    )

    edge_indices = [
        (0, 1),
        (0, 2),
        (0, 4),
        (1, 3),
        (1, 5),
        (2, 3),
        (2, 6),
        (3, 7),
        (4, 5),
        (4, 6),
        (5, 7),
        (6, 7),
    ]

    return [
        (
            vertices[i],
            vertices[j],
        )
        for i, j in edge_indices
    ]


def draw_boundary_cube(
    ax,
    limit: float,
) -> None:
    """Draw the prescribed component-wise operational boundary."""

    for p1, p2 in cube_edges(
        limit
    ):

        ax.plot(
            [p1[0], p2[0]],
            [p1[1], p2[1]],
            [p1[2], p2[2]],
            "--",
            color=BOUNDARY_COLOR,
            linewidth=1.20,
            alpha=0.90,
            zorder=2,
        )


# ============================================================================
# Convex-hull utilities
# ============================================================================

def convex_hull_2d(
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray | None:
    """
    Return an ordered closed convex-hull polygon in a 2-D projection.
    """

    xy = np.column_stack(
        [x, y]
    )

    if len(xy) < 3:
        return None

    try:
        hull = ConvexHull(
            xy
        )
    except Exception:
        return None

    polygon = xy[
        hull.vertices
    ]

    polygon = np.vstack(
        [
            polygon,
            polygon[0],
        ]
    )

    return polygon


# ============================================================================
# Projections onto coordinate planes
# ============================================================================

def draw_projections(
    ax,
    points: np.ndarray,
    density: np.ndarray,
    boundary_limit: float,
) -> None:
    """
    Project the measured state onto three boundary planes.

    XY projection:
        z = -limit

    XZ projection:
        y = +limit

    YZ projection:
        x = -limit
    """

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    # ------------------------------------------------------------------
    # XY projection: z = -limit
    # ------------------------------------------------------------------

    ax.scatter(
        x,
        y,
        np.full_like(
            x,
            -boundary_limit,
        ),
        c=density,
        cmap=DENSITY_CMAP,
        vmin=0,
        vmax=1,
        s=2.5,
        alpha=PROJECTION_ALPHA,
        linewidths=0,
        depthshade=False,
        zorder=1,
    )

    hull_xy = convex_hull_2d(
        x,
        y,
    )

    if hull_xy is not None:
        ax.plot(
            hull_xy[:, 0],
            hull_xy[:, 1],
            np.full(
                len(hull_xy),
                -boundary_limit,
            ),
            "--",
            color=HULL_COLOR,
            linewidth=1.10,
            alpha=0.90,
            zorder=3,
        )

    # ------------------------------------------------------------------
    # XZ projection: y = +limit
    # ------------------------------------------------------------------

    ax.scatter(
        x,
        np.full_like(
            x,
            +boundary_limit,
        ),
        z,
        c=density,
        cmap=DENSITY_CMAP,
        vmin=0,
        vmax=1,
        s=3.2,
        alpha=PROJECTION_ALPHA,
        linewidths=0,
        depthshade=False,
        zorder=1,
    )

    hull_xz = convex_hull_2d(
        x,
        z,
    )

    if hull_xz is not None:
        ax.plot(
            hull_xz[:, 0],
            np.full(
                len(hull_xz),
                +boundary_limit,
            ),
            hull_xz[:, 1],
            "--",
            color=HULL_COLOR,
            linewidth=1.10,
            alpha=0.90,
            zorder=3,
        )

    # ------------------------------------------------------------------
    # YZ projection: x = -limit
    # ------------------------------------------------------------------

    ax.scatter(
        np.full_like(
            y,
            -boundary_limit,
        ),
        y,
        z,
        c=density,
        cmap=DENSITY_CMAP,
        vmin=0,
        vmax=1,
        s=3.2,
        alpha=PROJECTION_ALPHA,
        linewidths=0,
        depthshade=False,
        zorder=1,
    )

    hull_yz = convex_hull_2d(
        y,
        z,
    )

    if hull_yz is not None:
        ax.plot(
            np.full(
                len(hull_yz),
                -boundary_limit,
            ),
            hull_yz[:, 0],
            hull_yz[:, 1],
            "--",
            color=HULL_COLOR,
            linewidth=1.10,
            alpha=0.90,
            zorder=3,
        )


# ============================================================================
# Draw one state panel
# ============================================================================

def draw_state_panel(
    ax,
    points: np.ndarray,
    density: np.ndarray,
    boundary_limit: float,
    axis_limit: float,
    title: str,
    labels: tuple[str, str, str],
    annotation: str,
) -> None:

    # ------------------------------------------------------------------
    # Plane projections
    # ------------------------------------------------------------------

    draw_projections(
        ax=ax,
        points=points,
        density=density,
        boundary_limit=boundary_limit,
    )

    # ------------------------------------------------------------------
    # Operational boundary
    # ------------------------------------------------------------------

    draw_boundary_cube(
        ax=ax,
        limit=boundary_limit,
    )

    # ------------------------------------------------------------------
    # Main 3-D measured trajectory
    #
    # Draw low-density points first and high-density points last so the
    # dense core remains visible.
    # ------------------------------------------------------------------

    order = np.argsort(
        density
    )

    p = points[
        order
    ]

    d = density[
        order
    ]

    ax.scatter(
        p[:, 0],
        p[:, 1],
        p[:, 2],
        c=d,
        cmap=DENSITY_CMAP,
        vmin=0,
        vmax=1,
        s=6.0,
        alpha=MAIN_POINT_ALPHA,
        linewidths=0,
        depthshade=False,
        zorder=5,
    )

    # ------------------------------------------------------------------
    # Axis formatting
    # ------------------------------------------------------------------

    ax.set_xlim(
        -axis_limit,
        axis_limit,
    )

    ax.set_ylim(
        -axis_limit,
        axis_limit,
    )

    ax.set_zlim(
        -axis_limit,
        axis_limit,
    )

    ax.set_xlabel(
        labels[0],
        labelpad=6,
        fontsize=10.5,
    )

    ax.set_ylabel(
        labels[1],
        labelpad=6,
        fontsize=10.5,
    )

    ax.set_zlabel(
        labels[2],
        labelpad=6,
        fontsize=10.5,
    )

    ax.set_title(
        title,
        fontsize=11.5,
        pad=10,
    )

    ax.view_init(
        elev=22,
        azim=-58,
    )

    ax.set_box_aspect(
        (1, 1, 1)
    )

    ax.tick_params(
        labelsize=8.5,
        pad=1,
        direction="in",
    )

    ax.grid(True)
    # 3D坐标轴网格：设置浅灰，降低透明度
    light_gray = (0.72, 0.72, 0.72, 0.14)  # RGBA，A越小越淡
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo["grid"]["color"] = light_gray
        axis._axinfo["grid"]["linewidth"] = 0.6

    # White / transparent panes.
    for axis in [
        ax.xaxis,
        ax.yaxis,
        ax.zaxis,
    ]:

        axis.pane.set_facecolor(
            (1, 1, 1, 0)
        )

        axis.pane.set_edgecolor(
            (0.70, 0.70, 0.70, 0.65)
        )

    # ------------------------------------------------------------------
    # Confinement annotation
    # ------------------------------------------------------------------

    ax.text2D(
        0.50,
        0.965,
        annotation,
        transform=ax.transAxes,
        fontsize=9.4,
        ha="left",
        va="top",
    )


# ============================================================================
# Numerical summary
# ============================================================================

def print_summary(
    translation: np.ndarray,
    rotation: np.ndarray,
) -> tuple[
    float,
    float,
    float,
    float,
]:

    (
        max_translation,
        translation_utilization,
        translation_margin,
    ) = confinement_metrics(
        translation,
        TRANSLATION_LIMIT_MM,
    )

    (
        max_rotation,
        rotation_utilization,
        rotation_margin,
    ) = confinement_metrics(
        rotation,
        ROTATION_LIMIT_DEG,
    )

    print(
        "\nFigure 5 raw-state confinement summary"
    )

    print(
        "-------------------------------------------------------------"
    )

    print(
        "State convention: SM with respect to PM"
    )

    print(
        "Baseline removal / centering: NONE"
    )

    print(
        "\nTranslation [mm]"
    )

    print(
        "  max |x|, |y|, |z| : "
        f"{max_translation[0]:.5f}, "
        f"{max_translation[1]:.5f}, "
        f"{max_translation[2]:.5f}"
    )

    print(
        "  peak utilization  : "
        f"{translation_utilization:.3f}%"
    )

    print(
        "  remaining margin  : "
        f"{translation_margin:.5f} mm"
    )

    print(
        "\nRelative attitude [deg]"
    )

    print(
        "  max |x|, |y|, |z| : "
        f"{max_rotation[0]:.6f}, "
        f"{max_rotation[1]:.6f}, "
        f"{max_rotation[2]:.6f}"
    )

    print(
        "  peak utilization  : "
        f"{rotation_utilization:.3f}%"
    )

    print(
        "  remaining margin  : "
        f"{rotation_margin:.6f} deg"
    )

    print(
        "-------------------------------------------------------------"
    )

    return (
        translation_utilization,
        translation_margin,
        rotation_utilization,
        rotation_margin,
    )


# ============================================================================
# Figure construction
# ============================================================================

def make_figure(
    translation: np.ndarray,
    rotation: np.ndarray,
    translation_density: np.ndarray,
    rotation_density: np.ndarray,
    translation_utilization: float,
    translation_margin: float,
    rotation_utilization: float,
    rotation_margin: float,
) -> plt.Figure:

    fig = plt.figure(
        figsize=(12.2, 5.8),
        facecolor="white",
    )

    ax_translation = fig.add_subplot(
        1,
        2,
        1,
        projection="3d",
    )

    ax_rotation = fig.add_subplot(
        1,
        2,
        2,
        projection="3d",
    )

    # ------------------------------------------------------------------
    # Panel (a): translation
    # ------------------------------------------------------------------

    draw_state_panel(
        ax=ax_translation,
        points=translation,
        density=translation_density,
        boundary_limit=TRANSLATION_LIMIT_MM,
        axis_limit=TRANSLATION_AXIS_LIMIT_MM,
        title="Relative translational state",
        labels=(
            r"$\Delta x$ (mm)",
            r"$\Delta y$ (mm)",
            r"$\Delta z$ (mm)",
        ),
        annotation=(
            f"Peak boundary utilization: "
            f"{translation_utilization:.1f}%\n"
            f"Remaining margin: "
            f"{translation_margin:.2f} mm"
        ),
    )

    # ------------------------------------------------------------------
    # Panel (b): rotation
    # ------------------------------------------------------------------

    draw_state_panel(
        ax=ax_rotation,
        points=rotation,
        density=rotation_density,
        boundary_limit=ROTATION_LIMIT_DEG,
        axis_limit=ROTATION_AXIS_LIMIT_DEG,
        title="Relative rotational state",
        labels=(
            r"$\Delta\theta_x$ (deg)",
            r"$\Delta\theta_y$ (deg)",
            r"$\Delta\theta_z$ (deg)",
        ),
        annotation=(
            f"Peak boundary utilization: "
            f"{rotation_utilization:.1f}%\n"
            f"Remaining margin: "
            f"{rotation_margin:.3f}°"
        ),
    )

    # ------------------------------------------------------------------
    # Panel labels
    # ------------------------------------------------------------------

    fig.text(
        0.035,
        0.945,
        "a",
        fontsize=17,
        fontweight="bold",
        va="top",
    )

    fig.text(
        0.505,
        0.945,
        "b",
        fontsize=17,
        fontweight="bold",
        va="top",
    )

    # ------------------------------------------------------------------
    # Shared colorbar
    # ------------------------------------------------------------------

    sm = ScalarMappable(
        norm=Normalize(
            0,
            1,
        ),
        cmap=DENSITY_CMAP,
    )

    sm.set_array([])

    cax = fig.add_axes(
        [
            0.925,
            0.205,
            0.014,
            0.60,
        ]
    )

    cb = fig.colorbar(
        sm,
        cax=cax,
    )

    cb.set_label(
        "Normalized local density",
        fontsize=10,
    )

    cb.set_ticks(
        [0, 1],
        labels=[
            "Low",
            "High",
        ],
    )

    cb.ax.tick_params(
        labelsize=8.5,
        length=0,
    )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    fig.subplots_adjust(
        left=0.035,
        right=0.900,
        bottom=0.045,
        top=0.915,
        wspace=0.01,
    )

    return fig


# ============================================================================
# Main
# ============================================================================

def main() -> None:

    translation, rotation = (
        load_relative_state()
    )

    # ------------------------------------------------------------------
    # IMPORTANT:
    # Raw state is used directly.
    # No baseline removal / no mean centering / no recentering.
    # ------------------------------------------------------------------

    translation_raw = (
        translation.copy()
    )

    rotation_raw = (
        rotation.copy()
    )

    # ------------------------------------------------------------------
    # Confinement metrics
    # ------------------------------------------------------------------

    (
        translation_utilization,
        translation_margin,
        rotation_utilization,
        rotation_margin,
    ) = print_summary(
        translation_raw,
        rotation_raw,
    )

    # ------------------------------------------------------------------
    # Local density for visualization only
    # ------------------------------------------------------------------

    print(
        "\n[Figure 5] Estimating local sample density..."
    )

    translation_density = local_density(
        translation_raw
    )

    rotation_density = local_density(
        rotation_raw
    )

    # ------------------------------------------------------------------
    # Build figure
    # ------------------------------------------------------------------

    fig = make_figure(
        translation=translation_raw,
        rotation=rotation_raw,
        translation_density=translation_density,
        rotation_density=rotation_density,
        translation_utilization=translation_utilization,
        translation_margin=translation_margin,
        rotation_utilization=rotation_utilization,
        rotation_margin=rotation_margin,
    )

    png_file = (
        RESULTS_DIR
        / "Figure5.png"
    )

    pdf_file = (
        RESULTS_DIR
        / "Figure5.pdf"
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