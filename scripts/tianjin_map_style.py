# -*- coding: utf-8 -*-
"""Reusable publication map styling for the Tianjin heat-risk project.

The module centralizes typography, the five-class palette, administrative
boundary styling, north arrow, scale bar, and classified-map legend.  Future
map scripts should import these helpers instead of copying drawing code.
All geometric annotations assume a projected CRS whose map unit is metre.
"""

from dataclasses import dataclass

import matplotlib.pyplot as plt
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, PathPatch, Polygon, Rectangle
import numpy as np


@dataclass(frozen=True)
class TianjinMapStyle:
    """Parameters shared by the project's publication maps."""

    # Ordered from low to high.
    palette: tuple[str, ...] = (
        "#6D73B9",
        "#72C5CE",
        "#F3E6C8",
        "#F3A45F",
        "#E85B45",
    )
    boundary_color: str = "#333333"

    # Match the ecosystem-vulnerability reference figure exactly.
    north_arrow_height_fraction: float = 0.068
    north_arrow_height_to_width: float = 2.0
    north_arrow_notch_fraction: float = 0.12
    north_arrow_x_from_east_fraction: float = 0.13
    north_arrow_top_offset_fraction: float = 0.025

    scale_ticks_km: tuple[int, ...] = (0, 5, 10, 20, 30, 40)


DEFAULT_MAP_STYLE = TianjinMapStyle()
FIVE_CLASS_PALETTE = list(DEFAULT_MAP_STYLE.palette)


def configure_publication_style(style=DEFAULT_MAP_STYLE):
    """Apply the project's Chinese/Western typography and white background."""
    plt.rcParams.update({
        # Times New Roman supplies Latin letters/numbers; SimSun supplies
        # Chinese glyphs through Matplotlib's fallback mechanism.
        "font.family": ["Times New Roman", "SimSun"],
        "font.serif": ["Times New Roman", "SimSun"],
        "font.sans-serif": ["Times New Roman", "SimSun"],
        "axes.unicode_minus": False,
        "font.size": 10,
        "axes.titlesize": 15,
        "axes.titleweight": "normal",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def add_north_arrow(ax, bounds, style=DEFAULT_MAP_STYLE):
    """Draw a 2:1 black north arrow with a shallow V-shaped notch."""
    left, bottom, right, top = bounds
    width = right - left
    height = top - bottom

    x = right + style.north_arrow_x_from_east_fraction * width
    tip_y = top - style.north_arrow_top_offset_fraction * height
    arrow_height = style.north_arrow_height_fraction * height
    base_y = tip_y - arrow_height

    # Both dimensions use projected map units, so the visible ratio is exact.
    arrow_width = arrow_height / style.north_arrow_height_to_width
    half_width = arrow_width / 2
    notch_y = base_y + style.north_arrow_notch_fraction * arrow_height

    arrow = Polygon(
        [
            (x, tip_y),
            (x - half_width, base_y),
            (x, notch_y),
            (x + half_width, base_y),
        ],
        closed=True,
        facecolor="black",
        edgecolor="black",
        linewidth=0.45,
        zorder=8,
    )
    ax.add_patch(arrow)
    ax.text(
        x,
        tip_y + 0.008 * height,
        "N",
        ha="center",
        va="bottom",
        fontsize=9,
        color="black",
        zorder=9,
    )


def add_scale_bar(ax, bounds, style=DEFAULT_MAP_STYLE):
    """Draw the standard external 0-5-10-20-30-40 km scale bar."""
    left, bottom, right, top = bounds
    width = right - left
    height = top - bottom
    tick_km = style.scale_ticks_km
    segment_km = np.diff(tick_km)
    x0 = left + 0.025 * width
    y0 = bottom - 0.060 * height
    bar_height = 0.009 * height

    current_x = x0
    for index, length_km in enumerate(segment_km):
        segment_m = float(length_km) * 1000
        ax.add_patch(Rectangle(
            (current_x, y0),
            segment_m,
            bar_height,
            facecolor="black" if index % 2 == 0 else "white",
            edgecolor="black",
            linewidth=0.65,
            zorder=8,
        ))
        current_x += segment_m

    label_y = y0 - 0.008 * height
    for value in tick_km:
        ax.text(
            x0 + value * 1000,
            label_y,
            str(value),
            ha="center",
            va="top",
            fontsize=7,
            color="black",
            zorder=9,
        )
    ax.text(
        x0 + tick_km[-1] * 1000 + 0.012 * width,
        y0 + bar_height / 2,
        "km",
        ha="left",
        va="center",
        fontsize=7.5,
        color="black",
        zorder=9,
    )


def add_boundary_outline(
    ax,
    geometry,
    linewidth=0.75,
    style=DEFAULT_MAP_STYLE,
):
    """Add a polygon outline and return it for use as a raster clip path."""
    coords = np.asarray(geometry.exterior.coords, dtype=np.float64)
    codes = np.full(coords.shape[0], MplPath.LINETO, dtype=np.uint8)
    codes[0] = MplPath.MOVETO
    codes[-1] = MplPath.CLOSEPOLY
    path = MplPath(coords, codes)
    patch = PathPatch(
        path,
        transform=ax.transData,
        facecolor="none",
        edgecolor=style.boundary_color,
        linewidth=linewidth,
        joinstyle="round",
        capstyle="round",
        zorder=6,
    )
    ax.add_patch(patch)
    return patch


def add_classified_legend(
    ax,
    title,
    labels,
    *,
    palette=None,
    compact=False,
    reverse=False,
):
    """Add the standard five-class legend and return the Legend object."""
    palette = list(palette or DEFAULT_MAP_STYLE.palette)
    labels = list(labels)
    if len(palette) != len(labels):
        raise ValueError("Palette and legend labels must have equal length")

    handles = [
        Patch(facecolor=color, edgecolor="#555555", linewidth=0.45)
        for color in palette
    ]
    if reverse:
        handles = handles[::-1]
        labels = labels[::-1]

    legend = ax.legend(
        handles,
        labels,
        title=title,
        frameon=False,
        loc="lower left",
        bbox_to_anchor=(0.785, 0.16),
        borderaxespad=0,
        handlelength=1.45,
        handleheight=0.85,
        labelspacing=0.55,
        handletextpad=0.45,
        fontsize=10,
        title_fontsize=11,
    )
    legend._legend_box.align = "left"
    return legend
