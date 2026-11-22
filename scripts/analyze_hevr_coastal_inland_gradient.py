"""Create a four-panel coastal-to-inland gradient analysis for H, E, V and R."""

from datetime import datetime
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.patches import Patch
from scipy import stats


PROJECT_ROOT = STUDY_ROOT
INDEX_PATHS = {
    "H": PROJECT_ROOT / "01_aligned_indices" / "01_H_heat_hazard_2022_aligned_common_1km.tif",
    "E": PROJECT_ROOT / "01_aligned_indices" / "02_E_ecosystem_exposure_2021_common_1km.tif",
    "V": PROJECT_ROOT / "01_aligned_indices" / "03_V_ecosystem_vulnerability_2021_common_1km.tif",
    "R": PROJECT_ROOT / "02_risk_results" / "04_R_ecosystem_heat_risk_HEV_product_continuous_1km.tif",
}
DISTANCE_PATH = (
    PROJECT_ROOT
    / "07_coastal_inland_gradient"
    / "distance_to_NaturalEarth_coastline_km_common_1km.tif"
)
COASTLINE_PATH = (
    PROJECT_ROOT
    / "07_coastal_inland_gradient"
    / "NaturalEarth_coastline_clip_Albers.gpkg"
)
OUTPUT_DIR = PROJECT_ROOT / "07_coastal_inland_gradient" / "current_updated_HEVR"

BIN_EDGES = np.asarray([0.0, 10.0, 20.0, 40.0, 60.0, 80.0, np.inf])
BIN_LABELS = ["0-10", "10-20", "20-40", "40-60", "60-80", ">80"]
INDEX_NAMES = {
    "H": "热危害指数",
    "E": "生态系统暴露指数",
    "V": "生态系统脆弱性指数",
    "R": "生态系统热风险指数",
}
COLORS = {
    "H": "#5B8FB9",
    "E": "#D9992E",
    "V": "#9673B1",
    "R": "#D87355",
}
EXPORT_DPI = 600


def configure_fonts():
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
    })


def read_common_frame():
    arrays = {}
    profiles = {}
    with rasterio.open(DISTANCE_PATH) as source:
        distance = source.read(1).astype(np.float64)
        profiles["distance"] = source.profile.copy()
        distance_nodata = source.nodata
        rows, cols = np.indices(source.shape)
        transform = source.transform
        xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
        ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e

    valid = np.isfinite(distance)
    if distance_nodata is not None:
        valid &= distance != distance_nodata

    reference = profiles["distance"]
    alignment = {}
    for code, path in INDEX_PATHS.items():
        with rasterio.open(path) as source:
            array = source.read(1).astype(np.float64)
            arrays[code] = array
            alignment[code] = {
                "shape": source.shape == (reference["height"], reference["width"]),
                "crs": source.crs == reference["crs"],
                "transform": source.transform.almost_equals(reference["transform"]),
                "resolution": source.res == (
                    abs(reference["transform"].a),
                    abs(reference["transform"].e),
                ),
            }
            if not all(alignment[code].values()):
                raise ValueError(f"{code} alignment failed: {alignment[code]}")
            index_valid = np.isfinite(array)
            if source.nodata is not None:
                index_valid &= array != source.nodata
            valid &= index_valid

    data = {
        "row": rows[valid].astype(np.int32),
        "col": cols[valid].astype(np.int32),
        "x_albers_m": xs[valid],
        "y_albers_m": ys[valid],
        "distance_to_coast_km": distance[valid],
    }
    for code, array in arrays.items():
        data[code] = array[valid]
    frame = pd.DataFrame(data)
    frame["distance_band_km"] = pd.cut(
        frame["distance_to_coast_km"],
        bins=BIN_EDGES,
        labels=BIN_LABELS,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    if frame["distance_band_km"].isna().any():
        raise ValueError("Some common-valid cells were not assigned to a band.")
    return frame, alignment


def summarize(frame):
    band_rows = []
    trend_rows = []
    for code in INDEX_PATHS:
        x = frame["distance_to_coast_km"].to_numpy()
        y = frame[code].to_numpy()
        spearman = stats.spearmanr(x, y)
        linear = stats.linregress(x, y)
        for label in BIN_LABELS:
            selected = frame["distance_band_km"] == label
            values = frame.loc[selected, code].to_numpy()
            distances = frame.loc[selected, "distance_to_coast_km"].to_numpy()
            q1, median, q3 = np.quantile(values, [0.25, 0.50, 0.75])
            band_rows.append({
                "index": code,
                "index_name_cn": INDEX_NAMES[code],
                "distance_band_km": label,
                "grid_cells": int(values.size),
                "percent_of_valid_cells": values.size / len(frame) * 100.0,
                "distance_median_km": float(np.median(distances)),
                "value_mean": float(np.mean(values)),
                "value_sd": float(np.std(values, ddof=1)),
                "value_q1": float(q1),
                "value_median": float(median),
                "value_q3": float(q3),
                "value_iqr": float(q3 - q1),
                "value_min": float(np.min(values)),
                "value_max": float(np.max(values)),
            })
        trend_rows.append({
            "index": code,
            "index_name_cn": INDEX_NAMES[code],
            "valid_grid_cells": int(len(frame)),
            "spearman_rho": float(spearman.statistic),
            "spearman_p_descriptive": float(spearman.pvalue),
            "ols_slope_per_km": float(linear.slope),
            "ols_slope_per_10km": float(linear.slope * 10.0),
            "ols_r": float(linear.rvalue),
            "ols_r_squared": float(linear.rvalue ** 2),
            "ols_p_descriptive": float(linear.pvalue),
        })
    return pd.DataFrame(band_rows), pd.DataFrame(trend_rows)


def add_panel_title(axis, panel, code, chinese, english):
    title = HPacker(
        children=[
            TextArea(f"({panel})", textprops={"fontproperties": english}),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    axis.add_artist(AnchoredOffsetbox(
        loc="lower left",
        child=title,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(0.0, 1.025),
        bbox_transform=axis.transAxes,
    ))


def save_figure(band_summary, trend_summary):
    configure_fonts()
    chinese = FontProperties(fname=font_path("chinese"), size=10.5)
    english = FontProperties(fname=font_path("english"), size=10.5)
    mixed = FontProperties(family=["Times New Roman", "SimSun"], size=10.5)
    legend_font = FontProperties(fname=font_path("chinese"), size=10)

    fig, axes = plt.subplots(2, 2, figsize=(12.2, 8.3), sharex=False)
    x = np.arange(len(BIN_LABELS))
    panels = ["a", "b", "c", "d"]

    for axis, code, panel in zip(axes.flat, INDEX_PATHS, panels):
        data = band_summary[band_summary["index"] == code].reset_index(drop=True)
        color = COLORS[code]
        axis.fill_between(
            x,
            data["value_q1"].to_numpy(),
            data["value_q3"].to_numpy(),
            color=color,
            alpha=0.18,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            data["value_median"].to_numpy(),
            color=color,
            linewidth=1.7,
            marker="o",
            markersize=4.8,
            markerfacecolor="white",
            markeredgecolor=color,
            markeredgewidth=1.1,
            zorder=3,
        )
        rho = trend_summary.loc[
            trend_summary["index"] == code, "spearman_rho"
        ].iloc[0]
        axis.text(
            0.97,
            0.95,
            f"Spearman ρ = {rho:.3f}",
            transform=axis.transAxes,
            ha="right",
            va="top",
            fontproperties=english,
            color="#333333",
        )
        add_panel_title(axis, panel, code, chinese, english)
        axis.set_ylabel(
            f"{INDEX_NAMES[code]}（{code}）",
            fontproperties=mixed,
        )
        axis.set_xticks(x, BIN_LABELS)
        axis.set_xlabel("距海岸线距离（km）", fontproperties=mixed)
        axis.grid(color="#E4E4E4", linewidth=0.65)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out")

    legend_handles = [
        Line2D(
            [0], [0], color="#666666", marker="o", markerfacecolor="white",
            linewidth=1.5, markersize=5, label="分组中位数",
        ),
        Patch(facecolor="#BDBDBD", edgecolor="none", alpha=0.35, label="四分位距"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=2,
        frameon=False,
        prop=legend_font,
        bbox_to_anchor=(0.5, 0.005),
    )
    fig.subplots_adjust(left=0.085, right=0.985, top=0.95, bottom=0.10, hspace=0.38, wspace=0.20)

    stem = OUTPUT_DIR / "HEVR_coastal_inland_gradient_median_IQR_2x2"
    fig.savefig(stem.with_suffix(".png"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), dpi=EXPORT_DPI, bbox_inches="tight")
    fig.savefig(
        stem.with_suffix(".tif"),
        dpi=EXPORT_DPI,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    plt.close(fig)
    return stem.with_suffix(".png")


def write_readme(frame, band_summary, trend_summary, alignment):
    lines = [
        "# Updated H-E-V-R coastal-to-inland gradient",
        "",
        f"- Common valid 1-km cells: {len(frame)}.",
        f"- Coastline: `{COASTLINE_PATH.name}` (Natural Earth 1:10m).",
        f"- Distance range: {frame['distance_to_coast_km'].min():.4f}-{frame['distance_to_coast_km'].max():.4f} km.",
        f"- Alignment checks: {alignment}.",
        f"- Input update time: {datetime.fromtimestamp(INDEX_PATHS['R'].stat().st_mtime).isoformat(timespec='seconds')}.",
        "- The vulnerability layer remains the annual-NDMI version requested by the user.",
        "",
        "## Spearman gradient coefficients",
        "",
    ]
    for row in trend_summary.itertuples(index=False):
        lines.append(f"- {row.index} ({row.index_name_cn}): rho={row.spearman_rho:.6f}.")
    lines.extend([
        "",
        "## Validation caveats",
        "",
        "- Medians and P25-P75 are the primary band summaries.",
        "- Conventional p-values in the trend CSV are descriptive only and are not spatially corrected.",
        "- The 0-10 km band has a much smaller sample than inland bands.",
        "- Natural Earth 1:10m generalizes reclaimed and fine-scale shoreline geometry.",
    ])
    (OUTPUT_DIR / "README_updated_HEVR_coastal_inland_gradient.md").write_text(
        "\n".join(lines), encoding="utf-8-sig"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame, alignment = read_common_frame()
    band_summary, trend_summary = summarize(frame)

    expected_total = len(frame) * len(INDEX_PATHS)
    if int(band_summary["grid_cells"].sum()) != expected_total:
        raise AssertionError("HEVR band counts do not match common support.")

    frame.to_csv(
        OUTPUT_DIR / "HEVR_grid_cells_with_coastal_distance.csv",
        index=False,
        encoding="utf-8-sig",
    )
    band_summary.to_csv(
        OUTPUT_DIR / "HEVR_coastal_inland_gradient_by_bin.csv",
        index=False,
        encoding="utf-8-sig",
    )
    trend_summary.to_csv(
        OUTPUT_DIR / "HEVR_coastal_distance_trend_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_path = save_figure(band_summary, trend_summary)
    write_readme(frame, band_summary, trend_summary, alignment)

    print(trend_summary.to_string(index=False))
    print(band_summary.to_string(index=False))
    print(f"figure={figure_path}")
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
