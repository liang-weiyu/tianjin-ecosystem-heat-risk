"""Sensitivity of ecosystem heat-risk coastal gradients to equal-width bins.

The analysis uses the strict-land common-support cell table and compares
10-km and 20-km bins over the complete observed coastal-distance range.
Spatial uncertainty for adjacent peak contrasts is estimated with a
stratified 10-km spatial-block bootstrap.
"""

from __future__ import annotations

import json
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties


PROJECT_ROOT = STUDY_ROOT
INPUT_CSV = (
    PROJECT_ROOT
    / "strict_land_recalculation_20260806"
    / "06_coastal_inland_gradient"
    / "HEVR_grid_cells_with_coastal_distance.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "strict_land_recalculation_20260806"
    / "06_coastal_inland_gradient"
    / "sensitivity_equal_width_bins_R"
)

R_COLOR = "#D87355"
R_FILL = "#F2C9BD"
BOOTSTRAP_REPLICATES = 4999
RANDOM_SEED = 20260908
SPATIAL_BLOCK_M = 10_000.0
EXPORT_DPI = 600


def make_bins(max_distance: float, width: int) -> tuple[np.ndarray, list[str]]:
    upper = int(np.ceil(max_distance / width) * width)
    edges = np.arange(0, upper + width, width, dtype=float)
    labels = [f"{int(a)}-{int(b)}" for a, b in zip(edges[:-1], edges[1:])]
    return edges, labels


def summarize_width(frame: pd.DataFrame, width: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    edges, labels = make_bins(frame["distance_to_coast_km"].max(), width)
    work = frame.copy()
    work["distance_band_km"] = pd.cut(
        work["distance_to_coast_km"],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
        ordered=True,
    )
    # Keep the categorical distance order (coast to inland), rather than the
    # order in which cells happen to occur in the source raster table.
    grouped = work.groupby("distance_band_km", observed=True, sort=True)
    summary = grouped.agg(
        grid_cells=("R", "size"),
        distance_min_observed_km=("distance_to_coast_km", "min"),
        distance_max_observed_km=("distance_to_coast_km", "max"),
        distance_median_km=("distance_to_coast_km", "median"),
        value_mean=("R", "mean"),
        value_sd=("R", "std"),
        value_q1=("R", lambda x: x.quantile(0.25)),
        value_median=("R", "median"),
        value_q3=("R", lambda x: x.quantile(0.75)),
        value_min=("R", "min"),
        value_max=("R", "max"),
    ).reset_index()
    summary.insert(0, "bin_width_km", width)
    summary["percent_of_valid_cells"] = summary["grid_cells"] / len(work) * 100.0
    summary["value_iqr"] = summary["value_q3"] - summary["value_q1"]
    return work, summary


def block_bootstrap_difference(
    work: pd.DataFrame,
    band_a: str,
    band_b: str,
    rng: np.random.Generator,
) -> dict[str, float | int | str]:
    """Bootstrap median(A)-median(B), sampling 10-km blocks within each band."""

    arrays: dict[str, list[np.ndarray]] = {}
    for band in (band_a, band_b):
        selected = work.loc[work["distance_band_km"].astype(str) == band]
        arrays[band] = [
            group["R"].to_numpy()
            for _, group in selected.groupby("spatial_block_id", sort=False)
        ]
        if not arrays[band]:
            raise ValueError(f"No cells in band {band}")

    observed = (
        work.loc[work["distance_band_km"].astype(str) == band_a, "R"].median()
        - work.loc[work["distance_band_km"].astype(str) == band_b, "R"].median()
    )
    differences = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for iteration in range(BOOTSTRAP_REPLICATES):
        medians = []
        for band in (band_a, band_b):
            blocks = arrays[band]
            indices = rng.integers(0, len(blocks), size=len(blocks))
            resample = np.concatenate([blocks[index] for index in indices])
            medians.append(float(np.median(resample)))
        differences[iteration] = medians[0] - medians[1]

    lower, upper = np.quantile(differences, [0.025, 0.975])
    return {
        "band_a": band_a,
        "band_b": band_b,
        "contrast": f"{band_a} minus {band_b}",
        "observed_median_difference": float(observed),
        "bootstrap_ci95_lower": float(lower),
        "bootstrap_ci95_upper": float(upper),
        "bootstrap_probability_difference_gt_0": float(np.mean(differences > 0)),
        "spatial_blocks_band_a": len(arrays[band_a]),
        "spatial_blocks_band_b": len(arrays[band_b]),
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }


def configure_fonts() -> tuple[FontProperties, FontProperties]:
    plt.rcParams.update({
        "font.family": "Times New Roman",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    chinese = FontProperties(fname=font_path("chinese"), size=10.5)
    english = FontProperties(fname=font_path("english"), size=10.5)
    return chinese, english


def plot_sensitivity(summaries: dict[int, pd.DataFrame]) -> Path:
    chinese, english = configure_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.5), sharey=True)

    for axis, panel, width in zip(axes, ("a", "b"), (10, 20)):
        data = summaries[width]
        x = np.arange(len(data))
        axis.fill_between(
            x,
            data["value_q1"].to_numpy(),
            data["value_q3"].to_numpy(),
            color=R_FILL,
            alpha=0.48,
            linewidth=0,
            zorder=1,
        )
        axis.plot(
            x,
            data["value_median"].to_numpy(),
            color=R_COLOR,
            linewidth=1.8,
            marker="o",
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=R_COLOR,
            markeredgewidth=1.1,
            zorder=3,
        )
        peak_position = int(data["value_median"].to_numpy().argmax())
        axis.scatter(
            [peak_position],
            [data.loc[peak_position, "value_median"]],
            s=34,
            color=R_COLOR,
            zorder=4,
        )
        axis.text(
            0.0,
            1.025,
            f"({panel})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontproperties=english,
        )
        axis.text(
            0.075,
            1.025,
            f"{width} km等距分组",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontproperties=chinese,
        )
        axis.set_xticks(x)
        axis.set_xticklabels(data["distance_band_km"], rotation=45, ha="right", fontproperties=english)
        axis.set_xlabel("距海岸线距离（km）", fontproperties=chinese)
        axis.grid(axis="y", color="#E5E5E5", linewidth=0.7)
        axis.spines[["top", "right"]].set_visible(False)
        axis.tick_params(direction="out", labelsize=9.5)

    axes[0].set_ylabel("生态系统热风险指数（R）", fontproperties=chinese)
    fig.subplots_adjust(left=0.085, right=0.985, top=0.90, bottom=0.26, wspace=0.12)
    stem = OUTPUT_DIR / "R_coastal_gradient_equal_10km_20km_sensitivity"
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


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT_CSV)
    required = {"x_albers_m", "y_albers_m", "distance_to_coast_km", "R"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame.dropna(subset=list(required)).copy()
    frame["spatial_block_id"] = (
        np.floor(frame["x_albers_m"] / SPATIAL_BLOCK_M).astype(int).astype(str)
        + "_"
        + np.floor(frame["y_albers_m"] / SPATIAL_BLOCK_M).astype(int).astype(str)
    )

    works: dict[int, pd.DataFrame] = {}
    summaries: dict[int, pd.DataFrame] = {}
    for width in (10, 20):
        work, summary = summarize_width(frame, width)
        if int(summary["grid_cells"].sum()) != len(frame):
            raise AssertionError(f"Cell-count mismatch for {width}-km bins")
        works[width] = work
        summaries[width] = summary

    combined_summary = pd.concat([summaries[10], summaries[20]], ignore_index=True)
    combined_summary.to_csv(
        OUTPUT_DIR / "01_R_equal_width_bin_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rng = np.random.default_rng(RANDOM_SEED)
    contrasts = [
        {"bin_width_km": 10, **block_bootstrap_difference(works[10], "10-20", "0-10", rng)},
        {"bin_width_km": 10, **block_bootstrap_difference(works[10], "10-20", "20-30", rng)},
        {"bin_width_km": 20, **block_bootstrap_difference(works[20], "0-20", "20-40", rng)},
    ]
    contrast_frame = pd.DataFrame(contrasts)
    contrast_frame.to_csv(
        OUTPUT_DIR / "02_peak_adjacent_spatial_block_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )

    peak_rows = []
    for width, summary in summaries.items():
        row = summary.loc[summary["value_median"].idxmax()]
        peak_rows.append({
            "bin_width_km": width,
            "peak_band_km": str(row["distance_band_km"]),
            "peak_grid_cells": int(row["grid_cells"]),
            "peak_R_median": float(row["value_median"]),
            "peak_R_q1": float(row["value_q1"]),
            "peak_R_q3": float(row["value_q3"]),
        })
    peak_frame = pd.DataFrame(peak_rows)
    peak_frame.to_csv(
        OUTPUT_DIR / "03_peak_stability_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )

    figure = plot_sensitivity(summaries)
    qa = {
        "input_csv": str(INPUT_CSV),
        "valid_cells": int(len(frame)),
        "distance_min_km": float(frame["distance_to_coast_km"].min()),
        "distance_max_km": float(frame["distance_to_coast_km"].max()),
        "bin_widths_km": [10, 20],
        "spatial_bootstrap_block_km": SPATIAL_BLOCK_M / 1000.0,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "random_seed": RANDOM_SEED,
        "figure": str(figure),
        "important_caveat": (
            "The final 120-130/140 km bin contains only the observed 120-124.32 km tail "
            "and should not be interpreted as a complete-width spatial sample."
        ),
    }
    (OUTPUT_DIR / "00_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    readme = f"""# 生态系统热风险海陆梯度等距分组敏感性检验

- 数据：严格陆地共同支撑的 {len(frame):,} 个1 km格网。
- 距岸范围：{frame['distance_to_coast_km'].min():.2f}–{frame['distance_to_coast_km'].max():.2f} km。
- 方案：全程10 km等距分组与全程20 km等距分组，区间均采用左闭右开。
- 描述统计：组内中位数与P25–P75。
- 峰值相邻比较：采用10 km规则空间块分层自助法，重复{BOOTSTRAP_REPLICATES}次；该区间反映空间分块抽样不确定性，而非因果效应。
- 注意：最末距离带仅覆盖120–124.32 km，样本量较小，不是完整宽度距离带。

## 判读原则

若两种等距方案的最高中位数均位于近岸0–20 km范围，可认为“近岸相对高值”对分组宽度稳健；只有10 km方案能够进一步定位10–20 km。20 km方案不能独立证明10–20 km峰值，因为其将0–10 km与10–20 km合并。
"""
    (OUTPUT_DIR / "README.md").write_text(readme, encoding="utf-8-sig")

    print(peak_frame.to_string(index=False))
    print(contrast_frame.to_string(index=False))
    print(f"output_dir={OUTPUT_DIR}")


if __name__ == "__main__":
    main()
