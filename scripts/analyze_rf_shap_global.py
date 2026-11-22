"""RF global interpretation with SHAP main effects, thresholds, and interactions."""

from __future__ import annotations

import json
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path
import time

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap


WORKSPACE = STUDY_ROOT.parent
PROJECT = STUDY_ROOT
STRICT = PROJECT / "strict_land_recalculation_20260806"
INPUT = (
    STRICT
    / "08_background_driver_model"
    / "02_tables"
    / "background_driver_model_table_13_factors.csv"
)
PARAMETERS = (
    STRICT
    / "08_background_driver_model"
    / "06_global_ml_repeated_spatial_block_cv_11f"
    / "02_fixed_parameters.json"
)
CV_SUMMARY = (
    STRICT
    / "08_background_driver_model"
    / "06_global_ml_repeated_spatial_block_cv_11f"
    / "05_model_performance_summary_by_block_size.csv"
)
OUTPUT = STRICT / "08_background_driver_model" / "08_RF_SHAP_global_11f"
FIGURES = OUTPUT / "figures"

TARGET = "R"
FEATURES = [
    "elevation_m",
    "summer_wind_climatology_2010_2021_ms",
    "surface_albedo",
    "building_density_percent",
    "building_height_m",
    "building_height_sd_m",
    "svf",
    "road_density_km_km2",
    "nighttime_light",
    "population_density_km2",
    "summer_precip_climatology_2010_2021_mm",
]
CHINESE = {
    "elevation_m": "地表高程",
    "summer_wind_climatology_2010_2021_ms": "夏季平均风速",
    "surface_albedo": "地表反照率",
    "building_density_percent": "建筑覆盖率",
    "building_height_m": "平均建筑高度",
    "building_height_sd_m": "建筑高度离散度",
    "svf": "天空可视因子",
    "road_density_km_km2": "路网密度",
    "nighttime_light": "夜间灯光强度",
    "population_density_km2": "人口密度",
    "summer_precip_climatology_2010_2021_mm": "夏季累计降水",
}
AXIS_LABELS = {
    "elevation_m": "地表高程（m）",
    "summer_wind_climatology_2010_2021_ms": "夏季平均风速（m/s）",
    "surface_albedo": "地表反照率",
    "building_density_percent": "建筑覆盖率（%）",
    "building_height_m": "平均建筑高度（m）",
    "building_height_sd_m": "建筑高度离散度（m）",
    "svf": "天空可视因子",
    "road_density_km_km2": "路网密度（km/km²）",
    "nighttime_light": "夜间灯光强度",
    "population_density_km2": "人口密度（人/km²）",
    "summer_precip_climatology_2010_2021_mm": "夏季累计降水（mm）",
}
ABBR = {
    "elevation_m": "ELEV",
    "summer_wind_climatology_2010_2021_ms": "WS",
    "surface_albedo": "ALB",
    "building_density_percent": "BCR",
    "building_height_m": "MBH",
    "building_height_sd_m": "BHSD",
    "svf": "SVF",
    "road_density_km_km2": "RD",
    "nighttime_light": "NTL",
    "population_density_km2": "PD",
    "summer_precip_climatology_2010_2021_mm": "PREC",
}

# Fixed SHAP visual encoding shared by every exported figure.  Green is
# reserved for model contribution/response so it does not compete with the
# established blue-yellow-purple H/E/V pathway encoding.
VALUE_CMAP = LinearSegmentedColormap.from_list(
    "shap_value_greens",
    [
        "#E9EFA7", "#C4DE97", "#9CD287", "#5EB265",
        "#28A058", "#078C4D", "#008344", "#186E4B",
    ],
)  # low-to-high sequence from the user-supplied eight-level RGB reference
IMPORTANCE_COLOR = "#28A058"
INTERACTION_CMAP = LinearSegmentedColormap.from_list(
    "shap_interaction_greens",
    [
        "#E9EFA7", "#C4DE97", "#9CD287", "#5EB265",
        "#28A058", "#078C4D", "#008344", "#186E4B",
    ],
)  # low interaction: light green; high interaction: dark green
THRESHOLD_COLOR = "#E76F51"
THRESHOLD_BAND_COLOR = "#F2A65A"
RESPONSE_COLOR = "#222222"
ZERO_COLOR = "#777777"

RANDOM_SEED = 20260814
INTERACTION_SAMPLE = 1_500
THRESHOLD_BOOTSTRAPS = 200


def configure_fonts() -> None:
    mpl.rcParams.update(
        {
            "font.family": ["Times New Roman", "SimSun"],
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def save_figure(figure, name: str) -> None:
    stem = FIGURES / name
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tif"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def build_blocks(frame: pd.DataFrame, block_km: int = 10) -> pd.Series:
    size = block_km * 1_000.0
    origin_x = np.floor(frame["x_m"].min() / size) * size
    origin_y = np.floor(frame["y_m"].min() / size) * size
    column = np.floor((frame["x_m"] - origin_x) / size).astype(int)
    row = np.floor((frame["y_m"] - origin_y) / size).astype(int)
    return "B10_" + column.astype(str) + "_" + row.astype(str)


def normalize_shap_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP main-effect shape: {array.shape}")
    return array


def normalize_interaction_array(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3:
        raise ValueError(f"Unexpected SHAP interaction shape: {array.shape}")
    return array


def quantile_curve(x: np.ndarray, y: np.ndarray, bins: int = 35) -> pd.DataFrame:
    ranks = pd.qcut(pd.Series(x), q=bins, duplicates="drop")
    result = (
        pd.DataFrame({"x": x, "y": y, "bin": ranks})
        .groupby("bin", observed=True)
        .agg(x=("x", "median"), y=("y", "median"), y_mean=("y", "mean"), n=("y", "size"))
        .reset_index(drop=True)
    )
    return result


def piecewise_fit(x: np.ndarray, y: np.ndarray) -> dict | None:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if len(x) < 100 or np.unique(x).size < 20:
        return None
    lower, upper = np.quantile(x, [0.02, 0.98])
    central = (x >= lower) & (x <= upper)
    x = x[central]
    y = y[central]
    mean = x.mean()
    standard_deviation = x.std(ddof=0)
    if standard_deviation <= 0:
        return None
    standardized = (x - mean) / standard_deviation
    linear_design = np.column_stack([np.ones(len(x)), standardized])
    linear_coefficients, *_ = np.linalg.lstsq(linear_design, y, rcond=None)
    linear_residual = y - linear_design @ linear_coefficients
    linear_sse = max(float(linear_residual @ linear_residual), 1e-30)
    linear_bic = len(x) * np.log(linear_sse / len(x)) + 2 * np.log(len(x))

    candidates_original = np.unique(np.quantile(x, np.linspace(0.15, 0.85, 51)))
    best = None
    for threshold_original in candidates_original:
        threshold = (threshold_original - mean) / standard_deviation
        hinge = np.maximum(0.0, standardized - threshold)
        design = np.column_stack([np.ones(len(x)), standardized, hinge])
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        sse = max(float(residual @ residual), 1e-30)
        bic = len(x) * np.log(sse / len(x)) + 4 * np.log(len(x))
        if best is None or bic < best["bic_piecewise"]:
            best = {
                "threshold": float(threshold_original),
                "threshold_standardized": float(threshold),
                "bic_piecewise": float(bic),
                "coefficients": coefficients,
                "x_mean": float(mean),
                "x_sd": float(standard_deviation),
                "x_low": float(lower),
                "x_high": float(upper),
            }
    if best is None:
        return None
    best["bic_linear"] = float(linear_bic)
    best["delta_bic"] = float(linear_bic - best["bic_piecewise"])
    best["slope_before_per_sd"] = float(best["coefficients"][1])
    best["slope_after_per_sd"] = float(best["coefficients"][1] + best["coefficients"][2])

    grid = np.linspace(lower, upper, 1_000)
    grid_z = (grid - mean) / standard_deviation
    fitted = (
        best["coefficients"][0]
        + best["coefficients"][1] * grid_z
        + best["coefficients"][2] * np.maximum(0.0, grid_z - best["threshold_standardized"])
    )
    crossing = np.where(np.signbit(fitted[:-1]) != np.signbit(fitted[1:]))[0]
    if len(crossing):
        candidates = grid[crossing]
        best["nearest_zero_crossing"] = float(
            candidates[np.argmin(np.abs(candidates - best["threshold"]))]
        )
    else:
        best["nearest_zero_crossing"] = np.nan
    return best


def threshold_table(
    frame: pd.DataFrame,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)
    blocks = frame["block_10km"].to_numpy()
    unique_blocks = np.unique(blocks)
    block_indices = {block: np.where(blocks == block)[0] for block in unique_blocks}
    rows = []
    for feature_index, feature in enumerate(FEATURES):
        x = frame[feature].to_numpy(dtype=float)
        y = shap_values[:, feature_index]
        point = piecewise_fit(x, y)
        if point is None:
            continue
        bootstrap_thresholds = []
        bootstrap_delta_bic = []
        for _ in range(THRESHOLD_BOOTSTRAPS):
            sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
            sampled_indices = np.concatenate([block_indices[block] for block in sampled_blocks])
            fitted = piecewise_fit(x[sampled_indices], y[sampled_indices])
            if fitted is not None and fitted["delta_bic"] > 0:
                bootstrap_thresholds.append(fitted["threshold"])
                bootstrap_delta_bic.append(fitted["delta_bic"])
        bootstrap_thresholds = np.asarray(bootstrap_thresholds, dtype=float)
        threshold_percentile = float((x <= point["threshold"]).mean() * 100.0)
        rows.append(
            {
                "factor": feature,
                "chinese_name": CHINESE[feature],
                "threshold": point["threshold"],
                "threshold_percentile": threshold_percentile,
                "threshold_ci95_low": np.quantile(bootstrap_thresholds, 0.025) if len(bootstrap_thresholds) else np.nan,
                "threshold_ci95_high": np.quantile(bootstrap_thresholds, 0.975) if len(bootstrap_thresholds) else np.nan,
                "bootstrap_valid_fraction": len(bootstrap_thresholds) / THRESHOLD_BOOTSTRAPS,
                "delta_bic_piecewise_vs_linear": point["delta_bic"],
                "median_bootstrap_delta_bic": np.median(bootstrap_delta_bic) if bootstrap_delta_bic else np.nan,
                "slope_before_per_feature_sd": point["slope_before_per_sd"],
                "slope_after_per_feature_sd": point["slope_after_per_sd"],
                "nearest_fitted_zero_crossing": point["nearest_zero_crossing"],
                "mean_shap_below_threshold": y[x <= point["threshold"]].mean(),
                "mean_shap_above_threshold": y[x > point["threshold"]].mean(),
                "strong_breakpoint_delta_bic_ge_10": point["delta_bic"] >= 10,
            }
        )
        print(
            f"threshold {feature}: {point['threshold']:.5g}, "
            f"CI=({rows[-1]['threshold_ci95_low']:.5g}, {rows[-1]['threshold_ci95_high']:.5g}), "
            f"deltaBIC={point['delta_bic']:.1f}",
            flush=True,
        )
    return pd.DataFrame(rows).sort_values("delta_bic_piecewise_vs_linear", ascending=False)


def interaction_sample(frame: pd.DataFrame) -> np.ndarray:
    rng = np.random.default_rng(RANDOM_SEED)
    all_indices = np.arange(len(frame))
    mandatory = []
    for _, group in frame.groupby("block_10km"):
        mandatory.append(rng.choice(group.index.to_numpy(), size=1)[0])
    mandatory = np.asarray(mandatory, dtype=int)
    remaining = np.setdiff1d(all_indices, mandatory, assume_unique=False)
    additional = rng.choice(
        remaining,
        size=min(INTERACTION_SAMPLE - len(mandatory), len(remaining)),
        replace=False,
    )
    return np.sort(np.concatenate([mandatory, additional]))


def plot_importance(importance: pd.DataFrame) -> None:
    configure_fonts()
    data = importance.sort_values("mean_abs_shap", ascending=False)
    figure, axis = plt.subplots(figsize=(7.2, 5.2))
    colors = VALUE_CMAP(np.linspace(0.98, 0.30, len(data)))
    axis.barh(
        data["chinese_name"],
        data["mean_abs_shap"],
        color=colors,
        edgecolor="#315C48",
        linewidth=0.5,
    )
    axis.invert_yaxis()
    axis.set_xlabel(
        "Mean|SHAP|（对生态系统热风险指数的平均绝对贡献）",
        fontsize=10.5,
    )
    axis.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    axis.set_axisbelow(True)
    maximum = float(data["mean_abs_shap"].max())
    axis.set_xlim(0, maximum * 1.16)
    for position, (_, row) in enumerate(data.iterrows()):
        axis.text(
            row["mean_abs_shap"] + maximum * 0.010,
            position,
            f"{row['importance_share_percent']:.2f}%",
            va="center",
            ha="left",
            fontsize=9.0,
            color="#222222",
        )
    axis.tick_params(axis="both", labelsize=9.5, colors="#222222")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color("#333333")
        axis.spines[name].set_linewidth(0.8)
    figure.subplots_adjust(left=0.29, right=0.96, bottom=0.14, top=0.97)
    save_figure(figure, "RF_mean_absolute_SHAP_importance")


def plot_beeswarm(frame: pd.DataFrame, shap_values: np.ndarray) -> None:
    configure_fonts()
    chinese_frame = frame[FEATURES].rename(columns=CHINESE)
    plt.figure(figsize=(8.4, 6.2))
    shap.summary_plot(
        shap_values,
        chinese_frame,
        max_display=len(FEATURES),
        show=False,
        plot_size=None,
        cmap=VALUE_CMAP,
    )
    current = plt.gcf()
    main_axis = current.axes[0]
    current.canvas.draw()
    for collection in main_axis.collections:
        if not isinstance(collection, PathCollection):
            continue
        facecolors = collection.get_facecolors()
        if facecolors.size:
            collection.set_facecolors(facecolors.copy())
        collection.set_edgecolors("none")
        collection.set_linewidths(0)
        collection.set_alpha(0.42)
        sizes = collection.get_sizes()
        if sizes.size:
            collection.set_sizes(sizes * 0.78)
    main_axis.set_xlabel("SHAP值（对模型输出的影响）", fontsize=10.5)
    main_axis.tick_params(axis="both", labelsize=9.5, colors="#222222")
    main_axis.spines["top"].set_visible(False)
    main_axis.spines["right"].set_visible(False)
    for name in ("left", "bottom"):
        main_axis.spines[name].set_color("#333333")
        main_axis.spines[name].set_linewidth(0.8)
    if len(current.axes) > 1:
        colorbar_axis = current.axes[-1]
        colorbar_axis.set_ylabel("特征值")
        ticks = colorbar_axis.get_yticks()
        if len(ticks) == 2:
            colorbar_axis.set_yticks(ticks, ["低", "高"])
    current.subplots_adjust(left=0.31, right=0.92, bottom=0.12, top=0.98)
    save_figure(current, "RF_SHAP_beeswarm_all_cells")


def plot_dependence_all(
    frame: pd.DataFrame,
    shap_values: np.ndarray,
    thresholds: pd.DataFrame,
    importance: pd.DataFrame,
) -> None:
    configure_fonts()
    threshold_lookup = thresholds.set_index("factor")
    ordered_features = importance.sort_values(
        ["mean_abs_shap", "factor"], ascending=[False, True]
    )["factor"].tolist()
    figure, axes = plt.subplots(3, 4, figsize=(13.2, 9.2))
    for axis, feature in zip(axes.ravel(), ordered_features):
        feature_index = FEATURES.index(feature)
        x = frame[feature].to_numpy()
        y = shap_values[:, feature_index]
        lower, upper = np.quantile(x, [0.02, 0.98])
        if lower == upper:
            lower, upper = float(np.min(x)), float(np.max(x) + 1e-12)
        axis.scatter(
            x,
            y,
            c=x,
            cmap=VALUE_CMAP,
            norm=Normalize(lower, upper),
            s=5,
            alpha=0.20,
            linewidth=0,
            rasterized=True,
        )
        curve = quantile_curve(x, y)
        axis.plot(curve["x"], curve["y"], color=RESPONSE_COLOR, linewidth=1.4)
        axis.axhline(0, color=ZERO_COLOR, linewidth=0.7, linestyle="--")
        if feature in threshold_lookup.index:
            row = threshold_lookup.loc[feature]
            axis.axvspan(
                row["threshold_ci95_low"],
                row["threshold_ci95_high"],
                color=THRESHOLD_BAND_COLOR,
                alpha=0.18,
            )
            axis.axvline(row["threshold"], color=THRESHOLD_COLOR, linewidth=1.0, linestyle="--")
        axis.set_xlabel(AXIS_LABELS[feature], fontsize=8.5)
        axis.set_ylabel("SHAP值", fontsize=8.5)
        axis.tick_params(labelsize=8)
        axis.grid(color="#E1E1E1", linewidth=0.4)
        axis.set_axisbelow(True)
    axes.ravel()[-1].axis("off")
    for label, axis in zip(
        [f"({chr(97 + index)})" for index in range(len(ordered_features))],
        axes.ravel(),
    ):
        axis.text(-0.15, 1.03, label, transform=axis.transAxes, fontsize=10, fontweight="normal")
    colorbar_axis = figure.add_axes([0.945, 0.18, 0.012, 0.66])
    colorbar = figure.colorbar(
        mpl.cm.ScalarMappable(norm=Normalize(0, 1), cmap=VALUE_CMAP),
        cax=colorbar_axis,
    )
    colorbar.set_ticks([0, 1])
    colorbar.set_ticklabels(["低", "高"])
    colorbar.set_label("特征值", fontsize=8.5)
    colorbar.ax.tick_params(labelsize=8)
    figure.subplots_adjust(left=0.08, right=0.925, bottom=0.08, top=0.97, hspace=0.43, wspace=0.32)
    save_figure(figure, "RF_SHAP_dependence_all11_with_thresholds")


def plot_dependence_top6_colored(
    frame: pd.DataFrame,
    shap_values: np.ndarray,
    importance: pd.DataFrame,
    interaction_matrix: np.ndarray,
    thresholds: pd.DataFrame,
) -> None:
    configure_fonts()
    top_features = importance.head(6)["factor"].tolist()
    threshold_lookup = thresholds.set_index("factor")
    figure, axes = plt.subplots(2, 3, figsize=(12.4, 7.3))
    for axis, feature in zip(axes.ravel(), top_features):
        index = FEATURES.index(feature)
        row = interaction_matrix[index].copy()
        row[index] = -np.inf
        partner_index = int(np.argmax(row))
        partner = FEATURES[partner_index]
        x = frame[feature].to_numpy()
        y = shap_values[:, index]
        color = frame[partner].to_numpy()
        lower, upper = np.quantile(color, [0.02, 0.98])
        scatter = axis.scatter(
            x,
            y,
            c=color,
            cmap=VALUE_CMAP,
            norm=Normalize(lower, upper),
            s=6,
            alpha=0.30,
            linewidth=0,
            rasterized=True,
        )
        curve = quantile_curve(x, y)
        axis.plot(curve["x"], curve["y"], color=RESPONSE_COLOR, linewidth=1.5)
        axis.axhline(0, color=ZERO_COLOR, linewidth=0.7, linestyle="--")
        if feature in threshold_lookup.index:
            threshold = threshold_lookup.loc[feature]
            axis.axvspan(
                threshold["threshold_ci95_low"],
                threshold["threshold_ci95_high"],
                color=THRESHOLD_BAND_COLOR,
                alpha=0.18,
            )
            axis.axvline(threshold["threshold"], color=THRESHOLD_COLOR, linewidth=1.0, linestyle="--")
        axis.set_xlabel(AXIS_LABELS[feature], fontsize=8.5)
        axis.set_ylabel("SHAP值", fontsize=8.5)
        axis.tick_params(labelsize=8)
        axis.grid(color="#E1E1E1", linewidth=0.4)
        axis.set_axisbelow(True)
        colorbar = figure.colorbar(scatter, ax=axis, fraction=0.044, pad=0.02)
        colorbar.ax.tick_params(labelsize=7)
        colorbar.set_label(CHINESE[partner], fontsize=7.5)
    for label, axis in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], axes.ravel()):
        axis.text(-0.15, 1.03, label, transform=axis.transAxes, fontsize=10, fontweight="normal")
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.10, top=0.97, hspace=0.32, wspace=0.42)
    save_figure(figure, "RF_SHAP_dependence_top6_interaction_colored")


def plot_interactions(interaction_matrix: np.ndarray, pairs: pd.DataFrame) -> None:
    configure_fonts()
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(16.5, 5.6),
        gridspec_kw={"width_ratios": [1.0, 1.10]},
    )
    displayed = interaction_matrix.copy()
    np.fill_diagonal(displayed, np.nan)
    interaction_norm = Normalize(
        vmin=float(np.nanmin(displayed)),
        vmax=float(np.nanmax(displayed)),
    )
    image = axes[0].imshow(
        displayed,
        cmap=INTERACTION_CMAP,
        norm=interaction_norm,
        interpolation="nearest",
    )
    labels = [ABBR[feature] for feature in FEATURES]
    axes[0].set_xticks(np.arange(len(FEATURES)), labels, rotation=50, ha="right")
    axes[0].set_yticks(np.arange(len(FEATURES)), labels)
    axes[0].tick_params(length=0, labelsize=8)
    colorbar = figure.colorbar(image, ax=axes[0], fraction=0.045, pad=0.03)
    colorbar.set_label("Mean |SHAP interaction|", fontsize=8.5)

    top = pairs.head(10).sort_values("mean_abs_shap_interaction", ascending=True)
    bar_colors = INTERACTION_CMAP(
        interaction_norm(top["mean_abs_shap_interaction"].to_numpy())
    )
    axes[1].barh(
        top["pair_chinese"],
        top["mean_abs_shap_interaction"],
        color=bar_colors,
        edgecolor="#4A4A4A",
        linewidth=0.5,
    )
    axes[1].set_xlabel("Mean |SHAP interaction|")
    axes[1].tick_params(axis="y", labelsize=8.5)
    axes[1].grid(axis="x", color="#D9D9D9", linewidth=0.55)
    axes[1].set_axisbelow(True)
    axes[0].text(-0.13, 1.03, "(a)", transform=axes[0].transAxes, fontsize=12, fontweight="normal")
    axes[1].text(-0.18, 1.03, "(b)", transform=axes[1].transAxes, fontsize=12, fontweight="normal")
    figure.subplots_adjust(left=0.055, right=0.985, bottom=0.17, top=0.95, wspace=0.82)
    save_figure(figure, "RF_SHAP_interaction_heatmap_and_top_pairs")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    required = ["cell_id", "x_m", "y_m", TARGET, *FEATURES]
    if frame[required].isna().any().any():
        raise ValueError("Model table contains missing values")
    if frame["cell_id"].duplicated().any():
        raise ValueError("cell_id is not unique")
    frame = frame.copy()
    frame["block_10km"] = build_blocks(frame)
    parameters = json.loads(PARAMETERS.read_text(encoding="utf-8"))["RF"]
    model = RandomForestRegressor(
        random_state=RANDOM_SEED,
        n_jobs=-1,
        criterion="squared_error",
        **parameters,
    )
    x = frame[FEATURES]
    y = frame[TARGET].to_numpy(dtype=float)
    started = time.perf_counter()
    model.fit(x, y)
    predictions = model.predict(x)
    joblib.dump(model, OUTPUT / "RF_final_full_data_model.joblib", compress=3)
    training_metrics = {
        "r2": float(r2_score(y, predictions)),
        "rmse": float(np.sqrt(mean_squared_error(y, predictions))),
        "mae": float(mean_absolute_error(y, predictions)),
        "fit_seconds": time.perf_counter() - started,
        "note": "仅描述全样本最终模型拟合，不用于泛化性能评价",
    }
    (OUTPUT / "00_RF_full_model_training_metrics.json").write_text(
        json.dumps(training_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    explanation = explainer(x, check_additivity=True)
    shap_values = normalize_shap_array(explanation.values)
    base_value = float(np.asarray(explanation.base_values).reshape(-1)[0])
    additivity_error = np.max(np.abs(base_value + shap_values.sum(axis=1) - predictions))
    shap_table = pd.DataFrame(shap_values, columns=[f"SHAP_{feature}" for feature in FEATURES])
    shap_table.insert(0, "predicted_R", predictions)
    shap_table.insert(0, "observed_R", y)
    shap_table.insert(0, "block_10km", frame["block_10km"])
    shap_table.insert(0, "cell_id", frame["cell_id"])
    shap_table.to_csv(OUTPUT / "01_RF_SHAP_values_all_cells.csv", index=False, encoding="utf-8-sig")

    importance = pd.DataFrame(
        {
            "factor": FEATURES,
            "chinese_name": [CHINESE[feature] for feature in FEATURES],
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
            "shap_sd": np.std(shap_values, axis=0, ddof=1),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    importance["importance_share_percent"] = importance["mean_abs_shap"] / importance["mean_abs_shap"].sum() * 100
    importance["rank"] = np.arange(1, len(importance) + 1)
    importance.to_csv(OUTPUT / "02_RF_mean_absolute_SHAP_importance.csv", index=False, encoding="utf-8-sig")
    plot_importance(importance)
    plot_beeswarm(frame, shap_values)
    print(f"main SHAP complete; max additivity error={additivity_error:.3e}", flush=True)

    thresholds = threshold_table(frame, shap_values)
    thresholds.to_csv(OUTPUT / "03_RF_SHAP_thresholds_spatial_block_bootstrap.csv", index=False, encoding="utf-8-sig")
    plot_dependence_all(frame, shap_values, thresholds, importance)

    sample_indices = interaction_sample(frame)
    interaction_frame = frame.iloc[sample_indices].copy()
    interaction_values = normalize_interaction_array(
        explainer.shap_interaction_values(interaction_frame[FEATURES])
    )
    interaction_matrix = np.mean(np.abs(interaction_values), axis=0)
    interaction_matrix_frame = pd.DataFrame(interaction_matrix, index=FEATURES, columns=FEATURES)
    interaction_matrix_frame.to_csv(OUTPUT / "04_RF_mean_absolute_SHAP_interaction_matrix.csv", encoding="utf-8-sig")
    pair_rows = []
    for first in range(len(FEATURES)):
        for second in range(first + 1, len(FEATURES)):
            pair_rows.append(
                {
                    "factor_1": FEATURES[first],
                    "factor_1_chinese": CHINESE[FEATURES[first]],
                    "factor_2": FEATURES[second],
                    "factor_2_chinese": CHINESE[FEATURES[second]],
                    "pair_chinese": f"{CHINESE[FEATURES[first]]} × {CHINESE[FEATURES[second]]}",
                    "mean_abs_shap_interaction": interaction_matrix[first, second],
                }
            )
    pairs = pd.DataFrame(pair_rows).sort_values("mean_abs_shap_interaction", ascending=False)
    pairs["rank"] = np.arange(1, len(pairs) + 1)
    pairs.to_csv(OUTPUT / "05_RF_SHAP_interaction_pairs_ranked.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(
        {
            "cell_id": interaction_frame["cell_id"],
            "block_10km": interaction_frame["block_10km"],
        }
    ).to_csv(OUTPUT / "06_interaction_sample_cells.csv", index=False, encoding="utf-8-sig")
    np.savez_compressed(
        OUTPUT / "07_RF_SHAP_interaction_values_sample1500.npz",
        values=interaction_values,
        feature_names=np.asarray(FEATURES),
        cell_id=interaction_frame["cell_id"].to_numpy(),
    )
    plot_interactions(interaction_matrix, pairs)
    plot_dependence_top6_colored(frame, shap_values, importance, interaction_matrix, thresholds)

    cv = pd.read_csv(CV_SUMMARY)
    rf_cv = cv[(cv["block_km"] == 10) & (cv["model"] == "RF")].iloc[0]
    qa = {
        "cells": len(frame),
        "features": len(FEATURES),
        "shap_expected_value": base_value,
        "maximum_additivity_error": float(additivity_error),
        "interaction_sample_cells": len(interaction_frame),
        "interaction_sample_blocks": interaction_frame["block_10km"].nunique(),
        "threshold_bootstraps": THRESHOLD_BOOTSTRAPS,
        "spatial_cv_10km_r2_mean": rf_cv["r2_mean"],
        "spatial_cv_10km_rmse_mean": rf_cv["rmse_mean"],
    }
    (OUTPUT / "08_SHAP_QA_summary.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    captions = """图X 随机森林模型中影响因素的平均绝对SHAP值。条形长度表示变量对生态系统热风险预测的平均绝对贡献。\n\n图Y 随机森林模型的SHAP beeswarm图。横坐标为SHAP值，正值表示提高模型预测的生态系统热风险，负值表示降低模型预测值；颜色表示变量取值由低到高。\n\n图Z 影响因素的SHAP dependence及阈值识别。黑线为分位数组内SHAP中位数，橙色虚线为连续分段回归识别的响应转折点，浅橙色区域为10 km空间块bootstrap 95%区间。\n\n图W 随机森林模型的SHAP交互作用。（a）Mean |SHAP interaction|矩阵；（b）排名前10的变量交互组合。"""
    (FIGURES / "figure_captions_zh.txt").write_text(captions, encoding="utf-8")
    print("\nImportance")
    print(importance[["chinese_name", "mean_abs_shap", "importance_share_percent"]].to_string(index=False))
    print("\nTop interactions")
    print(pairs.head(10)[["pair_chinese", "mean_abs_shap_interaction"]].to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
