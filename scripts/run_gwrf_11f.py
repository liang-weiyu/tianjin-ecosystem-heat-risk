"""Geographically weighted random forest for ecosystem heat risk.

The implementation keeps coordinates out of the predictor matrix. Coordinates
are used only to define an adaptive k-nearest-neighbour neighbourhood. A local
random forest is fitted at each prediction location with bi-square case
weights. Bandwidth selection and model evaluation use held-out 10 km spatial
blocks, so the response of a validation cell is never used by its local model.

The workflow is split into restartable phases because a strict cell-by-cell
GWRF is computationally expensive:

``bandwidth``
    Select one of a small, pre-specified set of adaptive bandwidths using an
    independent 10 km spatial-block assignment and a spatially dispersed
    validation subsample.
``evaluate``
    Evaluate the fixed bandwidth on repeated 10 km spatial-block assignments.
``final``
    Fit one local RF at every valid 1 km land cell and export local OOB
    diagnostics and local impurity importance. These diagnostics are
    descriptive; repeated spatial-block CV remains the predictive evidence.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import array_bounds
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


WORKSPACE = STUDY_ROOT.parent
PROJECT = STUDY_ROOT
STRICT = PROJECT / "strict_land_recalculation_20260806"
MODEL_ROOT = STRICT / "08_background_driver_model"
INPUT_TABLE = MODEL_ROOT / "02_tables" / "background_driver_model_table_13_factors.csv"
CV_ROOT = MODEL_ROOT / "06_global_ml_repeated_spatial_block_cv_11f"
SELECTION_FOLDS = CV_ROOT / "03_parameter_selection_10km_assignment.csv"
EVALUATION_FOLDS = CV_ROOT / "06_block_fold_assignments_all_repeats.csv"
GLOBAL_REPEAT_METRICS = CV_ROOT / "04_repeat_level_model_metrics.csv"
REFERENCE_RASTER = (
    STRICT
    / "02_risk_results"
    / "04_R_ecosystem_heat_risk_strict_land_HEV_product_1km.tif"
)
BOUNDARY = BOUNDARY_PATH
OUTPUT = MODEL_ROOT / "10_GWRF_11f"
FIGURES = OUTPUT / "figures"

TARGET = "R"
COORDINATES = ["x_m", "y_m"]
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
FEATURE_LABELS_ZH = {
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

BANDWIDTH_CANDIDATES = [150, 200, 300, 600, 1000]
LOCAL_WEIGHT_CANDIDATES = [0.0, 0.25, 0.50, 0.75, 1.0]
SEARCH_SEED = 20260814
EVALUATION_SEEDS = list(range(20260815, 20260825))
RF_MAX_FEATURES = 0.6
RF_MIN_SAMPLES_LEAF = 4
NODATA = -9999.0


def configure_fonts() -> dict[str, str]:
    """Use Times New Roman first and SimSun as the Chinese fallback."""

    mpl.rcParams.update(
        {
            "font.family": "Times New Roman",
            "font.size": 10,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )
    return {
        "Times New Roman": fm.findfont("Times New Roman", fallback_to_default=True),
        "SimSun": fm.findfont("SimSun", fallback_to_default=True),
    }


def load_and_validate() -> tuple[pd.DataFrame, dict[str, object]]:
    frame = pd.read_csv(INPUT_TABLE)
    required = ["cell_id", "row", "column", TARGET, *COORDINATES, *FEATURES]
    missing_columns = sorted(set(required) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    numeric = frame[[TARGET, *COORDINATES, *FEATURES]].to_numpy(dtype=float)
    finite = np.isfinite(numeric)
    duplicate_cells = int(frame["cell_id"].duplicated().sum())
    duplicate_coordinates = int(frame.duplicated(COORDINATES).sum())
    if not finite.all() or duplicate_cells or duplicate_coordinates:
        raise ValueError(
            "GWRF input failed QA: non-finite values or duplicated cell/coordinate keys."
        )

    qa = {
        "input_table": str(INPUT_TABLE),
        "grain": "one row per valid 1 km strict-land cell",
        "n_rows": int(len(frame)),
        "n_predictors": len(FEATURES),
        "target": TARGET,
        "coordinate_columns_not_predictors": COORDINATES,
        "duplicate_cell_ids": duplicate_cells,
        "duplicate_coordinates": duplicate_coordinates,
        "missing_or_nonfinite_values": int((~finite).sum()),
        "target_min": float(frame[TARGET].min()),
        "target_max": float(frame[TARGET].max()),
        "x_range_m": [float(frame["x_m"].min()), float(frame["x_m"].max())],
        "y_range_m": [float(frame["y_m"].min()), float(frame["y_m"].max())],
        "feature_names": FEATURES,
    }
    return frame, qa


def model_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rmse = math.sqrt(mean_squared_error(observed, predicted))
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": float(rmse),
        "mae": float(mean_absolute_error(observed, predicted)),
        "nrmse_percent": float(rmse / observed.mean() * 100.0),
    }


def bisquare_weights(distances: np.ndarray) -> tuple[np.ndarray, float, float]:
    bandwidth_radius = float(distances[-1] * 1.0000001)
    weights = np.square(1.0 - np.square(distances / bandwidth_radius))
    effective_n = float(np.square(weights.sum()) / np.square(weights).sum())
    return weights, bandwidth_radius, effective_n


def make_rf(n_estimators: int, random_state: int, oob_score: bool = False):
    return RandomForestRegressor(
        n_estimators=n_estimators,
        max_features=RF_MAX_FEATURES,
        min_samples_leaf=RF_MIN_SAMPLES_LEAF,
        max_depth=None,
        bootstrap=True,
        oob_score=oob_score,
        n_jobs=1,
        random_state=random_state,
    )


def fit_predict_one(
    local_x: np.ndarray,
    local_y: np.ndarray,
    distances: np.ndarray,
    target_x: np.ndarray,
    n_estimators: int,
    random_state: int,
) -> tuple[float, float, float]:
    weights, radius, effective_n = bisquare_weights(distances)
    model = make_rf(n_estimators, random_state)
    model.fit(local_x, local_y, sample_weight=weights)
    prediction = float(model.predict(target_x.reshape(1, -1))[0])
    return prediction, radius, effective_n


def predict_spatial_fold(
    frame: pd.DataFrame,
    train_mask: np.ndarray,
    validation_indices: np.ndarray,
    bandwidth: int,
    n_estimators: int,
    base_seed: int,
    n_jobs: int,
) -> pd.DataFrame:
    train = frame.loc[train_mask]
    validation = frame.iloc[validation_indices]
    train_x = train[FEATURES].to_numpy(dtype=float)
    train_y = train[TARGET].to_numpy(dtype=float)
    train_xy = train[COORDINATES].to_numpy(dtype=float)
    validation_x = validation[FEATURES].to_numpy(dtype=float)
    validation_xy = validation[COORDINATES].to_numpy(dtype=float)

    if len(train) < bandwidth:
        raise ValueError(f"Training fold has {len(train)} cells, below bandwidth {bandwidth}.")
    distances, neighbour_indices = cKDTree(train_xy).query(
        validation_xy, k=bandwidth, workers=-1
    )

    def task(local_position: int) -> tuple[float, float, float]:
        neighbours = neighbour_indices[local_position]
        cell_seed = int(
            (base_seed + int(validation.iloc[local_position]["cell_id"]) * 104729)
            % (2**31 - 1)
        )
        return fit_predict_one(
            train_x[neighbours],
            train_y[neighbours],
            distances[local_position],
            validation_x[local_position],
            n_estimators,
            cell_seed,
        )

    results = Parallel(n_jobs=n_jobs, prefer="threads", batch_size=4)(
        delayed(task)(position) for position in range(len(validation))
    )
    result_array = np.asarray(results, dtype=float)
    return pd.DataFrame(
        {
            "cell_id": validation["cell_id"].to_numpy(),
            "observed_R": validation[TARGET].to_numpy(dtype=float),
            "predicted_R": result_array[:, 0],
            "local_radius_km": result_array[:, 1] / 1000.0,
            "effective_local_n": result_array[:, 2],
        }
    )


def make_bandwidth_target_sample(
    frame: pd.DataFrame,
    assignments: pd.DataFrame,
    sample_fraction: float,
) -> pd.DataFrame:
    merged = frame[["cell_id", "x_m", "y_m"]].merge(
        assignments, on="cell_id", validate="one_to_one"
    )
    rng = np.random.default_rng(SEARCH_SEED)
    selected_parts = []
    for fold in sorted(merged["fold"].unique()):
        fold_frame = merged.loc[merged["fold"] == fold]
        for _, block in fold_frame.groupby("block_id", sort=True):
            n_select = max(1, int(round(len(block) * sample_fraction)))
            n_select = min(n_select, len(block))
            choices = rng.choice(block.index.to_numpy(), size=n_select, replace=False)
            selected_parts.append(merged.loc[choices])
    selected = (
        pd.concat(selected_parts, ignore_index=True)
        .drop_duplicates("cell_id")
        .sort_values(["fold", "block_id", "cell_id"])
        .reset_index(drop=True)
    )
    selected["sample_fraction_requested"] = sample_fraction
    selected["sample_seed"] = SEARCH_SEED
    return selected


def run_bandwidth_selection(
    frame: pd.DataFrame,
    bandwidths: list[int],
    sample_fraction: float,
    n_estimators: int,
    n_jobs: int,
) -> int:
    assignments = pd.read_csv(SELECTION_FOLDS)
    if set(assignments["cell_id"]) != set(frame["cell_id"]):
        raise ValueError("Bandwidth-selection fold assignment does not match model cells.")
    assignment_by_cell = assignments.set_index("cell_id")
    fold_array = frame["cell_id"].map(assignment_by_cell["fold"]).to_numpy(dtype=int)

    sample_path = OUTPUT / "01_bandwidth_selection_target_sample.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path)
    else:
        sample = make_bandwidth_target_sample(frame, assignments, sample_fraction)
        sample.to_csv(sample_path, index=False, encoding="utf-8-sig")
    index_by_cell = pd.Series(frame.index.to_numpy(), index=frame["cell_id"])

    all_predictions = []
    metric_rows = []
    for bandwidth in bandwidths:
        candidate_path = OUTPUT / f"checkpoints_bandwidth_{bandwidth}.csv"
        if candidate_path.exists():
            candidate = pd.read_csv(candidate_path)
        else:
            fold_results = []
            start = time.perf_counter()
            for fold in range(1, 6):
                target_cells = sample.loc[sample["fold"] == fold, "cell_id"]
                validation_indices = index_by_cell.loc[target_cells].to_numpy(dtype=int)
                fold_result = predict_spatial_fold(
                    frame=frame,
                    train_mask=fold_array != fold,
                    validation_indices=validation_indices,
                    bandwidth=bandwidth,
                    n_estimators=n_estimators,
                    base_seed=SEARCH_SEED + bandwidth * 10 + fold,
                    n_jobs=n_jobs,
                )
                fold_result["fold"] = fold
                fold_results.append(fold_result)
                print(
                    f"bandwidth={bandwidth}, fold={fold}, "
                    f"targets={len(fold_result)} completed",
                    flush=True,
                )
            candidate = pd.concat(fold_results, ignore_index=True)
            candidate["bandwidth_neighbors"] = bandwidth
            candidate["n_estimators"] = n_estimators
            candidate["elapsed_minutes"] = (time.perf_counter() - start) / 60.0
            candidate.to_csv(candidate_path, index=False, encoding="utf-8-sig")

        metrics = model_metrics(
            candidate["observed_R"].to_numpy(), candidate["predicted_R"].to_numpy()
        )
        metrics.update(
            {
                "bandwidth_neighbors": bandwidth,
                "n_validation_cells": int(len(candidate)),
                "n_estimators": n_estimators,
                "median_local_radius_km": float(candidate["local_radius_km"].median()),
                "p25_local_radius_km": float(candidate["local_radius_km"].quantile(0.25)),
                "p75_local_radius_km": float(candidate["local_radius_km"].quantile(0.75)),
                "median_effective_local_n": float(candidate["effective_local_n"].median()),
            }
        )
        metric_rows.append(metrics)
        all_predictions.append(candidate)

    predictions = pd.concat(all_predictions, ignore_index=True)
    metrics = pd.DataFrame(metric_rows).sort_values("bandwidth_neighbors")
    global_predictions = global_selection_predictions(
        frame, fold_array, sample, index_by_cell
    )
    combination_rows = []
    for bandwidth in bandwidths:
        local = predictions.loc[
            predictions["bandwidth_neighbors"] == bandwidth,
            ["cell_id", "observed_R", "predicted_R"],
        ].rename(columns={"predicted_R": "local_predicted_R"})
        combined = local.merge(global_predictions, on=["cell_id", "observed_R"], validate="one_to_one")
        for local_weight in LOCAL_WEIGHT_CANDIDATES:
            prediction = (
                local_weight * combined["local_predicted_R"].to_numpy()
                + (1.0 - local_weight) * combined["global_predicted_R"].to_numpy()
            )
            values = model_metrics(combined["observed_R"].to_numpy(), prediction)
            values.update(
                {
                    "bandwidth_neighbors": bandwidth,
                    "local_weight": local_weight,
                    "global_weight": 1.0 - local_weight,
                    "n_validation_cells": len(combined),
                }
            )
            combination_rows.append(values)
    combination_metrics = pd.DataFrame(combination_rows).sort_values(
        ["bandwidth_neighbors", "local_weight"]
    )
    best_combination = combination_metrics.sort_values(["rmse", "mae"]).iloc[0]
    selected_bandwidth = int(best_combination["bandwidth_neighbors"])
    selected_local_weight = float(best_combination["local_weight"])
    predictions.to_csv(
        OUTPUT / "02_bandwidth_selection_predictions.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        OUTPUT / "03_bandwidth_selection_metrics.csv", index=False, encoding="utf-8-sig"
    )
    global_predictions.to_csv(
        OUTPUT / "03a_global_RF_selection_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    combination_metrics.to_csv(
        OUTPUT / "03b_bandwidth_local_weight_selection_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )
    selection = {
        "selected_bandwidth_neighbors": selected_bandwidth,
        "selected_local_weight": selected_local_weight,
        "selected_global_weight": 1.0 - selected_local_weight,
        "selection_rule": (
            "joint minimum RMSE across pre-specified bandwidth and local-weight "
            "candidates; MAE used as deterministic tie-breaker"
        ),
        "kernel": "adaptive k-nearest neighbours with bi-square case weights",
        "validation": "independent no-buffer 10 km spatial-block 5-fold CV",
        "validation_sample": (
            "spatially dispersed block-stratified sample used only for bandwidth selection"
        ),
        "sample_fraction_requested": sample_fraction,
        "n_validation_cells": int(len(sample)),
        "n_estimators_search": n_estimators,
        "global_RF_n_estimators_for_weight_selection": 300,
        "local_weight_candidates": LOCAL_WEIGHT_CANDIDATES,
        "rf_parameters": {
            "max_features": RF_MAX_FEATURES,
            "min_samples_leaf": RF_MIN_SAMPLES_LEAF,
            "bootstrap": True,
        },
    }
    (OUTPUT / "04_selected_bandwidth.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pure_local_best = int(metrics.sort_values(["rmse", "mae"]).iloc[0]["bandwidth_neighbors"])
    plot_bandwidth_selection(metrics, pure_local_best)
    return selected_bandwidth


def global_selection_predictions(
    frame: pd.DataFrame,
    fold_array: np.ndarray,
    sample: pd.DataFrame,
    index_by_cell: pd.Series,
) -> pd.DataFrame:
    """Global-RF predictions on the independent selection cells."""

    output_path = OUTPUT / "checkpoints_global_RF_selection_predictions.csv"
    if output_path.exists():
        return pd.read_csv(output_path)
    parts = []
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    for fold in range(1, 6):
        target_cells = sample.loc[sample["fold"] == fold, "cell_id"]
        validation_indices = index_by_cell.loc[target_cells].to_numpy(dtype=int)
        model = make_rf(300, SEARCH_SEED + 9000 + fold)
        model.fit(x[fold_array != fold], y[fold_array != fold])
        part = pd.DataFrame(
            {
                "cell_id": frame.iloc[validation_indices]["cell_id"].to_numpy(),
                "observed_R": y[validation_indices],
                "global_predicted_R": model.predict(x[validation_indices]),
                "fold": fold,
            }
        )
        parts.append(part)
    result = pd.concat(parts, ignore_index=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


def plot_bandwidth_selection(metrics: pd.DataFrame, selected_bandwidth: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), dpi=180)
    color = "#3F7FBF"
    marker_face = "white"
    times = fm.FontProperties(fname=font_path("english"), size=10)
    simsun = fm.FontProperties(fname=font_path("chinese"), size=10)
    for axis, metric, label, panel in [
        (axes[0], "r2", "R²", "(a)"),
        (axes[1], "rmse", "RMSE", "(b)"),
    ]:
        axis.plot(
            metrics["bandwidth_neighbors"],
            metrics[metric],
            color=color,
            linewidth=1.6,
            marker="o",
            markersize=5.2,
            markerfacecolor=marker_face,
            markeredgecolor=color,
            markeredgewidth=1.2,
        )
        chosen = metrics.loc[metrics["bandwidth_neighbors"] == selected_bandwidth].iloc[0]
        axis.scatter(
            [selected_bandwidth], [chosen[metric]], s=42, color="#E58B52", zorder=4
        )
        axis.text(
            0.0,
            1.02,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontproperties=times,
        )
        axis.set_xlabel("自适应带宽（最近邻格网数）", fontproperties=simsun)
        axis.set_ylabel(label, fontproperties=times)
        for tick_label in [*axis.get_xticklabels(), *axis.get_yticklabels()]:
            tick_label.set_fontproperties(times)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.65, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(w_pad=2.0)
    for extension, dpi in [("png", 600), ("tif", 600), ("pdf", 300)]:
        fig.savefig(
            FIGURES / f"GWRF_bandwidth_selection.{extension}",
            dpi=dpi,
            bbox_inches="tight",
        )
    plt.close(fig)


def cell_fold_assignment(frame: pd.DataFrame, seed: int) -> pd.DataFrame:
    blocks = pd.read_csv(EVALUATION_FOLDS)
    blocks = blocks.loc[(blocks["block_km"] == 10) & (blocks["seed"] == seed)]
    if blocks.empty:
        raise ValueError(f"No 10 km fold assignment found for seed {seed}.")
    size_m = 10_000.0
    origin_x = math.floor(frame["x_m"].min() / size_m) * size_m
    origin_y = math.floor(frame["y_m"].min() / size_m) * size_m
    block_col = np.floor((frame["x_m"] - origin_x) / size_m).astype(int)
    block_row = np.floor((frame["y_m"] - origin_y) / size_m).astype(int)
    cell_blocks = pd.DataFrame(
        {
            "cell_id": frame["cell_id"],
            "block_id": "B10_" + block_col.astype(str) + "_" + block_row.astype(str),
        }
    )
    result = cell_blocks.merge(blocks[["block_id", "fold"]], on="block_id", how="left")
    if result["fold"].isna().any():
        raise ValueError(f"Incomplete fold assignment for seed {seed}.")
    return result


def run_evaluation(
    frame: pd.DataFrame,
    bandwidth: int,
    n_estimators: int,
    n_jobs: int,
    n_repeats: int,
) -> None:
    selection = json.loads((OUTPUT / "04_selected_bandwidth.json").read_text(encoding="utf-8"))
    local_weight = float(selection.get("selected_local_weight", 1.0))
    repeat_metrics = []
    all_predictions = []
    index_by_cell = pd.Series(frame.index.to_numpy(), index=frame["cell_id"])
    for repeat, seed in enumerate(EVALUATION_SEEDS[:n_repeats], start=1):
        checkpoint = OUTPUT / f"checkpoints_evaluation_seed_{seed}.csv"
        if checkpoint.exists():
            predictions = pd.read_csv(checkpoint)
        else:
            assignment = cell_fold_assignment(frame, seed)
            fold_by_cell = assignment.set_index("cell_id")["fold"]
            fold_array = frame["cell_id"].map(fold_by_cell).to_numpy(dtype=int)
            parts = []
            start = time.perf_counter()
            for fold in range(1, 6):
                validation_indices = np.flatnonzero(fold_array == fold)
                part = predict_spatial_fold(
                    frame,
                    fold_array != fold,
                    validation_indices,
                    bandwidth,
                    n_estimators,
                    seed + fold,
                    n_jobs,
                )
                part["fold"] = fold
                parts.append(part)
                print(f"evaluation seed={seed}, fold={fold} completed", flush=True)
            predictions = pd.concat(parts, ignore_index=True)
            predictions["repeat"] = repeat
            predictions["seed"] = seed
            predictions["bandwidth_neighbors"] = bandwidth
            predictions["n_estimators"] = n_estimators
            predictions["elapsed_minutes"] = (time.perf_counter() - start) / 60.0
            predictions.to_csv(checkpoint, index=False, encoding="utf-8-sig")
        global_oof = pd.read_csv(CV_ROOT / "08_all_out_of_fold_predictions.csv")
        global_oof = global_oof.loc[
            (global_oof["block_km"] == 10)
            & (global_oof["seed"] == seed)
            & (global_oof["model"] == "RF"),
            ["cell_id", "predicted_R"],
        ].rename(columns={"predicted_R": "global_predicted_R"})
        evaluated = predictions.rename(columns={"predicted_R": "local_predicted_R"}).merge(
            global_oof, on="cell_id", validate="one_to_one"
        )
        evaluated["combined_predicted_R"] = (
            local_weight * evaluated["local_predicted_R"]
            + (1.0 - local_weight) * evaluated["global_predicted_R"]
        )
        evaluated["local_weight"] = local_weight
        evaluated["local_residual"] = (
            evaluated["observed_R"] - evaluated["local_predicted_R"]
        )
        evaluated["global_residual"] = (
            evaluated["observed_R"] - evaluated["global_predicted_R"]
        )
        evaluated["combined_residual"] = (
            evaluated["observed_R"] - evaluated["combined_predicted_R"]
        )
        for model_name, prediction_column in [
            ("RF_global", "global_predicted_R"),
            ("GWRF_local", "local_predicted_R"),
            ("GWRF_global_local_combined", "combined_predicted_R"),
        ]:
            values = model_metrics(
                evaluated["observed_R"].to_numpy(),
                evaluated[prediction_column].to_numpy(),
            )
            values.update(
                {
                    "model": model_name,
                    "repeat": repeat,
                    "seed": seed,
                    "block_km": 10,
                    "bandwidth_neighbors": bandwidth,
                    "local_weight": (
                        local_weight
                        if "combined" in model_name
                        else (0.0 if model_name == "RF_global" else 1.0)
                    ),
                    "n_estimators": n_estimators,
                    "n_validation_cells": int(len(evaluated)),
                    "median_local_radius_km": float(evaluated["local_radius_km"].median()),
                }
            )
            repeat_metrics.append(values)
        all_predictions.append(evaluated)

    metrics = pd.DataFrame(repeat_metrics)
    metrics.to_csv(OUTPUT / "05_GWRF_repeat_metrics.csv", index=False, encoding="utf-8-sig")
    pd.concat(all_predictions, ignore_index=True).to_csv(
        OUTPUT / "06_GWRF_spatial_block_oof_predictions.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary = metrics.groupby("model")[["r2", "rmse", "mae", "nrmse_percent"]].agg(
        ["mean", "std", "min", "max"]
    )
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary.reset_index().to_csv(
        OUTPUT / "07_GWRF_performance_summary.csv", index=False, encoding="utf-8-sig"
    )


def fit_final_one(
    position: int,
    neighbour_indices: np.ndarray,
    distances: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cell_id: int,
    n_estimators: int,
) -> np.ndarray:
    neighbours = neighbour_indices[position]
    weights, radius, effective_n = bisquare_weights(distances[position])
    model = make_rf(
        n_estimators=n_estimators,
        random_state=int((SEARCH_SEED + cell_id * 104729) % (2**31 - 1)),
        oob_score=True,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Some inputs do not have OOB scores")
        model.fit(x[neighbours], y[neighbours], sample_weight=weights)
    focal_positions = np.flatnonzero(neighbours == position)
    focal_oob = np.nan
    if len(focal_positions):
        focal_oob = float(model.oob_prediction_[focal_positions[0]])
    fitted = float(model.predict(x[position : position + 1])[0])
    return np.r_[fitted, focal_oob, model.oob_score_, radius / 1000.0, effective_n,
                 model.feature_importances_]


def run_final_fit(
    frame: pd.DataFrame,
    bandwidth: int,
    n_estimators: int,
    n_jobs: int,
) -> None:
    x = frame[FEATURES].to_numpy(dtype=float)
    y = frame[TARGET].to_numpy(dtype=float)
    xy = frame[COORDINATES].to_numpy(dtype=float)
    distances, neighbour_indices = cKDTree(xy).query(xy, k=bandwidth, workers=-1)
    results = []
    chunk_size = 500
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        checkpoint = OUTPUT / f"checkpoints_final_{start:05d}_{stop - 1:05d}.npz"
        if checkpoint.exists():
            chunk_result = np.load(checkpoint)["result"]
        else:
            chunk_result = np.asarray(
                Parallel(n_jobs=n_jobs, prefer="threads", batch_size=4)(
                    delayed(fit_final_one)(
                        position,
                        neighbour_indices,
                        distances,
                        x,
                        y,
                        int(frame.iloc[position]["cell_id"]),
                        n_estimators,
                    )
                    for position in range(start, stop)
                ),
                dtype=float,
            )
            np.savez_compressed(checkpoint, result=chunk_result)
        results.append(chunk_result)
        print(f"final local forests: {stop}/{len(frame)} completed", flush=True)
    values = np.vstack(results)
    columns = [
        "local_fitted_R",
        "focal_local_oob_R",
        "local_oob_r2",
        "local_radius_km",
        "effective_local_n",
        *[f"importance_{name}" for name in FEATURES],
    ]
    output = frame[["cell_id", "row", "column", "x_m", "y_m", "R"]].copy()
    output[columns] = values
    output.to_csv(OUTPUT / "08_GWRF_final_local_diagnostics.csv", index=False, encoding="utf-8-sig")

    importance_columns = [f"importance_{name}" for name in FEATURES]
    dominant_indices = output[importance_columns].to_numpy().argmax(axis=1)
    output["dominant_factor_code"] = dominant_indices + 1
    output["dominant_factor"] = [FEATURE_LABELS_ZH[FEATURES[index]] for index in dominant_indices]
    output[["cell_id", "dominant_factor_code", "dominant_factor"]].to_csv(
        OUTPUT / "09_GWRF_local_dominant_factor.csv", index=False, encoding="utf-8-sig"
    )
    save_local_importance_summary(output, importance_columns)
    write_final_rasters(output, importance_columns)
    plot_final_spatial_outputs(output, importance_columns)


def save_local_importance_summary(
    frame: pd.DataFrame, importance_columns: list[str]
) -> None:
    rows = []
    for code, (feature, column) in enumerate(
        zip(FEATURES, importance_columns), start=1
    ):
        values = frame[column].to_numpy(dtype=float)
        count = int((frame["dominant_factor_code"] == code).sum())
        rows.append(
            {
                "factor_code": code,
                "feature": feature,
                "feature_label_zh": FEATURE_LABELS_ZH[feature],
                "mean_local_importance": float(values.mean()),
                "median_local_importance": float(np.median(values)),
                "p25_local_importance": float(np.quantile(values, 0.25)),
                "p75_local_importance": float(np.quantile(values, 0.75)),
                "dominant_cell_count": count,
                "dominant_cell_percent": float(count / len(frame) * 100.0),
            }
        )
    pd.DataFrame(rows).sort_values(
        "mean_local_importance", ascending=False
    ).to_csv(
        OUTPUT / "10_GWRF_local_importance_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


def frame_to_array(frame: pd.DataFrame, values: np.ndarray) -> np.ndarray:
    with rasterio.open(REFERENCE_RASTER) as source:
        shape = source.shape
    array = np.full(shape, np.nan, dtype=float)
    array[
        frame["row"].to_numpy(dtype=int),
        frame["column"].to_numpy(dtype=int),
    ] = values
    return array


def draw_land_outline(axis: plt.Axes, valid_mask: np.ndarray) -> None:
    # ``imshow(origin="upper")`` places row 0 at the top while contour's
    # ``origin="upper"`` generates vertically reversed contour coordinates.
    # Use native row/column coordinates here and invert the y axis explicitly
    # so the outline and raster cells share exactly the same grid geometry.
    axis.contour(
        valid_mask.astype(float),
        levels=[0.5],
        colors="#555555",
        linewidths=0.35,
    )
    axis.set_xlim(-0.5, valid_mask.shape[1] - 0.5)
    axis.set_ylim(valid_mask.shape[0] - 0.5, -0.5)
    axis.set_aspect("equal")
    axis.set_axis_off()


def load_publication_map_context():
    """Load the Tianjin outer administrative frame in the raster CRS."""

    import geopandas as gpd

    with rasterio.open(REFERENCE_RASTER) as source:
        raster_extent = array_bounds(source.height, source.width, source.transform)
        raster_crs = source.crs
    boundary = gpd.read_file(BOUNDARY).to_crs(raster_crs)
    geometry = boundary.geometry.union_all()
    if geometry.geom_type != "Polygon":
        geometry = max(geometry.geoms, key=lambda item: item.area)
    return geometry, geometry.bounds, raster_extent


def format_publication_map(
    axis: plt.Axes,
    image,
    geometry,
    map_bounds: tuple[float, float, float, float],
    *,
    compact_scale: bool = False,
) -> None:
    """Clip to Tianjin and add only the outer frame, north arrow and scale."""

    # Shared map helpers are stored alongside these research scripts.
    from tianjin_map_style import (
        add_boundary_outline,
        add_north_arrow,
        add_scale_bar,
    )

    west, south, east, north = map_bounds
    width = east - west
    height = north - south
    outline = add_boundary_outline(axis, geometry, linewidth=0.70)
    image.set_clip_path(outline)
    axis.set_xlim(west - 0.01 * width, east + 0.42 * width)
    axis.set_ylim(south - 0.10 * height, north + 0.01 * height)
    axis.set_aspect("equal")
    axis.set_axis_off()
    add_north_arrow(axis, map_bounds)
    add_scale_bar(axis, map_bounds)
    if compact_scale:
        scale_labels = {"0", "5", "10", "20", "30", "40", "km"}
        for text_object in axis.texts:
            if text_object.get_text() in scale_labels:
                text_object.set_fontsize(
                    4.6 if text_object.get_text() != "km" else 5.0
                )


def save_figure_all_formats(figure: plt.Figure, stem: Path) -> None:
    for extension, dpi in [("png", 600), ("tif", 600), ("pdf", 300)]:
        figure.savefig(
            stem.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
        )


def plot_final_spatial_outputs(
    frame: pd.DataFrame, importance_columns: list[str]
) -> None:
    times = fm.FontProperties(fname=font_path("english"), size=9)
    times_small = fm.FontProperties(fname=font_path("english"), size=7.5)
    simsun = fm.FontProperties(fname=font_path("chinese"), size=9)
    simsun_legend = fm.FontProperties(fname=font_path("chinese"), size=7.5)
    simsun_legend_title = fm.FontProperties(fname=font_path("chinese"), size=8.25)
    simsun_legend_small = fm.FontProperties(fname=font_path("chinese"), size=6.3)
    geometry, map_bounds, raster_bounds = load_publication_map_context()
    raster_west, raster_south, raster_east, raster_north = raster_bounds
    raster_extent = [raster_west, raster_east, raster_south, raster_north]
    category_colors = [
        "#67AFCB",  # 地表高程：全局重要性第6，浅蓝
        "#A8C977",  # 夏季平均风速：第4，绿
        "#E4775E",  # 地表反照率：第1，红
        "#8A7EBB",  # 建筑覆盖率：第8，紫
        "#9B624B",  # 平均建筑高度：第11，棕
        "#9A9488",  # 建筑高度离散度：第10，灰
        "#BA7EAC",  # 天空可视因子：第9，浅紫
        "#6D91C4",  # 路网密度：第7，蓝
        "#E7C568",  # 夜间灯光强度：第3，黄
        "#69B8A8",  # 人口密度：第5，青
        "#E8A25C",  # 夏季累计降水：第2，橙
    ]
    category_cmap = ListedColormap(category_colors)
    global_importance_order = [2, 10, 8, 1, 9, 0, 7, 3, 6, 5, 4]
    shared_vmax = 0.40
    dominant_array = frame_to_array(
        frame, frame["dominant_factor_code"].to_numpy(dtype=float)
    )

    figure, axes = plt.subplots(3, 4, figsize=(10.2, 10.0), dpi=180)
    image = None
    for index, feature_index in enumerate(global_importance_order):
        feature = FEATURES[feature_index]
        column = importance_columns[feature_index]
        axis = axes.flat[index]
        array = frame_to_array(frame, frame[column].to_numpy(dtype=float))
        feature_cmap = LinearSegmentedColormap.from_list(
            f"gwrf_{feature}",
            ["#F7F7F4", category_colors[feature_index]],
            N=256,
        )
        image = axis.imshow(
            array,
            extent=raster_extent,
            cmap=feature_cmap,
            vmin=0.0,
            vmax=shared_vmax,
            origin="upper",
            interpolation="nearest",
        )
        format_publication_map(
            axis, image, geometry, map_bounds, compact_scale=True
        )
        axis.text(
            0.01,
            1.012,
            f"({chr(97 + index)})",
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontproperties=times,
            clip_on=False,
        )
        axis.text(
            0.075,
            1.012,
            FEATURE_LABELS_ZH[feature],
            transform=axis.transAxes,
            ha="left",
            va="bottom",
            fontproperties=simsun,
            clip_on=False,
        )
        local_color_axis = inset_axes(
            axis,
            width="3.4%",
            height="35%",
            loc="lower left",
            bbox_to_anchor=(0.79, 0.30, 1.0, 1.0),
            bbox_transform=axis.transAxes,
            borderpad=0,
        )
        local_colorbar = figure.colorbar(
            image, cax=local_color_axis, orientation="vertical"
        )
        local_colorbar.set_ticks([0.0, 0.2, 0.4])
        local_colorbar.set_ticklabels(["0", "0.20", "0.40"])
        local_colorbar.ax.set_title(
            "重要性", fontproperties=simsun_legend_title, pad=8
        )
        for label in local_colorbar.ax.get_yticklabels():
            label.set_fontproperties(times_small)

    dominant_axis = axes.flat[-1]
    dominant_image = dominant_axis.imshow(
        dominant_array,
        extent=raster_extent,
        cmap=category_cmap,
        vmin=0.5,
        vmax=len(FEATURES) + 0.5,
        origin="upper",
        interpolation="nearest",
    )
    format_publication_map(
        dominant_axis, dominant_image, geometry, map_bounds, compact_scale=True
    )
    dominant_axis.text(
        0.01, 1.012, "(l)", transform=dominant_axis.transAxes,
        ha="left", va="bottom", fontproperties=times, clip_on=False,
    )
    dominant_axis.text(
        0.075, 1.012, "局地主导因素", transform=dominant_axis.transAxes,
        ha="left", va="bottom", fontproperties=simsun, clip_on=False,
    )
    preferred_legend_order = [3, 11, 9, 2, 10, 1, 8, 4, 7, 6, 5]
    present = set(frame["dominant_factor_code"].unique())
    present_codes = [code for code in preferred_legend_order if code in present]
    dominant_handles = [
        Patch(
            facecolor=category_colors[code - 1],
            edgecolor="#555555",
            linewidth=0.3,
            label=FEATURE_LABELS_ZH[FEATURES[code - 1]],
        )
        for code in present_codes
    ]
    dominant_legend = dominant_axis.legend(
        handles=dominant_handles,
        loc="center left",
        bbox_to_anchor=(0.72, 0.46),
        frameon=False,
        prop=simsun_legend_small,
        handlelength=1.3,
        handleheight=0.75,
        labelspacing=0.18,
        handletextpad=0.28,
    )
    dominant_legend._legend_box.align = "left"
    figure.subplots_adjust(
        left=0.03, right=0.97, top=0.985, bottom=0.025,
        wspace=0.04, hspace=0.13
    )
    save_figure_all_formats(
        figure, FIGURES / "GWRF_local_importance_11_factors_4x3"
    )
    plt.close(figure)

    local_r2 = frame["local_oob_r2"].to_numpy(dtype=float)
    r2_array = frame_to_array(frame, local_r2)
    lower = float(np.nanmin(local_r2))
    upper = float(np.nanmax(local_r2))
    lower = min(lower, -1e-6)
    upper = max(upper, 1e-6)
    r2_cmap = LinearSegmentedColormap.from_list(
        "publication_local_shap_diverging",
        [
            "#4C82B8", "#A9C8DF", "#E7E2DA", "#E9AAA0", "#C85B4B",
        ],
        N=256,
    )
    figure, axes = plt.subplots(1, 2, figsize=(8.0, 5.2), dpi=180)
    r2_image = axes[0].imshow(
        r2_array,
        extent=raster_extent,
        cmap=r2_cmap,
        norm=TwoSlopeNorm(vmin=-0.20, vcenter=0.0, vmax=0.80),
        origin="upper",
        interpolation="nearest",
    )
    format_publication_map(axes[0], r2_image, geometry, map_bounds)
    axes[0].text(0.01, 0.99, "(a)", transform=axes[0].transAxes, ha="left", va="top", fontproperties=times)
    color_axis = inset_axes(
        axes[0], width="3.7%", height="42%", loc="lower left",
        bbox_to_anchor=(0.778, 0.215, 1.0, 1.0),
        bbox_transform=axes[0].transAxes, borderpad=0,
    )
    colorbar = figure.colorbar(r2_image, cax=color_axis, orientation="vertical")
    colorbar.set_ticks([-0.20, 0.0, 0.20, 0.40, 0.60, 0.80])
    colorbar.set_ticklabels(["-0.20", "0", "0.20", "0.40", "0.60", "0.80"])
    colorbar.ax.set_title(
        "局部袋外\n决定系数", fontproperties=simsun_legend_title, pad=8
    )
    for label in colorbar.ax.get_yticklabels():
        label.set_fontproperties(times_small)

    axes[1].imshow(
        dominant_array,
        extent=raster_extent,
        cmap=category_cmap,
        vmin=0.5,
        vmax=len(FEATURES) + 0.5,
        origin="upper",
        interpolation="nearest",
    )
    dominant_image = axes[1].images[-1]
    format_publication_map(axes[1], dominant_image, geometry, map_bounds)
    axes[1].text(0.01, 0.99, "(b)", transform=axes[1].transAxes, ha="left", va="top", fontproperties=times)
    preferred_legend_order = [3, 11, 9, 2, 10, 1, 8, 4, 7, 6, 5]
    present = set(frame["dominant_factor_code"].unique())
    present_codes = [code for code in preferred_legend_order if code in present]
    handles = [
        Patch(
            facecolor=category_colors[code - 1],
            edgecolor="#555555",
            linewidth=0.35,
            label=FEATURE_LABELS_ZH[FEATURES[code - 1]],
        )
        for code in present_codes
    ]
    legend = axes[1].legend(
        handles=handles,
        title="局地主导因素",
        loc="center left",
        bbox_to_anchor=(0.73, 0.43),
        ncol=1,
        frameon=False,
        prop=simsun_legend,
        handlelength=1.45,
        handleheight=0.85,
        labelspacing=0.28,
        columnspacing=0.6,
        handletextpad=0.32,
    )
    legend.get_title().set_fontproperties(simsun_legend_title)
    legend._legend_box.align = "left"
    for text_object in legend.get_texts():
        text_object.set_fontproperties(simsun_legend)
    figure.subplots_adjust(left=0.02, right=0.99, top=0.985, bottom=0.025, wspace=0.04)
    save_figure_all_formats(
        figure, FIGURES / "GWRF_local_OOB_R2_and_dominant_factor"
    )
    plt.close(figure)

    (FIGURES / "GWRF_final_figure_captions_zh.txt").write_text(
        "图X GWRF局部拟合与驱动因素的空间分布。（a）局部随机森林的OOB R²；"
        "（b）各1 km格网局部变量重要性最大的驱动因素。局部OOB R²为描述性诊断，"
        "模型外推能力以重复空间分块交叉验证为准。\n\n"
        "图X GWRF局部变量重要性及主导因素的空间分布。（a）—（k）分别表示11项驱动"
        "因素的局部变量重要性，各面板采用与（l）局地主导因素分类一致的固定色相，并以"
        "浅色至深色表示重要性由低到高。连续色标统一为0—0.40，超过上限的值以最高等级"
        "颜色显示，原始CSV和GeoTIFF数值未截断；（l）表示各1 km格网局部变量重要性最大"
        "的驱动因素。",
        encoding="utf-8",
    )


def write_raster(frame: pd.DataFrame, values: np.ndarray, path: Path, dtype: str) -> None:
    with rasterio.open(REFERENCE_RASTER) as source:
        profile = source.profile.copy()
        shape = source.shape
    if dtype == "float32":
        raster = np.full(shape, NODATA, dtype="float32")
        nodata = NODATA
    else:
        raster = np.zeros(shape, dtype="uint8")
        nodata = 0
    raster[frame["row"].to_numpy(dtype=int), frame["column"].to_numpy(dtype=int)] = values
    profile.update(dtype=dtype, count=1, nodata=nodata, compress="deflate")
    with rasterio.open(path, "w", **profile) as destination:
        destination.write(raster, 1)


def write_final_rasters(frame: pd.DataFrame, importance_columns: list[str]) -> None:
    write_raster(
        frame,
        frame["local_oob_r2"].to_numpy(dtype="float32"),
        OUTPUT / "GWRF_local_OOB_R2_1km.tif",
        "float32",
    )
    write_raster(
        frame,
        frame["local_radius_km"].to_numpy(dtype="float32"),
        OUTPUT / "GWRF_adaptive_radius_km_1km.tif",
        "float32",
    )
    write_raster(
        frame,
        frame["dominant_factor_code"].to_numpy(dtype="uint8"),
        OUTPUT / "GWRF_local_dominant_factor_code_1km.tif",
        "uint8",
    )
    for feature, column in zip(FEATURES, importance_columns):
        write_raster(
            frame,
            frame[column].to_numpy(dtype="float32"),
            OUTPUT / f"GWRF_local_importance_{feature}_1km.tif",
            "float32",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=["bandwidth", "evaluate", "final", "all"], default="bandwidth"
    )
    parser.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--bandwidth-sample-fraction", type=float, default=0.15)
    parser.add_argument("--search-trees", type=int, default=60)
    parser.add_argument("--evaluation-trees", type=int, default=100)
    parser.add_argument("--final-trees", type=int, default=100)
    parser.add_argument("--evaluation-repeats", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    fonts = configure_fonts()
    frame, qa = load_and_validate()
    qa["resolved_fonts"] = fonts
    qa["method"] = {
        "model": "strict target-centred geographically weighted random forest",
        "kernel": "adaptive k-nearest neighbours; bi-square case weights",
        "coordinates_as_predictors": False,
        "spatial_buffer": False,
    }
    (OUTPUT / "00_GWRF_input_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    selected_path = OUTPUT / "04_selected_bandwidth.json"
    if args.phase in {"bandwidth", "all"}:
        selected_bandwidth = run_bandwidth_selection(
            frame,
            BANDWIDTH_CANDIDATES,
            args.bandwidth_sample_fraction,
            args.search_trees,
            args.jobs,
        )
    elif selected_path.exists():
        selected_bandwidth = int(
            json.loads(selected_path.read_text(encoding="utf-8"))[
                "selected_bandwidth_neighbors"
            ]
        )
    else:
        raise FileNotFoundError("Run --phase bandwidth before evaluate/final.")

    if args.phase in {"evaluate", "all"}:
        run_evaluation(
            frame,
            selected_bandwidth,
            args.evaluation_trees,
            args.jobs,
            args.evaluation_repeats,
        )
    if args.phase in {"final", "all"}:
        run_final_fit(
            frame,
            selected_bandwidth,
            args.final_trees,
            args.jobs,
        )
    print(f"GWRF phase '{args.phase}' completed. Output: {OUTPUT}")


if __name__ == "__main__":
    main()
