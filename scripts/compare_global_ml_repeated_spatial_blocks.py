"""Repeated spatial-block CV for global ecosystem heat-risk models.

Protocol
--------
1. Aggregate 1 km grid centres into regular 5, 10, or 20 km blocks.
2. Assign whole blocks to five spatially dispersed and composition-balanced folds.
3. Select a small fixed hyperparameter set once on an independent 10 km split.
4. Evaluate fixed models on ten new fold assignments for each block size.
5. Coordinates are used only for block/fold construction, never as predictors.
6. No spatial buffer is applied.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path
import warnings

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from scipy.stats import t
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


WORKSPACE = STUDY_ROOT.parent
PROJECT = STUDY_ROOT
STRICT = PROJECT / "strict_land_recalculation_20260806"
INPUT_TABLE = (
    STRICT
    / "08_background_driver_model"
    / "02_tables"
    / "background_driver_model_table_13_factors.csv"
)
REFERENCE_RASTER = (
    STRICT
    / "02_risk_results"
    / "04_R_ecosystem_heat_risk_strict_land_HEV_product_1km.tif"
)
CLCD_2021 = CLCD_PATH
OUTPUT = (
    STRICT
    / "08_background_driver_model"
    / "06_global_ml_repeated_spatial_block_cv_11f"
)
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
MODEL_ORDER = ["MLR", "SVR", "RF", "XGBoost"]
MODEL_COLORS = {
    "MLR": "#4C78A8",
    "SVR": "#72B7B2",
    "RF": "#F2A65A",
    "XGBoost": "#E76F51",
}
FOLD_COLORS = ["#4C78A8", "#F2A65A", "#72B7B2", "#E76F51", "#8F75B5"]
CLCD_NAMES = {
    1: "耕地",
    2: "林地",
    3: "灌木地",
    4: "草地",
    5: "水体",
    6: "冰雪",
    7: "裸地",
    8: "不透水面",
    9: "湿地",
}

N_FOLDS = 5
BLOCK_SIZES_KM = [5, 10, 20]
MAIN_BLOCK_KM = 10
SELECTION_SEED = 20260814
EVALUATION_SEEDS = list(range(20260815, 20260825))
BALANCE_TRIALS = 10
SWAP_ITERATIONS = 2_000


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


def evaluate(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(observed, predicted)),
        "nrmse_percent": float(rmse / np.mean(observed) * 100.0),
    }


def add_clcd_dominant(frame: pd.DataFrame) -> pd.DataFrame:
    """Resample CLCD 2021 to the 1 km reference grid by modal class."""

    with rasterio.open(REFERENCE_RASTER) as reference:
        destination = np.zeros(reference.shape, dtype="uint8")
        reference_transform = reference.transform
        reference_crs = reference.crs
    with rasterio.open(CLCD_2021) as source:
        reproject(
            source=rasterio.band(source, 1),
            destination=destination,
            src_transform=source.transform,
            src_crs=source.crs,
            src_nodata=source.nodata,
            dst_transform=reference_transform,
            dst_crs=reference_crs,
            dst_nodata=0,
            resampling=Resampling.mode,
        )
    result = frame.copy()
    result["clcd_2021_dominant"] = destination[
        result["row"].to_numpy(dtype=int), result["column"].to_numpy(dtype=int)
    ]
    result["ecosystem_type"] = result["clcd_2021_dominant"].map(CLCD_NAMES)
    result["ecosystem_type"] = result["ecosystem_type"].fillna("其他")
    return result


def add_balance_strata(frame: pd.DataFrame) -> pd.DataFrame:
    """Create balancing strata without using the response variable."""

    result = frame.copy()
    result["coast_band"] = pd.cut(
        result["coast_distance_km"],
        bins=[-np.inf, 20, 40, 60, 80, np.inf],
        labels=["0–20 km", "20–40 km", "40–60 km", "60–80 km", ">80 km"],
    ).astype(str)
    result["landscape_zone"] = "其他陆地区域"
    result.loc[result["clcd_2021_dominant"] == 1, "landscape_zone"] = "农业区"
    result.loc[result["clcd_2021_dominant"] == 8, "landscape_zone"] = "城区"
    result.loc[result["elevation_m"] >= 50.0, "landscape_zone"] = "北部山区"

    # Coarse coordinate strata improve spatial dispersion. They are used only
    # for fold construction and never enter a fitted model.
    x_rank = pd.qcut(result["x_m"], 3, labels=["西", "中", "东"], duplicates="drop")
    y_rank = pd.qcut(result["y_m"], 3, labels=["南", "中", "北"], duplicates="drop")
    result["macro_zone"] = x_rank.astype(str) + "-" + y_rank.astype(str)
    return result


def add_blocks(frame: pd.DataFrame, block_km: int) -> pd.DataFrame:
    result = frame.copy()
    size_m = block_km * 1_000.0
    # Fixed projection origin makes block definitions reproducible across runs.
    origin_x = math.floor(result["x_m"].min() / size_m) * size_m
    origin_y = math.floor(result["y_m"].min() / size_m) * size_m
    result["block_col"] = np.floor((result["x_m"] - origin_x) / size_m).astype(int)
    result["block_row"] = np.floor((result["y_m"] - origin_y) / size_m).astype(int)
    result["block_id"] = (
        f"B{block_km:02d}_"
        + result["block_col"].astype(str)
        + "_"
        + result["block_row"].astype(str)
    )
    return result


def make_block_matrix(frame: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, list[str], np.ndarray]:
    blocks = (
        frame.groupby("block_id")
        .agg(
            n_cells=("cell_id", "size"),
            x_center=("x_m", "mean"),
            y_center=("y_m", "mean"),
        )
        .reset_index()
        .sort_values("block_id")
        .reset_index(drop=True)
    )
    contributions = [blocks["n_cells"].to_numpy(dtype=float)]
    names = ["n_cells"]
    weights = [4.0]
    for column, prefix, weight in [
        ("coast_band", "coast", 1.5),
        ("landscape_zone", "landscape", 1.5),
        ("ecosystem_type", "ecosystem", 0.8),
        ("macro_zone", "macro", 0.6),
    ]:
        cross = pd.crosstab(frame["block_id"], frame[column]).reindex(
            blocks["block_id"], fill_value=0
        )
        for category in cross.columns:
            contributions.append(cross[category].to_numpy(dtype=float))
            names.append(f"{prefix}:{category}")
            weights.append(weight)
    return blocks, np.column_stack(contributions), names, np.asarray(weights)


def balance_objective(
    assignment: np.ndarray,
    matrix: np.ndarray,
    weights: np.ndarray,
) -> float:
    fold_values = np.vstack(
        [matrix[assignment == fold].sum(axis=0) for fold in range(N_FOLDS)]
    )
    target = matrix.sum(axis=0) / N_FOLDS
    active = target > 0
    relative = (fold_values[:, active] - target[active]) / target[active]
    weighted = relative**2 * weights[active][None, :]
    total_penalty = float(np.max(np.abs(relative[:, 0]))) * 2.0
    empty_penalty = float(np.any(np.bincount(assignment, minlength=N_FOLDS) == 0)) * 1e6
    return float(np.mean(weighted) + total_penalty + empty_penalty)


def balanced_assignment(frame: pd.DataFrame, seed: int) -> tuple[pd.DataFrame, float]:
    """Randomized greedy allocation followed by deterministic swap improvement."""

    blocks, matrix, feature_names, weights = make_block_matrix(frame)
    rng = np.random.default_rng(seed)
    best_assignment = None
    best_score = np.inf
    block_priority = matrix[:, 0] + matrix[:, 1:].max(axis=1) * 0.2

    for trial in range(BALANCE_TRIALS):
        jitter = rng.uniform(0.90, 1.10, size=len(blocks))
        order = np.argsort(-(block_priority * jitter))
        assignment = np.full(len(blocks), -1, dtype=int)
        initial_folds = rng.permutation(N_FOLDS)
        for index, block_index in enumerate(order):
            if index < N_FOLDS:
                assignment[block_index] = initial_folds[index]
                continue
            candidate_scores = []
            for fold in rng.permutation(N_FOLDS):
                candidate = assignment.copy()
                candidate[block_index] = fold
                assigned = candidate >= 0
                score = balance_objective(
                    candidate[assigned], matrix[assigned], weights
                )
                candidate_scores.append((score, fold))
            assignment[block_index] = min(candidate_scores)[1]

        current_score = balance_objective(assignment, matrix, weights)
        for _ in range(SWAP_ITERATIONS):
            first, second = rng.integers(0, len(blocks), size=2)
            if first == second or assignment[first] == assignment[second]:
                continue
            candidate = assignment.copy()
            candidate[first], candidate[second] = candidate[second], candidate[first]
            score = balance_objective(candidate, matrix, weights)
            if score < current_score:
                assignment = candidate
                current_score = score
        if current_score < best_score:
            best_assignment = assignment.copy()
            best_score = current_score

    block_assignment = blocks[["block_id", "n_cells", "x_center", "y_center"]].copy()
    block_assignment["fold"] = best_assignment + 1
    block_assignment["seed"] = seed
    block_assignment.attrs["balance_features"] = feature_names
    return block_assignment, best_score


def attach_assignment(frame: pd.DataFrame, block_assignment: pd.DataFrame) -> pd.DataFrame:
    return frame.merge(
        block_assignment[["block_id", "fold"]], on="block_id", how="left", validate="many_to_one"
    )


def fold_balance_summary(frame: pd.DataFrame, block_km: int, repeat: int, seed: int, score: float) -> pd.DataFrame:
    global_mean = frame[TARGET].mean()
    global_sd = frame[TARGET].std(ddof=1)
    rows = []
    for fold in range(1, N_FOLDS + 1):
        subset = frame[frame["fold"] == fold]
        row = {
            "block_km": block_km,
            "repeat": repeat,
            "seed": seed,
            "fold": fold,
            "blocks": subset["block_id"].nunique(),
            "cells": len(subset),
            "cell_share": len(subset) / len(frame),
            "R_mean": subset[TARGET].mean(),
            "R_sd": subset[TARGET].std(ddof=1),
            "R_q10": subset[TARGET].quantile(0.10),
            "R_median": subset[TARGET].median(),
            "R_q90": subset[TARGET].quantile(0.90),
            "R_mean_standardized_difference": (subset[TARGET].mean() - global_mean) / global_sd,
            "balance_objective": score,
        }
        for column, prefix in [
            ("coast_band", "coast"),
            ("landscape_zone", "landscape"),
            ("ecosystem_type", "ecosystem"),
        ]:
            shares = subset[column].value_counts(normalize=True)
            for category in sorted(frame[column].unique()):
                row[f"{prefix}_share:{category}"] = shares.get(category, 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def make_estimator(model_name: str, parameters: dict) -> object:
    scaled = [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    if model_name == "MLR":
        return Pipeline([*scaled, ("model", LinearRegression())])
    if model_name == "SVR":
        base = SVR(
            kernel="rbf",
            cache_size=4_000,
            max_iter=-1,
            tol=1e-3,
            **parameters,
        )
        return TransformedTargetRegressor(
            regressor=Pipeline([*scaled, ("model", base)]),
            transformer=StandardScaler(),
        )
    if model_name == "RF":
        base = RandomForestRegressor(
            random_state=42,
            n_jobs=-1,
            criterion="squared_error",
            **parameters,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", base)])
    if model_name == "XGBoost":
        base = XGBRegressor(
            objective="reg:squarederror",
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            verbosity=0,
            **parameters,
        )
        return Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", base)])
    raise KeyError(model_name)


def parameter_candidates() -> dict[str, list[dict]]:
    return {
        "MLR": [{}],
        "SVR": [
            {"C": 1, "gamma": "scale", "epsilon": 0.10},
            {"C": 1, "gamma": 0.03, "epsilon": 0.03},
            {"C": 1, "gamma": 0.03, "epsilon": 0.10},
            {"C": 10, "gamma": 0.03, "epsilon": 0.10},
        ],
        "RF": [
            {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 4, "max_features": 0.6},
            {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 8, "max_features": 0.8},
            {"n_estimators": 300, "max_depth": 20, "min_samples_leaf": 4, "max_features": 0.8},
        ],
        "XGBoost": [
            {"n_estimators": 500, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "reg_lambda": 3},
            {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.02, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 1, "reg_lambda": 3},
            {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85, "min_child_weight": 3, "reg_lambda": 3},
        ],
    }


def cross_validated_predictions(frame: pd.DataFrame, estimator: object) -> tuple[np.ndarray, list[dict]]:
    predictions = np.full(len(frame), np.nan, dtype=float)
    fold_rows = []
    for fold in range(1, N_FOLDS + 1):
        test = frame["fold"].to_numpy() == fold
        train = ~test
        fitted = clone(estimator).fit(frame.loc[train, FEATURES], frame.loc[train, TARGET])
        predicted = fitted.predict(frame.loc[test, FEATURES])
        predictions[test] = predicted
        fold_rows.append({"fold": fold, **evaluate(frame.loc[test, TARGET].to_numpy(), predicted)})
    if np.isnan(predictions).any():
        raise RuntimeError("Out-of-fold predictions are incomplete")
    return predictions, fold_rows


def select_fixed_parameters(frame: pd.DataFrame) -> tuple[dict[str, dict], pd.DataFrame, pd.DataFrame]:
    block_frame = add_blocks(frame, MAIN_BLOCK_KM)
    assignment, score = balanced_assignment(block_frame, SELECTION_SEED)
    assigned = attach_assignment(block_frame, assignment)
    rows = []
    selected = {}
    for model_name, candidates in parameter_candidates().items():
        for candidate_id, parameters in enumerate(candidates, start=1):
            started = time.perf_counter()
            predictions, fold_rows = cross_validated_predictions(
                assigned, make_estimator(model_name, parameters)
            )
            fold_table = pd.DataFrame(fold_rows)
            pooled = evaluate(assigned[TARGET].to_numpy(), predictions)
            rows.append(
                {
                    "model": model_name,
                    "candidate": candidate_id,
                    "parameters": json.dumps(parameters, ensure_ascii=False, sort_keys=True),
                    "mean_fold_rmse": fold_table["rmse"].mean(),
                    "sd_fold_rmse": fold_table["rmse"].std(ddof=1),
                    "pooled_oof_r2": pooled["r2"],
                    "pooled_oof_rmse": pooled["rmse"],
                    "pooled_oof_mae": pooled["mae"],
                    "elapsed_seconds": time.perf_counter() - started,
                    "selection_seed": SELECTION_SEED,
                    "balance_objective": score,
                }
            )
            print(
                f"parameter selection {model_name} candidate={candidate_id}: "
                f"mean fold RMSE={fold_table['rmse'].mean():.5f}",
                flush=True,
            )
    results = pd.DataFrame(rows)
    for model_name in MODEL_ORDER:
        best = results[results["model"] == model_name].sort_values(
            ["mean_fold_rmse", "pooled_oof_rmse"]
        ).iloc[0]
        selected[model_name] = json.loads(best["parameters"])
    return selected, results, assigned


def evaluate_block_size(
    frame: pd.DataFrame,
    block_km: int,
    fixed_parameters: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    base = add_blocks(frame, block_km)
    metric_rows = []
    prediction_rows = []
    assignment_rows = []
    balance_rows = []
    for repeat, seed in enumerate(EVALUATION_SEEDS, start=1):
        assignment, score = balanced_assignment(base, seed)
        assigned = attach_assignment(base, assignment)
        assignment["block_km"] = block_km
        assignment["repeat"] = repeat
        assignment_rows.append(assignment)
        balance_rows.append(fold_balance_summary(assigned, block_km, repeat, seed, score))
        for model_name in MODEL_ORDER:
            started = time.perf_counter()
            predictions, fold_rows = cross_validated_predictions(
                assigned, make_estimator(model_name, fixed_parameters[model_name])
            )
            pooled = evaluate(assigned[TARGET].to_numpy(), predictions)
            fold_frame = pd.DataFrame(fold_rows)
            metric_rows.append(
                {
                    "block_km": block_km,
                    "repeat": repeat,
                    "seed": seed,
                    "model": model_name,
                    **pooled,
                    "mean_fold_r2": fold_frame["r2"].mean(),
                    "sd_fold_r2": fold_frame["r2"].std(ddof=1),
                    "mean_fold_rmse": fold_frame["rmse"].mean(),
                    "sd_fold_rmse": fold_frame["rmse"].std(ddof=1),
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "cell_id": assigned["cell_id"],
                        "block_km": block_km,
                        "repeat": repeat,
                        "seed": seed,
                        "block_id": assigned["block_id"],
                        "fold": assigned["fold"],
                        "model": model_name,
                        "observed_R": assigned[TARGET],
                        "predicted_R": predictions,
                        "residual": assigned[TARGET].to_numpy() - predictions,
                    }
                )
            )
            print(
                f"block={block_km:02d}km repeat={repeat:02d} {model_name}: "
                f"R2={pooled['r2']:.3f}, RMSE={pooled['rmse']:.5f}",
                flush=True,
            )
    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.concat(assignment_rows, ignore_index=True),
        pd.concat(balance_rows, ignore_index=True),
    )


def ci_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    critical = float(t.ppf(0.975, len(EVALUATION_SEEDS) - 1))
    for (block_km, model), group in metrics.groupby(["block_km", "model"], sort=False):
        row = {"block_km": block_km, "model": model, "repeats": len(group)}
        for metric in ["r2", "rmse", "mae", "nrmse_percent"]:
            mean = group[metric].mean()
            sd = group[metric].std(ddof=1)
            half = critical * sd / np.sqrt(len(group))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_sd"] = sd
            row[f"{metric}_ci95_low"] = mean - half
            row[f"{metric}_ci95_high"] = mean + half
        rows.append(row)
    result = pd.DataFrame(rows)
    result["rank_within_block_by_rmse"] = result.groupby("block_km")["rmse_mean"].rank(
        method="min"
    ).astype(int)
    return result.sort_values(["block_km", "rank_within_block_by_rmse"])


def paired_model_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    """Paired repeat-level differences for all model pairs (A minus B)."""

    rows = []
    critical = float(t.ppf(0.975, len(EVALUATION_SEEDS) - 1))
    for block_km in BLOCK_SIZES_KM:
        subset = metrics[metrics["block_km"] == block_km]
        for first_index, first in enumerate(MODEL_ORDER):
            for second in MODEL_ORDER[first_index + 1 :]:
                left = subset[subset["model"] == first].set_index("repeat")
                right = subset[subset["model"] == second].set_index("repeat")
                row = {
                    "block_km": block_km,
                    "model_A": first,
                    "model_B": second,
                    "repeats": len(left),
                }
                for metric in ["r2", "rmse", "mae"]:
                    difference = left[metric] - right[metric]
                    mean = difference.mean()
                    sd = difference.std(ddof=1)
                    half = critical * sd / np.sqrt(len(difference))
                    row[f"{metric}_difference_A_minus_B_mean"] = mean
                    row[f"{metric}_difference_ci95_low"] = mean - half
                    row[f"{metric}_difference_ci95_high"] = mean + half
                rows.append(row)
    return pd.DataFrame(rows)


def plot_main_performance(metrics: pd.DataFrame) -> None:
    configure_fonts()
    data = metrics[metrics["block_km"] == MAIN_BLOCK_KM]
    figure, axes = plt.subplots(1, 3, figsize=(12.2, 4.2))
    for axis, metric, ylabel in zip(
        axes,
        ["r2", "rmse", "mae"],
        ["空间分块交叉验证 R²", "RMSE", "MAE"],
    ):
        groups = [data[data["model"] == model][metric].to_numpy() for model in MODEL_ORDER]
        box = axis.boxplot(
            groups,
            positions=np.arange(len(MODEL_ORDER)),
            widths=0.55,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#222222", "linewidth": 1.2},
            whiskerprops={"color": "#555555"},
            capprops={"color": "#555555"},
        )
        for patch, model in zip(box["boxes"], MODEL_ORDER):
            patch.set_facecolor(MODEL_COLORS[model])
            patch.set_alpha(0.68)
            patch.set_edgecolor("#444444")
        for position, model in enumerate(MODEL_ORDER):
            values = data[data["model"] == model][metric].to_numpy()
            jitter = np.linspace(-0.12, 0.12, len(values))
            axis.scatter(
                position + jitter,
                values,
                s=17,
                facecolor="white",
                edgecolor=MODEL_COLORS[model],
                linewidth=0.8,
                zorder=3,
            )
        axis.set_xticks(np.arange(len(MODEL_ORDER)), MODEL_ORDER)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        axis.set_axisbelow(True)
    for label, axis in zip(["(a)", "(b)", "(c)"], axes):
        axis.text(-0.15, 1.02, label, transform=axis.transAxes, fontsize=12, fontweight="normal")
    figure.subplots_adjust(left=0.08, right=0.99, bottom=0.15, top=0.95, wspace=0.32)
    stem = FIGURES / "global_models_repeated_10km_spatial_block_cv"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_block_sensitivity(summary: pd.DataFrame) -> None:
    configure_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 4.1))
    for axis, metric, ylabel in [
        (axes[0], "r2", "重复空间分块交叉验证 R²"),
        (axes[1], "rmse", "RMSE"),
    ]:
        for model in MODEL_ORDER:
            data = summary[summary["model"] == model].sort_values("block_km")
            axis.errorbar(
                data["block_km"],
                data[f"{metric}_mean"],
                yerr=[
                    data[f"{metric}_mean"] - data[f"{metric}_ci95_low"],
                    data[f"{metric}_ci95_high"] - data[f"{metric}_mean"],
                ],
                color=MODEL_COLORS[model],
                marker="o",
                markersize=5,
                linewidth=1.5,
                capsize=3,
                label=model,
            )
        axis.set_xticks(BLOCK_SIZES_KM)
        axis.set_xlabel("空间块边长（km）")
        axis.set_ylabel(ylabel)
        axis.grid(color="#D9D9D9", linewidth=0.55)
        axis.set_axisbelow(True)
    axes[1].legend(frameon=False, ncol=2, loc="best")
    for label, axis in zip(["(a)", "(b)"], axes):
        axis.text(-0.14, 1.02, label, transform=axis.transAxes, fontsize=12, fontweight="normal")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.94, wspace=0.30)
    stem = FIGURES / "block_size_sensitivity_5_10_20km"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_assignment_examples(frame: pd.DataFrame, assignments: pd.DataFrame) -> None:
    configure_fonts()
    figure, axes = plt.subplots(1, 3, figsize=(10.8, 4.7))
    for axis, block_km in zip(axes, BLOCK_SIZES_KM):
        selected = assignments[
            (assignments["block_km"] == block_km) & (assignments["repeat"] == 1)
        ][["block_id", "fold"]]
        plotted = add_blocks(frame, block_km).merge(selected, on="block_id", validate="many_to_one")
        for fold in range(1, N_FOLDS + 1):
            subset = plotted[plotted["fold"] == fold]
            axis.scatter(
                subset["x_m"] / 1_000,
                subset["y_m"] / 1_000,
                s=4.0,
                marker="s",
                color=FOLD_COLORS[fold - 1],
                linewidth=0,
                label=f"Fold {fold}",
            )
        axis.set_aspect("equal")
        axis.set_xlabel("投影东向坐标（km）")
        axis.set_title(f"{block_km} km", fontsize=10, fontweight="normal")
        axis.tick_params(labelsize=8)
        axis.grid(color="#E1E1E1", linewidth=0.4)
    axes[0].set_ylabel("投影北向坐标（km）")
    axes[-1].legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1.0))
    for label, axis in zip(["(a)", "(b)", "(c)"], axes):
        axis.text(-0.12, 1.02, label, transform=axis.transAxes, fontsize=12, fontweight="normal")
    figure.subplots_adjust(left=0.08, right=0.88, bottom=0.12, top=0.92, wspace=0.22)
    stem = FIGURES / "spatial_block_fold_assignment_examples"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_notes(
    frame: pd.DataFrame,
    selected: dict[str, dict],
    summary: pd.DataFrame,
    balance: pd.DataFrame,
) -> None:
    main = summary[summary["block_km"] == MAIN_BLOCK_KM].sort_values("rmse_mean")
    best = main.iloc[0]
    max_cell_deviation = (
        balance.assign(deviation=lambda d: (d["cell_share"] - 0.2).abs())["deviation"].max()
    )
    max_r_difference = balance["R_mean_standardized_difference"].abs().max()
    parameters_text = "\n".join(
        f"- {name}: `{json.dumps(parameters, ensure_ascii=False, sort_keys=True)}`"
        for name, parameters in selected.items()
    )
    paired_path = OUTPUT / "09_paired_model_differences.csv"
    paired_sentence = ""
    if paired_path.exists():
        paired = pd.read_csv(paired_path)
        rf_xgb = paired[
            (paired["block_km"] == MAIN_BLOCK_KM)
            & (paired["model_A"] == "RF")
            & (paired["model_B"] == "XGBoost")
        ].iloc[0]
        paired_sentence = (
            f"在逐次配对比较中，RF相对XGBoost的平均RMSE差为"
            f"{rf_xgb['rmse_difference_A_minus_B_mean']:.6f}"
            f"（95% CI {rf_xgb['rmse_difference_ci95_low']:.6f}—"
            f"{rf_xgb['rmse_difference_ci95_high']:.6f}；负值表示RF误差更低）。"
        )
    text = f"""# 重复空间分块交叉验证下的全局机器学习模型比较

## 方法

- 分析样本为严格陆地生态系统掩膜内{len(frame):,}个1 km格网，因变量为生态系统热风险指数R。
- 解释变量为筛选后的11变量；距海岸线距离与主导风向加权迎风面积指数不进入模型。投影坐标仅用于构建规则空间块与宏观方位分层。
- 主分析使用10 km×10 km规则空间块。同一空间块内全部1 km格网整体进入同一折，不拆分；不设置训练—验证缓冲区。
- 空间块而非单格网被分配至5折。分配目标同时平衡有效格网数量、5级滨海—内陆距离带、城区—农业区—北部山区构成、CLCD 2021主导生态系统类型和粗尺度方位分层，以保证每折由分散在天津不同区域的多个空间块构成。
- R没有参与折分配优化，仅在分配完成后检查各折均值、标准差和分位数是否极端失衡。
- 使用独立种子{SELECTION_SEED}对应的一组10 km空间块分配，在预设小候选集中选择一次参数；随后固定参数，在另外10组种子（{EVALUATION_SEEDS[0]}—{EVALUATION_SEEDS[-1]}）下重复5折评价。
- 以完全相同流程对5 km与20 km空间块进行敏感性检验。所有检验均为无缓冲空间分块交叉验证。

固定参数：
{parameters_text}

## 结果概述

10 km主分析中，按10次重复的平均RMSE排序，{best['model']}表现最佳：R²={best['r2_mean']:.3f}（95% CI {best['r2_ci95_low']:.3f}–{best['r2_ci95_high']:.3f}），RMSE={best['rmse_mean']:.5f}，MAE={best['mae_mean']:.5f}。{paired_sentence}完整结果见汇总表与敏感性图。

全部分配中，各折格网占比相对理论20%的最大绝对偏差为{max_cell_deviation:.3f}；各折R均值相对全域均值的最大标准化差异为{max_r_difference:.3f}个全域标准差。该R诊断仅用于检查极端失衡，没有反向参与fold优化。

## 解释边界

空间分块交叉验证评价的是模型对未见空间块的泛化能力，而不是因果效应。未设置缓冲区意味着相邻训练块和验证块可在边界处接触，因此本方案比单格网随机交叉验证严格，但通常比带缓冲空间交叉验证宽松；这一点应在论文方法和局限性中明确说明。
"""
    (OUTPUT / "重复空间分块CV_方法与结果.md").write_text(text, encoding="utf-8")
    captions = """图X 基于重复空间分块交叉验证的全局模型性能比较。（a）决定系数R²；（b）均方根误差RMSE；（c）平均绝对误差MAE。箱体与散点表示10次独立空间块分配结果，空间块边长为10 km，每次均进行5折无缓冲空间分块交叉验证。\n\n图Y 空间块尺度敏感性分析。（a）不同空间块边长下的R²；（b）不同空间块边长下的RMSE。点为10次重复均值，误差线为95%置信区间。\n\n图Z 规则空间块到5个fold的示例分配。（a）5 km；（b）10 km；（c）20 km。同色格网属于同一fold，同一规则空间块内的1 km格网未被拆分。"""
    (FIGURES / "figure_captions_zh.txt").write_text(captions, encoding="utf-8")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT_TABLE)
    required = ["cell_id", "row", "column", "x_m", "y_m", TARGET, *FEATURES, "coast_distance_km"]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}")
    if frame[required].isna().any().any():
        raise ValueError("Required fields contain missing values")
    if frame["cell_id"].duplicated().any():
        raise ValueError("cell_id is not unique")
    frame = add_balance_strata(add_clcd_dominant(frame))
    frame.to_csv(OUTPUT / "00_model_table_11f_with_balance_strata.csv", index=False, encoding="utf-8-sig")

    selected, selection_results, selection_frame = select_fixed_parameters(frame)
    selection_results.to_csv(OUTPUT / "01_small_candidate_parameter_selection.csv", index=False, encoding="utf-8-sig")
    (OUTPUT / "02_fixed_parameters.json").write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    selection_frame[["cell_id", "block_id", "fold"]].to_csv(
        OUTPUT / "03_parameter_selection_10km_assignment.csv", index=False, encoding="utf-8-sig"
    )

    all_metrics = []
    all_predictions = []
    all_assignments = []
    all_balance = []
    for block_km in BLOCK_SIZES_KM:
        metrics, predictions, assignments, balance = evaluate_block_size(
            frame, block_km, selected
        )
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        all_assignments.append(assignments)
        all_balance.append(balance)

    metrics = pd.concat(all_metrics, ignore_index=True)
    predictions = pd.concat(all_predictions, ignore_index=True)
    assignments = pd.concat(all_assignments, ignore_index=True)
    balance = pd.concat(all_balance, ignore_index=True)
    summary = ci_summary(metrics)
    paired = paired_model_differences(metrics)

    metrics.to_csv(OUTPUT / "04_repeat_level_model_metrics.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUTPUT / "05_model_performance_summary_by_block_size.csv", index=False, encoding="utf-8-sig")
    assignments.to_csv(OUTPUT / "06_block_fold_assignments_all_repeats.csv", index=False, encoding="utf-8-sig")
    balance.to_csv(OUTPUT / "07_fold_balance_and_R_diagnostics.csv", index=False, encoding="utf-8-sig")
    predictions.to_csv(OUTPUT / "08_all_out_of_fold_predictions.csv", index=False, encoding="utf-8-sig")
    paired.to_csv(OUTPUT / "09_paired_model_differences.csv", index=False, encoding="utf-8-sig")

    plot_main_performance(metrics)
    plot_block_sensitivity(summary)
    plot_assignment_examples(frame, assignments)
    write_notes(frame, selected, summary, balance)
    print("\nFixed parameters")
    print(json.dumps(selected, ensure_ascii=False, indent=2))
    print("\nSummary")
    print(summary.to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    main()
