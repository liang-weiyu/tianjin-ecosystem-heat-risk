"""Tree-number stability check for the fixed K=300, alpha=0.5 GWRF."""

from __future__ import annotations

import json
import time
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree

import run_gwrf_11f as gwrf


TREE_COUNTS = (100, 200, 300)
BANDWIDTH = 300
LOCAL_WEIGHT = 0.5
SEED = 20260815
OUTPUT = gwrf.OUTPUT.parent / "11_GWRF_tree_stability_K300_alpha05"


def predict_fold(frame: pd.DataFrame, fold_array: np.ndarray, fold: int, n_jobs: int) -> pd.DataFrame:
    checkpoint = OUTPUT / f"checkpoint_seed_{SEED}_fold_{fold}_max300.csv"
    if checkpoint.exists():
        return pd.read_csv(checkpoint)

    train_mask = fold_array != fold
    validation_indices = np.flatnonzero(fold_array == fold)
    train = frame.loc[train_mask]
    validation = frame.iloc[validation_indices]
    train_x = train[gwrf.FEATURES].to_numpy(dtype=float)
    train_y = train[gwrf.TARGET].to_numpy(dtype=float)
    train_xy = train[gwrf.COORDINATES].to_numpy(dtype=float)
    validation_x = validation[gwrf.FEATURES].to_numpy(dtype=float)
    validation_xy = validation[gwrf.COORDINATES].to_numpy(dtype=float)
    distances, neighbour_indices = cKDTree(train_xy).query(
        validation_xy, k=BANDWIDTH, workers=-1
    )

    def task(position: int) -> tuple[float, ...]:
        neighbours = neighbour_indices[position]
        weights, radius, effective_n = gwrf.bisquare_weights(distances[position])
        cell_id = int(validation.iloc[position]["cell_id"])
        random_state = int((SEED + fold + cell_id * 104729) % (2**31 - 1))
        model = gwrf.make_rf(max(TREE_COUNTS), random_state)
        model.fit(train_x[neighbours], train_y[neighbours], sample_weight=weights)
        target = validation_x[position : position + 1]
        tree_predictions = np.fromiter(
            (tree.predict(target)[0] for tree in model.estimators_),
            dtype=float,
            count=max(TREE_COUNTS),
        )
        predictions = tuple(float(tree_predictions[:n].mean()) for n in TREE_COUNTS)
        return (*predictions, radius / 1000.0, effective_n)

    values = np.asarray(
        Parallel(n_jobs=n_jobs, prefer="processes", batch_size=4, max_nbytes="10M")(
            delayed(task)(position) for position in range(len(validation))
        )
    )
    result = pd.DataFrame(
        {
            "cell_id": validation["cell_id"].to_numpy(),
            "observed_R": validation[gwrf.TARGET].to_numpy(),
            "fold": fold,
            "local_pred_100": values[:, 0],
            "local_pred_200": values[:, 1],
            "local_pred_300": values[:, 2],
            "local_radius_km": values[:, 3],
            "effective_local_n": values[:, 4],
        }
    )
    result.to_csv(checkpoint, index=False, encoding="utf-8-sig")
    return result


def make_figure(metrics: pd.DataFrame) -> None:
    figures = OUTPUT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    times = fm.FontProperties(fname=font_path("english"), size=9)
    simsun = fm.FontProperties(fname=font_path("chinese"), size=9)
    figure, axes = plt.subplots(1, 3, figsize=(8.4, 2.8), dpi=180)
    for axis, metric, label, panel in zip(
        axes, ("r2", "rmse", "mae"), ("R²", "RMSE", "MAE"), ("(a)", "(b)", "(c)")
    ):
        baseline = float(metrics.loc[metrics["n_estimators"] == 100, metric].iloc[0])
        relative_change = (metrics[metric] / baseline - 1.0) * 100.0
        axis.plot(metrics["n_estimators"], relative_change, color="#397FB9", marker="o",
                  markerfacecolor="white", markeredgewidth=1.2, linewidth=1.5)
        axis.axhline(0.0, color="#777777", linewidth=0.7, linestyle="--")
        axis.set_xlabel("决策树数量", fontproperties=simsun)
        axis.set_ylabel(f"{label}相对变化（%）", fontproperties=simsun)
        axis.text(0.0, 1.03, panel, transform=axis.transAxes, fontproperties=times)
        axis.set_xticks(TREE_COUNTS)
        axis.grid(axis="y", color="#DDDDDD", linewidth=0.6)
        axis.spines[["top", "right"]].set_visible(False)
        for tick in axis.get_xticklabels() + axis.get_yticklabels():
            tick.set_fontproperties(times)
    figure.subplots_adjust(left=0.08, right=0.98, bottom=0.22, top=0.91, wspace=0.34)
    for extension, dpi in (("png", 600), ("tif", 600), ("pdf", 300)):
        figure.savefig(figures / f"GWRF_tree_number_stability.{extension}", dpi=dpi,
                       bbox_inches="tight")
    plt.close(figure)


def main(n_jobs: int = 8) -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame, _ = gwrf.load_and_validate()
    assignment = gwrf.cell_fold_assignment(frame, SEED)
    fold_by_cell = assignment.set_index("cell_id")["fold"]
    fold_array = frame["cell_id"].map(fold_by_cell).to_numpy(dtype=int)
    start = time.perf_counter()
    parts = []
    for fold in range(1, 6):
        parts.append(predict_fold(frame, fold_array, fold, n_jobs))
        print(f"tree stability fold={fold}/5 completed", flush=True)
    predictions = pd.concat(parts, ignore_index=True).sort_values("cell_id")
    global_oof = pd.read_csv(gwrf.CV_ROOT / "08_all_out_of_fold_predictions.csv")
    global_oof = global_oof.loc[
        (global_oof["block_km"] == 10)
        & (global_oof["seed"] == SEED)
        & (global_oof["model"] == "RF"),
        ["cell_id", "predicted_R"],
    ].rename(columns={"predicted_R": "global_predicted_R"})
    predictions = predictions.merge(global_oof, on="cell_id", validate="one_to_one")

    rows = []
    for n_trees in TREE_COUNTS:
        local_column = f"local_pred_{n_trees}"
        combined_column = f"combined_pred_{n_trees}"
        predictions[combined_column] = (
            LOCAL_WEIGHT * predictions[local_column]
            + (1.0 - LOCAL_WEIGHT) * predictions["global_predicted_R"]
        )
        values = gwrf.model_metrics(
            predictions["observed_R"].to_numpy(), predictions[combined_column].to_numpy()
        )
        values.update(
            {"n_estimators": n_trees, "bandwidth_neighbors": BANDWIDTH,
             "local_weight": LOCAL_WEIGHT, "seed": SEED,
             "n_validation_cells": len(predictions)}
        )
        rows.append(values)
    metrics = pd.DataFrame(rows).sort_values("n_estimators")
    predictions.to_csv(OUTPUT / "01_tree_stability_oof_predictions.csv", index=False,
                       encoding="utf-8-sig")
    metrics.to_csv(OUTPUT / "02_tree_stability_metrics.csv", index=False,
                   encoding="utf-8-sig")
    correlations = predictions[[f"combined_pred_{n}" for n in TREE_COUNTS]].corr()
    correlations.to_csv(OUTPUT / "03_prediction_correlations.csv", encoding="utf-8-sig")
    relative = metrics.copy()
    for metric in ("r2", "rmse", "mae"):
        baseline = float(relative.loc[relative["n_estimators"] == 100, metric].iloc[0])
        relative[f"{metric}_change_vs_100_percent"] = (relative[metric] / baseline - 1.0) * 100.0
    relative.to_csv(OUTPUT / "04_relative_changes_vs_100.csv", index=False,
                    encoding="utf-8-sig")
    summary = {
        "bandwidth_neighbors": BANDWIDTH,
        "local_weight": LOCAL_WEIGHT,
        "global_weight": 1.0 - LOCAL_WEIGHT,
        "spatial_block_km": 10,
        "seed": SEED,
        "tree_counts": list(TREE_COUNTS),
        "elapsed_minutes": (time.perf_counter() - start) / 60.0,
        "note": "Each local forest was fit once with 300 trees; predictions use the first 100, 200, or 300 trees from the same deterministic tree sequence.",
    }
    (OUTPUT / "00_tree_stability_design.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    make_figure(metrics)
    print(metrics.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
