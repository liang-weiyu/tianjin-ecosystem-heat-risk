"""Calculate and map Combined GWRF SHAP contributions for all land cells.

Combined SHAP follows the selected prediction mixture exactly:

    combined_phi = local_weight * local_phi + global_weight * global_phi

The local component refits the final K-neighbour weighted RF at every focal
cell using the same deterministic seed and RF settings as ``run_gwrf_11f``.
Chunk checkpoints make the expensive calculation restartable.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Patch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
import shap

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import run_gwrf_11f as gwrf


OUTPUT = gwrf.OUTPUT / "11_Combined_GWRF_SHAP"
CHECKPOINTS = OUTPUT / "checkpoints"
FIGURES = OUTPUT / "figures"
GLOBAL_SHAP = (
    gwrf.MODEL_ROOT / "08_RF_SHAP_global_11f" / "01_RF_SHAP_values_all_cells.csv"
)
N_ESTIMATORS = 100
GLOBAL_IMPORTANCE_ORDER = [2, 10, 8, 1, 9, 0, 7, 3, 6, 5, 4]
CATEGORY_COLORS = [
    "#67AFCB",  # 地表高程
    "#A8C977",  # 夏季平均风速
    "#E4775E",  # 地表反照率
    "#8A7EBB",  # 建筑覆盖率
    "#9B624B",  # 平均建筑高度
    "#9A9488",  # 建筑高度离散度
    "#BA7EAC",  # 天空可视因子
    "#6D91C4",  # 路网密度
    "#E7C568",  # 夜间灯光强度
    "#69B8A8",  # 人口密度
    "#E8A25C",  # 夏季累计降水
]
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "combined_shap_diverging",
    ["#3E6FA8", "#9FC2DC", "#E7E2DA", "#E8AAA0", "#C85B4B"],
    N=256,
)


def normalize_shap(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 2 and array.shape[0] == 1:
        array = array[0]
    return array.reshape(-1)


def fit_local_shap_one(
    position: int,
    neighbour_indices: np.ndarray,
    distances: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    cell_id: int,
    n_estimators: int,
) -> np.ndarray:
    neighbours = neighbour_indices[position]
    weights, _, _ = gwrf.bisquare_weights(distances[position])
    model = gwrf.make_rf(
        n_estimators=n_estimators,
        random_state=int((gwrf.SEARCH_SEED + cell_id * 104729) % (2**31 - 1)),
    )
    model.fit(x[neighbours], y[neighbours], sample_weight=weights)
    focal_x = x[position : position + 1]
    prediction = float(model.predict(focal_x)[0])
    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    explanation = explainer(focal_x, check_additivity=False)
    values = normalize_shap(explanation.values)
    base = float(np.asarray(explanation.base_values).reshape(-1)[0])
    return np.r_[prediction, base, values]


def compute_local_shap(
    frame: pd.DataFrame,
    bandwidth: int,
    n_estimators: int,
    n_jobs: int,
    chunk_size: int,
) -> np.ndarray:
    x = frame[gwrf.FEATURES].to_numpy(dtype=float)
    y = frame[gwrf.TARGET].to_numpy(dtype=float)
    xy = frame[gwrf.COORDINATES].to_numpy(dtype=float)
    distances, neighbour_indices = cKDTree(xy).query(xy, k=bandwidth, workers=-1)
    chunks = []
    for start in range(0, len(frame), chunk_size):
        stop = min(start + chunk_size, len(frame))
        checkpoint = CHECKPOINTS / f"local_SHAP_{start:05d}_{stop - 1:05d}.npz"
        if checkpoint.exists():
            result = np.load(checkpoint)["result"]
        else:
            started = time.perf_counter()
            result = np.asarray(
                Parallel(n_jobs=n_jobs, prefer="threads", batch_size=2)(
                    delayed(fit_local_shap_one)(
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
            np.savez_compressed(checkpoint, result=result)
            print(
                f"Combined SHAP local forests: {stop}/{len(frame)} completed "
                f"in {(time.perf_counter() - started) / 60:.2f} min",
                flush=True,
            )
        chunks.append(result)
    return np.vstack(chunks)


def combine_components(frame: pd.DataFrame, local_result: np.ndarray) -> pd.DataFrame:
    selection = json.loads(
        (gwrf.OUTPUT / "04_selected_bandwidth.json").read_text(encoding="utf-8")
    )
    local_weight = float(selection["selected_local_weight"])
    global_weight = float(selection["selected_global_weight"])
    global_table = pd.read_csv(GLOBAL_SHAP).set_index("cell_id").loc[frame["cell_id"]]
    global_values = global_table[
        [f"SHAP_{feature}" for feature in gwrf.FEATURES]
    ].to_numpy(dtype=float)
    global_prediction = global_table["predicted_R"].to_numpy(dtype=float)
    global_base_by_cell = global_prediction - global_values.sum(axis=1)
    global_base = float(np.median(global_base_by_cell))

    local_prediction = local_result[:, 0]
    local_base = local_result[:, 1]
    local_values = local_result[:, 2:]
    combined_values = local_weight * local_values + global_weight * global_values
    combined_prediction = local_weight * local_prediction + global_weight * global_prediction
    combined_base = local_weight * local_base + global_weight * global_base
    additivity_error = np.abs(
        combined_base + combined_values.sum(axis=1) - combined_prediction
    )

    output = frame[["cell_id", "row", "column", "x_m", "y_m", "R"]].copy()
    output["local_predicted_R"] = local_prediction
    output["global_predicted_R"] = global_prediction
    output["combined_predicted_R"] = combined_prediction
    output["combined_base_value"] = combined_base
    for index, feature in enumerate(gwrf.FEATURES):
        output[f"Combined_SHAP_{feature}"] = combined_values[:, index]
    dominant_index = np.argmax(np.abs(combined_values), axis=1)
    output["dominant_combined_shap_code"] = dominant_index + 1
    output["dominant_combined_shap_factor"] = [
        gwrf.FEATURE_LABELS_ZH[gwrf.FEATURES[index]] for index in dominant_index
    ]
    output["dominant_combined_shap_value"] = combined_values[
        np.arange(len(output)), dominant_index
    ]
    output.to_csv(
        OUTPUT / "01_Combined_GWRF_SHAP_by_cell.csv",
        index=False,
        encoding="utf-8-sig",
    )

    qa = {
        "formula": "combined_phi = local_weight * local_phi + global_weight * global_phi",
        "local_weight": local_weight,
        "global_weight": global_weight,
        "bandwidth_neighbors": int(selection["selected_bandwidth_neighbors"]),
        "local_rf_n_estimators": N_ESTIMATORS,
        "n_cells": int(len(output)),
        "global_base_value_median": global_base,
        "global_base_value_range": [
            float(global_base_by_cell.min()), float(global_base_by_cell.max())
        ],
        "maximum_combined_additivity_error": float(additivity_error.max()),
        "mean_combined_additivity_error": float(additivity_error.mean()),
        "dominant_rule": "maximum absolute Combined SHAP at each cell",
    }
    (OUTPUT / "00_Combined_GWRF_SHAP_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output


def write_rasters(frame: pd.DataFrame) -> None:
    for feature in gwrf.FEATURES:
        gwrf.write_raster(
            frame,
            frame[f"Combined_SHAP_{feature}"].to_numpy(dtype="float32"),
            OUTPUT / f"Combined_GWRF_SHAP_{feature}_1km.tif",
            "float32",
        )
    gwrf.write_raster(
        frame,
        frame["dominant_combined_shap_code"].to_numpy(dtype="uint8"),
        OUTPUT / "Combined_GWRF_SHAP_dominant_factor_code_1km.tif",
        "uint8",
    )


def tick_text(value: float) -> str:
    absolute = abs(value)
    if absolute >= 0.01:
        return f"{value:.3f}"
    return f"{value:.4f}"


def plot_combined_shap(frame: pd.DataFrame) -> None:
    times = fm.FontProperties(fname=font_path("english"), size=9)
    times_small = fm.FontProperties(fname=font_path("english"), size=6.6)
    simsun = fm.FontProperties(fname=font_path("chinese"), size=9)
    simsun_small = fm.FontProperties(fname=font_path("chinese"), size=6.3)
    geometry, map_bounds, raster_bounds = gwrf.load_publication_map_context()
    west, south, east, north = raster_bounds
    raster_extent = [west, east, south, north]
    category_cmap = ListedColormap(CATEGORY_COLORS)

    figure, axes = plt.subplots(3, 4, figsize=(10.2, 10.0), dpi=180)
    for panel_index, feature_index in enumerate(GLOBAL_IMPORTANCE_ORDER):
        feature = gwrf.FEATURES[feature_index]
        axis = axes.flat[panel_index]
        values = frame[f"Combined_SHAP_{feature}"].to_numpy(dtype=float)
        array = gwrf.frame_to_array(frame, values)
        limit = float(np.nanquantile(np.abs(values), 0.99))
        limit = max(limit, np.finfo(float).eps)
        image = axis.imshow(
            array,
            extent=raster_extent,
            cmap=DIVERGING_CMAP,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
            origin="upper",
            interpolation="nearest",
        )
        gwrf.format_publication_map(
            axis, image, geometry, map_bounds, compact_scale=True
        )
        axis.text(
            0.01, 1.012, f"({chr(97 + panel_index)})",
            transform=axis.transAxes, ha="left", va="bottom",
            fontproperties=times, clip_on=False,
        )
        axis.text(
            0.075, 1.012, gwrf.FEATURE_LABELS_ZH[feature],
            transform=axis.transAxes, ha="left", va="bottom",
            fontproperties=simsun, clip_on=False,
        )
        color_axis = inset_axes(
            axis, width="3.4%", height="35%", loc="lower left",
            bbox_to_anchor=(0.79, 0.30, 1.0, 1.0),
            bbox_transform=axis.transAxes, borderpad=0,
        )
        colorbar = figure.colorbar(image, cax=color_axis, orientation="vertical")
        colorbar.set_ticks([-limit, 0.0, limit])
        colorbar.set_ticklabels([tick_text(-limit), "0", tick_text(limit)])
        colorbar.ax.set_title("Combined\nSHAP", fontproperties=times_small, pad=6)
        for label in colorbar.ax.get_yticklabels():
            label.set_fontproperties(times_small)

    dominant_axis = axes.flat[-1]
    dominant_array = gwrf.frame_to_array(
        frame, frame["dominant_combined_shap_code"].to_numpy(dtype=float)
    )
    dominant_image = dominant_axis.imshow(
        dominant_array,
        extent=raster_extent,
        cmap=category_cmap,
        vmin=0.5,
        vmax=len(gwrf.FEATURES) + 0.5,
        origin="upper",
        interpolation="nearest",
    )
    gwrf.format_publication_map(
        dominant_axis, dominant_image, geometry, map_bounds, compact_scale=True
    )
    dominant_axis.text(
        0.01, 1.012, "(l)", transform=dominant_axis.transAxes,
        ha="left", va="bottom", fontproperties=times, clip_on=False,
    )
    dominant_axis.text(
        0.075, 1.012, "局地主导影响因素", transform=dominant_axis.transAxes,
        ha="left", va="bottom", fontproperties=simsun, clip_on=False,
    )
    present = set(frame["dominant_combined_shap_code"].unique())
    present_codes = [code + 1 for code in GLOBAL_IMPORTANCE_ORDER if code + 1 in present]
    handles = [
        Patch(
            facecolor=CATEGORY_COLORS[code - 1], edgecolor="#555555",
            linewidth=0.3, label=gwrf.FEATURE_LABELS_ZH[gwrf.FEATURES[code - 1]],
        )
        for code in present_codes
    ]
    legend = dominant_axis.legend(
        handles=handles, loc="center left", bbox_to_anchor=(0.72, 0.46),
        title="影响因素", title_fontproperties=simsun_small,
        frameon=False, prop=simsun_small, handlelength=1.3,
        handleheight=0.75, labelspacing=0.18, handletextpad=0.28,
    )
    legend._legend_box.align = "left"
    figure.subplots_adjust(
        left=0.03, right=0.97, top=0.985, bottom=0.025,
        wspace=0.04, hspace=0.13,
    )
    gwrf.save_figure_all_formats(
        figure, FIGURES / "Combined_GWRF_SHAP_11_factors_and_dominant_4x3"
    )
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=["compute", "plot", "all"], default="all")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--trees", type=int, default=N_ESTIMATORS)
    parser.add_argument("--chunk-size", type=int, default=250)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    gwrf.configure_fonts()
    frame, _ = gwrf.load_and_validate()
    output_csv = OUTPUT / "01_Combined_GWRF_SHAP_by_cell.csv"
    if args.phase in {"compute", "all"}:
        selection = json.loads(
            (gwrf.OUTPUT / "04_selected_bandwidth.json").read_text(encoding="utf-8")
        )
        local_result = compute_local_shap(
            frame,
            int(selection["selected_bandwidth_neighbors"]),
            args.trees,
            args.jobs,
            args.chunk_size,
        )
        result = combine_components(frame, local_result)
        write_rasters(result)
    elif output_csv.exists():
        result = pd.read_csv(output_csv)
    else:
        raise FileNotFoundError("Run --phase compute before --phase plot.")
    if args.phase in {"plot", "all"}:
        plot_combined_shap(result)
    print(f"Combined GWRF SHAP completed: {OUTPUT}")


if __name__ == "__main__":
    main()
