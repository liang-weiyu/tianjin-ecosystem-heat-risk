"""Compare RF and XGBoost TreeSHAP rankings with spatial-block bootstrap."""

from __future__ import annotations

import json
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, rankdata, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor, __version__ as xgboost_version
import shap

import analyze_rf_shap_global as rf_analysis


OUTPUT = (
    rf_analysis.STRICT
    / "08_background_driver_model"
    / "09_XGBoost_SHAP_ranking_robustness"
)
FIGURES = OUTPUT / "figures"
RF_OUTPUT = rf_analysis.OUTPUT

RANDOM_SEED = 20260815
BOOTSTRAPS = 500
RF_COLOR = "#28A058"
XGB_COLOR = "#186E4B"
CONNECTOR_COLOR = "#A7A7A7"
GRID_COLOR = "#D9D9D9"


def save_figure(figure: plt.Figure, name: str) -> None:
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


def normalize_shap(values) -> np.ndarray:
    array = np.asarray(values)
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 2:
        raise ValueError(f"Unexpected SHAP shape: {array.shape}")
    return array


def rank_descending(values: np.ndarray) -> np.ndarray:
    return rankdata(-np.asarray(values), method="average")


def ranking_metrics(rf_values: np.ndarray, xgb_values: np.ndarray) -> dict:
    rf_ranks = rank_descending(rf_values)
    xgb_ranks = rank_descending(xgb_values)
    top3_rf = set(np.argsort(-rf_values)[:3])
    top3_xgb = set(np.argsort(-xgb_values)[:3])
    top5_rf = set(np.argsort(-rf_values)[:5])
    top5_xgb = set(np.argsort(-xgb_values)[:5])
    return {
        "spearman_rho": float(spearmanr(rf_ranks, xgb_ranks).statistic),
        "kendall_tau_b": float(kendalltau(rf_ranks, xgb_ranks).statistic),
        "importance_share_pearson_r": float(pearsonr(rf_values, xgb_values).statistic),
        "top3_overlap_count": len(top3_rf & top3_xgb),
        "top3_overlap_rate": len(top3_rf & top3_xgb) / 3,
        "top5_overlap_count": len(top5_rf & top5_xgb),
        "top5_overlap_rate": len(top5_rf & top5_xgb) / 5,
        "top5_jaccard": len(top5_rf & top5_xgb) / len(top5_rf | top5_xgb),
        "mean_absolute_rank_difference": float(np.mean(np.abs(rf_ranks - xgb_ranks))),
        "maximum_absolute_rank_difference": float(np.max(np.abs(rf_ranks - xgb_ranks))),
    }


def spatial_block_bootstrap(
    frame: pd.DataFrame,
    rf_shap: np.ndarray,
    xgb_shap: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_SEED)
    blocks = rf_analysis.build_blocks(frame, block_km=10).to_numpy()
    unique_blocks = np.unique(blocks)
    block_indices = {block: np.flatnonzero(blocks == block) for block in unique_blocks}

    rf_ranks = np.empty((BOOTSTRAPS, len(rf_analysis.FEATURES)), dtype=float)
    xgb_ranks = np.empty_like(rf_ranks)
    metric_rows = []
    for bootstrap in range(BOOTSTRAPS):
        sampled_blocks = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        sampled_indices = np.concatenate(
            [block_indices[block] for block in sampled_blocks]
        )
        rf_importance = np.mean(np.abs(rf_shap[sampled_indices]), axis=0)
        xgb_importance = np.mean(np.abs(xgb_shap[sampled_indices]), axis=0)
        rf_ranks[bootstrap] = rank_descending(rf_importance)
        xgb_ranks[bootstrap] = rank_descending(xgb_importance)
        metrics = ranking_metrics(rf_importance, xgb_importance)
        metrics["bootstrap"] = bootstrap + 1
        metrics["sampled_cells_with_replacement"] = len(sampled_indices)
        metric_rows.append(metrics)

    summaries = []
    for index, feature in enumerate(rf_analysis.FEATURES):
        summaries.append(
            {
                "factor": feature,
                "chinese_name": rf_analysis.CHINESE[feature],
                "rf_rank_bootstrap_median": np.median(rf_ranks[:, index]),
                "rf_rank_ci95_low": np.quantile(rf_ranks[:, index], 0.025),
                "rf_rank_ci95_high": np.quantile(rf_ranks[:, index], 0.975),
                "rf_probability_top3": np.mean(rf_ranks[:, index] <= 3),
                "rf_probability_top5": np.mean(rf_ranks[:, index] <= 5),
                "xgb_rank_bootstrap_median": np.median(xgb_ranks[:, index]),
                "xgb_rank_ci95_low": np.quantile(xgb_ranks[:, index], 0.025),
                "xgb_rank_ci95_high": np.quantile(xgb_ranks[:, index], 0.975),
                "xgb_probability_top3": np.mean(xgb_ranks[:, index] <= 3),
                "xgb_probability_top5": np.mean(xgb_ranks[:, index] <= 5),
            }
        )
    return pd.DataFrame(summaries), pd.DataFrame(metric_rows)


def plot_ranking_comparison(
    comparison: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
    metrics: dict,
) -> None:
    rf_analysis.configure_fonts()
    data = comparison.merge(
        bootstrap_summary,
        on=["factor", "chinese_name"],
        validate="one_to_one",
    ).sort_values(["mean_rank", "rf_rank"])
    positions = np.arange(len(data))

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(12.6, 6.2),
        gridspec_kw={"width_ratios": [1.10, 1.0]},
    )

    height = 0.34
    axes[0].barh(
        positions - height / 2,
        data["rf_importance_share_percent"],
        height=height,
        color=RF_COLOR,
        edgecolor="#374151",
        linewidth=0.45,
        label="RF",
    )
    axes[0].barh(
        positions + height / 2,
        data["xgb_importance_share_percent"],
        height=height,
        color=XGB_COLOR,
        edgecolor="#6B4A24",
        linewidth=0.45,
        label="XGBoost",
    )
    axes[0].set_yticks(positions, data["chinese_name"])
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Mean |SHAP|贡献占比（%）")
    axes[0].grid(axis="x", color=GRID_COLOR, linewidth=0.55)
    axes[0].set_axisbelow(True)
    axes[0].legend(frameon=False, loc="lower right")

    for position, (_, row) in zip(positions, data.iterrows()):
        rf_rank = row["rf_rank"]
        xgb_rank = row["xgb_rank"]
        axes[1].plot(
            [rf_rank, xgb_rank],
            [position - 0.10, position + 0.10],
            color=CONNECTOR_COLOR,
            linewidth=0.9,
            zorder=1,
        )

        rf_low = min(row["rf_rank_ci95_low"], rf_rank)
        rf_high = max(row["rf_rank_ci95_high"], rf_rank)
        xgb_low = min(row["xgb_rank_ci95_low"], xgb_rank)
        xgb_high = max(row["xgb_rank_ci95_high"], xgb_rank)
        axes[1].errorbar(
            rf_rank,
            position - 0.10,
            xerr=[[rf_rank - rf_low], [rf_high - rf_rank]],
            fmt="o",
            color=RF_COLOR,
            ecolor=RF_COLOR,
            markersize=5.2,
            elinewidth=1.0,
            capsize=2.2,
            zorder=3,
            label="RF" if position == 0 else None,
        )
        axes[1].errorbar(
            xgb_rank,
            position + 0.10,
            xerr=[[xgb_rank - xgb_low], [xgb_high - xgb_rank]],
            fmt="s",
            color=XGB_COLOR,
            ecolor=XGB_COLOR,
            markersize=4.8,
            elinewidth=1.0,
            capsize=2.2,
            zorder=3,
            label="XGBoost" if position == 0 else None,
        )

    axes[1].set_yticks(positions, [])
    axes[1].invert_yaxis()
    axes[1].set_xlim(0.5, len(data) + 0.5)
    axes[1].set_xticks(np.arange(1, len(data) + 1))
    axes[1].set_xlabel("重要性排序（1=最高）")
    axes[1].grid(axis="x", color=GRID_COLOR, linewidth=0.55)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="lower right")
    annotation = (
        f"Spearman ρ = {metrics['spearman_rho']:.3f}\n"
        f"Kendall τ = {metrics['kendall_tau_b']:.3f}\n"
        f"Top-5重合 = {metrics['top5_overlap_count']}/5"
    )
    axes[1].text(
        0.98,
        0.97,
        annotation,
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "#B8B8B8", "alpha": 0.90, "pad": 4},
    )

    axes[0].text(-0.15, 1.03, "(a)", transform=axes[0].transAxes, fontsize=11)
    axes[1].text(-0.08, 1.03, "(b)", transform=axes[1].transAxes, fontsize=11)
    figure.subplots_adjust(left=0.19, right=0.98, bottom=0.12, top=0.96, wspace=0.16)
    save_figure(figure, "RF_XGBoost_SHAP_ranking_robustness")


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(rf_analysis.INPUT)
    required = ["cell_id", "x_m", "y_m", rf_analysis.TARGET, *rf_analysis.FEATURES]
    missing_values = int(frame[required].isna().sum().sum())
    duplicate_ids = int(frame["cell_id"].duplicated().sum())
    if missing_values or duplicate_ids:
        raise ValueError(
            f"Input QA failed: missing={missing_values}, duplicate cell_id={duplicate_ids}"
        )

    rf_shap_table = pd.read_csv(RF_OUTPUT / "01_RF_SHAP_values_all_cells.csv")
    if not np.array_equal(
        frame["cell_id"].to_numpy(), rf_shap_table["cell_id"].to_numpy()
    ):
        rf_shap_table = (
            rf_shap_table.set_index("cell_id")
            .loc[frame["cell_id"]]
            .reset_index()
        )
    rf_shap = rf_shap_table[
        [f"SHAP_{feature}" for feature in rf_analysis.FEATURES]
    ].to_numpy()

    parameters = json.loads(rf_analysis.PARAMETERS.read_text(encoding="utf-8"))[
        "XGBoost"
    ]
    model = XGBRegressor(
        objective="reg:squarederror",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
        verbosity=0,
        **parameters,
    )
    x = frame[rf_analysis.FEATURES]
    y = frame[rf_analysis.TARGET].to_numpy(dtype=float)
    started = time.perf_counter()
    model.fit(x, y)
    predictions = model.predict(x)
    model.save_model(OUTPUT / "XGBoost_final_full_data_model.json")

    explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
    explanation = explainer(x, check_additivity=True)
    xgb_shap = normalize_shap(explanation.values)
    base_values = np.asarray(explanation.base_values).reshape(-1)
    if len(base_values) == 1:
        base_values = np.repeat(base_values, len(frame))
    additivity_error = float(
        np.max(np.abs(base_values + xgb_shap.sum(axis=1) - predictions))
    )

    xgb_shap_table = pd.DataFrame(
        xgb_shap,
        columns=[f"SHAP_{feature}" for feature in rf_analysis.FEATURES],
    )
    xgb_shap_table.insert(0, "predicted_R", predictions)
    xgb_shap_table.insert(0, "observed_R", y)
    xgb_shap_table.insert(0, "cell_id", frame["cell_id"])
    xgb_shap_table.to_csv(
        OUTPUT / "01_XGBoost_SHAP_values_all_cells.csv",
        index=False,
        encoding="utf-8-sig",
    )

    rf_mean = np.mean(np.abs(rf_shap), axis=0)
    xgb_mean = np.mean(np.abs(xgb_shap), axis=0)
    rf_share = rf_mean / rf_mean.sum() * 100
    xgb_share = xgb_mean / xgb_mean.sum() * 100
    rf_rank = rank_descending(rf_mean).astype(int)
    xgb_rank = rank_descending(xgb_mean).astype(int)

    xgb_importance = pd.DataFrame(
        {
            "factor": rf_analysis.FEATURES,
            "chinese_name": [rf_analysis.CHINESE[f] for f in rf_analysis.FEATURES],
            "mean_abs_shap": xgb_mean,
            "importance_share_percent": xgb_share,
            "rank": xgb_rank,
        }
    ).sort_values("rank")
    xgb_importance.to_csv(
        OUTPUT / "02_XGBoost_mean_absolute_SHAP_importance.csv",
        index=False,
        encoding="utf-8-sig",
    )

    comparison = pd.DataFrame(
        {
            "factor": rf_analysis.FEATURES,
            "chinese_name": [rf_analysis.CHINESE[f] for f in rf_analysis.FEATURES],
            "rf_mean_abs_shap": rf_mean,
            "rf_importance_share_percent": rf_share,
            "rf_rank": rf_rank,
            "xgb_mean_abs_shap": xgb_mean,
            "xgb_importance_share_percent": xgb_share,
            "xgb_rank": xgb_rank,
        }
    )
    comparison["rank_change_xgb_minus_rf"] = comparison["xgb_rank"] - comparison["rf_rank"]
    comparison["absolute_rank_difference"] = comparison["rank_change_xgb_minus_rf"].abs()
    comparison["mean_rank"] = (comparison["rf_rank"] + comparison["xgb_rank"]) / 2
    comparison.to_csv(
        OUTPUT / "03_RF_XGBoost_SHAP_ranking_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics = ranking_metrics(rf_share, xgb_share)
    bootstrap_summary, bootstrap_metrics = spatial_block_bootstrap(
        frame, rf_shap, xgb_shap
    )
    bootstrap_summary.to_csv(
        OUTPUT / "04_spatial_block_bootstrap_rank_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_metrics.to_csv(
        OUTPUT / "05_spatial_block_bootstrap_ranking_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metric_columns = [
        "spearman_rho",
        "kendall_tau_b",
        "importance_share_pearson_r",
        "top3_overlap_rate",
        "top5_overlap_rate",
        "mean_absolute_rank_difference",
    ]
    metrics["spatial_block_bootstrap"] = {
        column: {
            "median": float(bootstrap_metrics[column].median()),
            "ci95_low": float(bootstrap_metrics[column].quantile(0.025)),
            "ci95_high": float(bootstrap_metrics[column].quantile(0.975)),
        }
        for column in metric_columns
    }
    metrics["rf_top5"] = comparison.nsmallest(5, "rf_rank")["chinese_name"].tolist()
    metrics["xgb_top5"] = comparison.nsmallest(5, "xgb_rank")["chinese_name"].tolist()
    metrics["largest_rank_shifts"] = (
        comparison.nlargest(3, "absolute_rank_difference")[
            ["chinese_name", "rf_rank", "xgb_rank", "rank_change_xgb_minus_rf"]
        ].to_dict(orient="records")
    )
    (OUTPUT / "06_RF_XGBoost_SHAP_ranking_robustness_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    cv = pd.read_csv(rf_analysis.CV_SUMMARY)
    cv10 = cv[cv["block_km"].eq(10) & cv["model"].isin(["RF", "XGBoost"])]
    qa = {
        "cells": len(frame),
        "features": len(rf_analysis.FEATURES),
        "missing_required_values": missing_values,
        "duplicate_cell_id": duplicate_ids,
        "rf_shap_rows_aligned": len(rf_shap) == len(frame),
        "xgboost_version": xgboost_version,
        "shap_version": shap.__version__,
        "xgboost_full_fit_seconds_including_shap": time.perf_counter() - started,
        "xgboost_full_sample_training_r2_not_generalization": float(r2_score(y, predictions)),
        "xgboost_full_sample_training_rmse_not_generalization": float(
            np.sqrt(mean_squared_error(y, predictions))
        ),
        "xgboost_full_sample_training_mae_not_generalization": float(
            mean_absolute_error(y, predictions)
        ),
        "xgboost_maximum_shap_additivity_error": additivity_error,
        "spatial_blocks_10km": int(rf_analysis.build_blocks(frame).nunique()),
        "spatial_block_bootstraps": BOOTSTRAPS,
        "spatial_cv_10km": cv10[
            ["model", "r2_mean", "rmse_mean", "mae_mean"]
        ].to_dict(orient="records"),
    }
    (OUTPUT / "00_input_and_SHAP_QA.json").write_text(
        json.dumps(qa, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    plot_ranking_comparison(comparison, bootstrap_summary, metrics)

    top5_same = metrics["top5_overlap_count"] == 5
    conclusion = "完全一致" if top5_same else "高度重合"
    results = f"""# RF与XGBoost的SHAP排序稳健性对照

采用与全局模型比较相同的11个变量和固定XGBoost参数，在8,981个格网上拟合最终XGBoost模型并计算TreeSHAP。RF与XGBoost在10 km重复空间分块交叉验证中的平均R²分别为{cv10.loc[cv10['model'].eq('RF'), 'r2_mean'].iloc[0]:.3f}和{cv10.loc[cv10['model'].eq('XGBoost'), 'r2_mean'].iloc[0]:.3f}，预测能力接近，适合用于检验变量重要性排序对算法选择是否敏感。

全样本结果中，两种模型的Mean|SHAP|排序Spearman相关系数为{metrics['spearman_rho']:.3f}，Kendall τ为{metrics['kendall_tau_b']:.3f}，归一化重要性占比的Pearson相关系数为{metrics['importance_share_pearson_r']:.3f}。Top-3重合{metrics['top3_overlap_count']}/3，Top-5重合{metrics['top5_overlap_count']}/5，说明主导因素排序在RF与XGBoost之间{conclusion}。

为避免单个空间格网被视为相互独立，进一步以10 km空间块为重抽样单元进行{BOOTSTRAPS}次bootstrap。Spearman相关系数的bootstrap中位数为{metrics['spatial_block_bootstrap']['spearman_rho']['median']:.3f}，95%区间为{metrics['spatial_block_bootstrap']['spearman_rho']['ci95_low']:.3f}–{metrics['spatial_block_bootstrap']['spearman_rho']['ci95_high']:.3f}；Top-5重合率中位数为{metrics['spatial_block_bootstrap']['top5_overlap_rate']['median']:.2f}，95%区间为{metrics['spatial_block_bootstrap']['top5_overlap_rate']['ci95_low']:.2f}–{metrics['spatial_block_bootstrap']['top5_overlap_rate']['ci95_high']:.2f}。因此，排序稳健性结论并非由少数相邻格网主导。

图X RF与XGBoost模型的SHAP重要性排序稳健性对照。（a）归一化Mean|SHAP|贡献占比；（b）变量排序及10 km空间块bootstrap 95%区间。蓝色圆点表示RF，橙色方点表示XGBoost，灰线连接同一变量在两种模型中的排序。

注意：该对照检验的是变量重要性排序对模型算法与空间抽样的稳健性，并不证明SHAP效应具有因果性。
"""
    (OUTPUT / "07_results_and_figure_caption_zh.md").write_text(
        results, encoding="utf-8"
    )

    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Outputs: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
