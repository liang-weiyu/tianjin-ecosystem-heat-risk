"""Compare four global regression models for ecosystem heat risk.

The primary assessment uses nested spatial cross-validation. Coordinates are
used only to construct geographically contiguous folds and are never supplied
to a model as predictors.
"""

from __future__ import annotations

import json
from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path
import time

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


WORKSPACE = STUDY_ROOT.parent
PROJECT = STUDY_ROOT
INPUT_TABLE = (
    PROJECT
    / "strict_land_recalculation_20260806"
    / "08_background_driver_model"
    / "02_tables"
    / "background_driver_model_table_13_factors.csv"
)
OUTPUT = (
    PROJECT
    / "strict_land_recalculation_20260806"
    / "08_background_driver_model"
    / "05_global_ml_comparison_11f"
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
RANDOM_SEED = 42
OUTER_FOLDS = 5
INNER_FOLDS = 3
SPATIAL_BLOCK_M = 20_000.0


def configure_fonts() -> None:
    """Use Times New Roman for Latin text and SimSun for Chinese text."""

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


def build_models() -> dict[str, tuple[object, dict[str, list], int]]:
    """Return estimators, search spaces, and randomized-search budgets."""

    scaled_steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    mlr = Pipeline([*scaled_steps, ("model", LinearRegression())])
    svr = TransformedTargetRegressor(
        regressor=Pipeline(
            [
                *scaled_steps,
                (
                    "model",
                    SVR(kernel="rbf", cache_size=2_000, max_iter=250_000),
                ),
            ]
        ),
        transformer=StandardScaler(),
    )
    rf = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                RandomForestRegressor(
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                    criterion="squared_error",
                ),
            ),
        ]
    )
    xgb = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=RANDOM_SEED,
                    n_jobs=1,
                    tree_method="hist",
                    verbosity=0,
                ),
            ),
        ]
    )
    return {
        "MLR": (mlr, {}, 0),
        "SVR": (
            svr,
            {
                "regressor__model__C": [1, 10, 100],
                "regressor__model__gamma": ["scale", 0.03],
                "regressor__model__epsilon": [0.03, 0.1],
            },
            6,
        ),
        "RF": (
            rf,
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [None, 10, 20, 30],
                "model__min_samples_leaf": [1, 2, 4, 8],
                "model__max_features": [0.5, 0.8, 1.0],
            },
            8,
        ),
        "XGBoost": (
            xgb,
            {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.02, 0.05, 0.1],
                "model__subsample": [0.7, 0.85, 1.0],
                "model__colsample_bytree": [0.7, 0.85, 1.0],
                "model__min_child_weight": [1, 3, 5],
                "model__reg_lambda": [1, 3, 10],
            },
            10,
        ),
    }


def make_spatial_folds(frame: pd.DataFrame) -> np.ndarray:
    """Create five contiguous outer folds using coordinate clustering."""

    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype="float64")
    coordinates = StandardScaler().fit_transform(coordinates)
    raw_labels = KMeans(
        n_clusters=OUTER_FOLDS,
        random_state=RANDOM_SEED,
        n_init=50,
    ).fit_predict(coordinates)

    centroids = (
        frame.assign(raw_fold=raw_labels)
        .groupby("raw_fold")[["x_m", "y_m"]]
        .mean()
        .sort_values(["y_m", "x_m"], ascending=[False, True])
    )
    relabel = {old: new for new, old in enumerate(centroids.index, start=1)}
    return np.array([relabel[label] for label in raw_labels], dtype="int16")


def make_inner_groups(frame: pd.DataFrame) -> np.ndarray:
    """Assign 20 km spatial blocks for grouped inner cross-validation."""

    x_index = np.floor((frame["x_m"] - frame["x_m"].min()) / SPATIAL_BLOCK_M)
    y_index = np.floor((frame["y_m"] - frame["y_m"].min()) / SPATIAL_BLOCK_M)
    return (x_index.astype(int).astype(str) + "_" + y_index.astype(int).astype(str)).to_numpy()


def evaluate_predictions(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Calculate the common regression metrics."""

    rmse = float(np.sqrt(mean_squared_error(observed, predicted)))
    return {
        "r2": float(r2_score(observed, predicted)),
        "rmse": rmse,
        "mae": float(mean_absolute_error(observed, predicted)),
        "nrmse_percent": float(rmse / np.mean(observed) * 100.0),
        "pearson_r": float(pearsonr(observed, predicted).statistic),
    }


def fit_nested_spatial_cv(frame: pd.DataFrame):
    """Tune in spatial blocks and predict held-out contiguous outer regions."""

    models = build_models()
    x = frame[FEATURES]
    y = frame[TARGET].to_numpy(dtype="float64")
    outer_fold = frame["spatial_fold"].to_numpy()
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    parameter_rows: list[dict] = []

    for model_name in MODEL_ORDER:
        base_estimator, parameter_space, search_budget = models[model_name]
        for fold in range(1, OUTER_FOLDS + 1):
            started = time.perf_counter()
            train_mask = outer_fold != fold
            test_mask = outer_fold == fold
            x_train, x_test = x.loc[train_mask], x.loc[test_mask]
            y_train, y_test = y[train_mask], y[test_mask]
            estimator = clone(base_estimator)

            if parameter_space:
                train_frame = frame.loc[train_mask]
                inner_groups = make_inner_groups(train_frame)
                unique_groups = np.unique(inner_groups)
                inner_splits = min(INNER_FOLDS, len(unique_groups))
                if inner_splits < 2:
                    raise RuntimeError("Insufficient spatial blocks for inner CV")
                search = RandomizedSearchCV(
                    estimator=estimator,
                    param_distributions=parameter_space,
                    n_iter=search_budget,
                    scoring="neg_root_mean_squared_error",
                    cv=GroupKFold(n_splits=inner_splits),
                    random_state=RANDOM_SEED + fold,
                    n_jobs=-1,
                    refit=True,
                    verbose=0,
                    return_train_score=False,
                )
                search.fit(x_train, y_train, groups=inner_groups)
                fitted = search.best_estimator_
                best_parameters = search.best_params_
                inner_rmse = float(-search.best_score_)
            else:
                fitted = estimator.fit(x_train, y_train)
                best_parameters = {}
                inner_rmse = np.nan

            predictions = fitted.predict(x_test)
            metrics = evaluate_predictions(y_test, predictions)
            elapsed = time.perf_counter() - started
            metric_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()),
                    **metrics,
                    "elapsed_seconds": elapsed,
                }
            )
            parameter_rows.append(
                {
                    "model": model_name,
                    "fold": fold,
                    "inner_cv_best_rmse": inner_rmse,
                    "best_parameters": json.dumps(
                        best_parameters, ensure_ascii=False, sort_keys=True
                    ),
                }
            )
            prediction_rows.append(
                pd.DataFrame(
                    {
                        "cell_id": frame.loc[test_mask, "cell_id"].to_numpy(),
                        "x_m": frame.loc[test_mask, "x_m"].to_numpy(),
                        "y_m": frame.loc[test_mask, "y_m"].to_numpy(),
                        "spatial_fold": fold,
                        "model": model_name,
                        "observed_R": y_test,
                        "predicted_R": predictions,
                        "residual": y_test - predictions,
                    }
                )
            )
            print(
                f"{model_name} fold={fold}: R2={metrics['r2']:.3f}, "
                f"RMSE={metrics['rmse']:.5f}, elapsed={elapsed:.1f}s",
                flush=True,
            )

    return (
        pd.concat(prediction_rows, ignore_index=True),
        pd.DataFrame(metric_rows),
        pd.DataFrame(parameter_rows),
    )


def fit_random_holdout(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run a conventional 80/20 random split as an optimism diagnostic."""

    train_index, test_index = train_test_split(
        np.arange(len(frame)), test_size=0.20, random_state=RANDOM_SEED
    )
    x_train = frame.iloc[train_index][FEATURES]
    x_test = frame.iloc[test_index][FEATURES]
    y_train = frame.iloc[train_index][TARGET].to_numpy(dtype="float64")
    y_test = frame.iloc[test_index][TARGET].to_numpy(dtype="float64")
    rows = []
    prediction_rows = []
    for model_name in MODEL_ORDER:
        estimator, parameter_space, search_budget = build_models()[model_name]
        started = time.perf_counter()
        if parameter_space:
            search = RandomizedSearchCV(
                estimator=estimator,
                param_distributions=parameter_space,
                n_iter=search_budget,
                scoring="neg_root_mean_squared_error",
                cv=KFold(n_splits=3, shuffle=True, random_state=RANDOM_SEED),
                random_state=RANDOM_SEED,
                n_jobs=-1,
                refit=True,
                verbose=0,
            )
            search.fit(x_train, y_train)
            fitted = search.best_estimator_
            parameters = search.best_params_
        else:
            fitted = estimator.fit(x_train, y_train)
            parameters = {}
        predicted = fitted.predict(x_test)
        metrics = evaluate_predictions(y_test, predicted)
        rows.append(
            {
                "model": model_name,
                "n_train": len(train_index),
                "n_test": len(test_index),
                **metrics,
                "elapsed_seconds": time.perf_counter() - started,
                "best_parameters": json.dumps(
                    parameters, ensure_ascii=False, sort_keys=True
                ),
            }
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "cell_id": frame.iloc[test_index]["cell_id"].to_numpy(),
                    "model": model_name,
                    "observed_R": y_test,
                    "predicted_R": predicted,
                    "residual": y_test - predicted,
                }
            )
        )
        print(
            f"random {model_name}: R2={metrics['r2']:.3f}, "
            f"RMSE={metrics['rmse']:.5f}",
            flush=True,
        )
    return pd.DataFrame(rows), pd.concat(prediction_rows, ignore_index=True)


def summarize_metrics(predictions: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Summarize fold variability and pooled out-of-fold performance."""

    rows = []
    for model_name in MODEL_ORDER:
        model_folds = folds[folds["model"] == model_name]
        model_predictions = predictions[predictions["model"] == model_name]
        pooled = evaluate_predictions(
            model_predictions["observed_R"].to_numpy(),
            model_predictions["predicted_R"].to_numpy(),
        )
        rows.append(
            {
                "model": model_name,
                "fold_mean_r2": model_folds["r2"].mean(),
                "fold_sd_r2": model_folds["r2"].std(ddof=1),
                "fold_mean_rmse": model_folds["rmse"].mean(),
                "fold_sd_rmse": model_folds["rmse"].std(ddof=1),
                "fold_mean_mae": model_folds["mae"].mean(),
                "fold_sd_mae": model_folds["mae"].std(ddof=1),
                "pooled_oof_r2": pooled["r2"],
                "pooled_oof_rmse": pooled["rmse"],
                "pooled_oof_mae": pooled["mae"],
                "pooled_oof_nrmse_percent": pooled["nrmse_percent"],
                "pooled_oof_pearson_r": pooled["pearson_r"],
                "total_elapsed_seconds": model_folds["elapsed_seconds"].sum(),
            }
        )
    summary = pd.DataFrame(rows)
    summary["rank_by_spatial_cv"] = summary["fold_mean_rmse"].rank(
        method="min", ascending=True
    ).astype(int)
    return summary.sort_values("rank_by_spatial_cv")


def plot_model_performance(summary: pd.DataFrame) -> None:
    """Plot pooled spatial out-of-fold R2 and error metrics."""

    configure_fonts()
    ordered = summary.set_index("model").loc[MODEL_ORDER].reset_index()
    colors = [MODEL_COLORS[name] for name in MODEL_ORDER]
    positions = np.arange(len(MODEL_ORDER))
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))

    r2_values = ordered["pooled_oof_r2"].to_numpy()
    axes[0].bar(
        positions,
        np.maximum(r2_values, -0.45),
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=0.6,
    )
    axes[0].axhline(0, color="#555555", linewidth=0.8)
    upper_r2 = max(0.38, float(np.nanmax(r2_values)) + 0.05)
    axes[0].set_ylim(-0.50, upper_r2)
    axes[0].set_xticks(positions, MODEL_ORDER)
    axes[0].set_ylabel("空间交叉验证合并外层预测 R²")
    axes[0].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[0].set_axisbelow(True)
    for position, value in zip(positions, r2_values):
        if value < -0.50:
            axes[0].annotate(
                f"{value:.3f}",
                xy=(position, -0.47),
                xytext=(position, -0.37),
                ha="center",
                va="bottom",
                fontsize=9,
                arrowprops={"arrowstyle": "-|>", "color": "#333333", "lw": 0.8},
            )
            continue
        axes[0].text(
            position,
            value + (0.025 if value >= 0 else -0.025),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    width = 0.34
    axes[1].bar(
        positions - width / 2,
        ordered["pooled_oof_rmse"],
        width,
        color=colors,
        edgecolor="#4A4A4A",
        linewidth=0.6,
        label="RMSE",
    )
    axes[1].bar(
        positions + width / 2,
        ordered["pooled_oof_mae"],
        width,
        facecolor="white",
        edgecolor=colors,
        linewidth=1.6,
        label="MAE",
    )
    axes[1].set_xticks(positions, MODEL_ORDER)
    axes[1].set_ylabel("预测误差")
    axes[1].set_yscale("log")
    axes[1].set_ylim(0.0065, 0.10)
    axes[1].grid(axis="y", color="#D9D9D9", linewidth=0.6)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, loc="upper right")
    for position, rmse, mae in zip(
        positions, ordered["pooled_oof_rmse"], ordered["pooled_oof_mae"]
    ):
        axes[1].text(position - width / 2, rmse * 1.06, f"{rmse:.4f}", ha="center", fontsize=8)
        axes[1].text(position + width / 2, mae * 0.94, f"{mae:.4f}", ha="center", va="top", fontsize=8)

    axes[0].text(-0.12, 1.03, "(a)", transform=axes[0].transAxes, fontsize=12)
    axes[1].text(-0.12, 1.03, "(b)", transform=axes[1].transAxes, fontsize=12)
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.14, top=0.95, wspace=0.27)
    stem = FIGURES / "global_model_performance_spatial_cv"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tif"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def plot_observed_predicted(predictions: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Plot out-of-fold observed versus predicted values for all models."""

    configure_fonts()
    observed_min = predictions["observed_R"].min()
    observed_max = predictions["observed_R"].max()
    x_padding = (observed_max - observed_min) * 0.04
    x_limits = (observed_min - x_padding, observed_max + x_padding)
    nonlinear = predictions[predictions["model"] != "MLR"]
    nonlinear_max = max(observed_max, nonlinear["predicted_R"].max())
    nonlinear_min = min(observed_min, nonlinear["predicted_R"].min())
    y_padding = (nonlinear_max - nonlinear_min) * 0.04
    nonlinear_y_limits = (nonlinear_min - y_padding, nonlinear_max + y_padding)
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 8.2), sharex=True, sharey=False)
    for panel, model_name in zip(axes.ravel(), MODEL_ORDER):
        data = predictions[predictions["model"] == model_name]
        metrics = summary.set_index("model").loc[model_name]
        density = panel.hexbin(
            data["observed_R"],
            data["predicted_R"],
            gridsize=48,
            mincnt=1,
            cmap="Blues",
            linewidths=0,
        )
        panel.set_xlim(x_limits)
        if model_name == "MLR":
            model_y_min = min(observed_min, data["predicted_R"].min())
            model_y_max = max(observed_max, data["predicted_R"].max())
            model_padding = (model_y_max - model_y_min) * 0.04
            y_limits = (model_y_min - model_padding, model_y_max + model_padding)
        else:
            y_limits = nonlinear_y_limits
        panel.set_ylim(y_limits)
        identity_min = max(x_limits[0], y_limits[0])
        identity_max = min(x_limits[1], y_limits[1])
        panel.plot(
            [identity_min, identity_max],
            [identity_min, identity_max],
            color="#D95F4C",
            linewidth=1.0,
            linestyle="--",
        )
        panel.set_title(model_name, fontsize=11, pad=5)
        panel.text(
            0.04,
            0.94,
            f"$R^2$ = {metrics['pooled_oof_r2']:.3f}\n"
            f"RMSE = {metrics['pooled_oof_rmse']:.4f}\n"
            f"MAE = {metrics['pooled_oof_mae']:.4f}",
            transform=panel.transAxes,
            ha="left",
            va="top",
            fontsize=9,
        )
        panel.grid(color="#E1E1E1", linewidth=0.45)
        panel.set_axisbelow(True)
        density.set_rasterized(True)

    for panel in axes[-1, :]:
        panel.set_xlabel("观测生态系统热风险指数")
    for panel in axes[:, 0]:
        panel.set_ylabel("预测生态系统热风险指数")
    for label, panel in zip(["(a)", "(b)", "(c)", "(d)"], axes.ravel()):
        panel.text(-0.13, 1.04, label, transform=panel.transAxes, fontsize=12)
    figure.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.97, hspace=0.20, wspace=0.16)
    stem = FIGURES / "global_models_observed_vs_predicted_spatial_oof"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(
        stem.with_suffix(".tif"),
        dpi=300,
        bbox_inches="tight",
        pil_kwargs={"compression": "tiff_lzw"},
    )
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_method_note(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    random_metrics: pd.DataFrame | None = None,
) -> None:
    """Write a concise Chinese methods and results note."""

    by_fold_rmse = summary.sort_values("fold_mean_rmse")
    by_pooled = summary.sort_values("pooled_oof_rmse")
    best_fold = by_fold_rmse.iloc[0]
    best_pooled = by_pooled.iloc[0]
    random_text = ""
    if random_metrics is not None:
        random_ordered = random_metrics.sort_values("rmse")
        random_best = random_ordered.iloc[0]
        random_text = f"""

作为对照，常规80%/20%随机划分下，{random_best['model']}的精度最高（R²={random_best['r2']:.3f}，RMSE={random_best['rmse']:.5f}），但四种模型的随机划分结果均明显优于空间验证，说明随机混合相邻网格会产生空间信息泄漏并高估泛化能力。因此，本文以嵌套空间交叉验证作为模型选择依据，随机划分仅作为补充对照。
"""
    note = f"""# 全局机器学习模型比较

## 数据与验证设计

- 样本：严格陆地生态系统掩膜内的1 km网格，共{len(frame):,}个。
- 因变量：生态系统热风险指数（R）。
- 自变量：平均地表高程、多年夏季平均风速、2021年夏季平均地表反照率、建筑覆盖率、平均建筑高度、建筑高度离散度、天空可视因子、路网密度、夜间灯光强度、人口密度和多年平均夏季累计降水。根据Pearson相关与VIF诊断，删除距海岸线距离和主导风向加权迎风面积指数；剩余11变量最大VIF约为3.96。
- 模型：多元线性回归（MLR）、支持向量回归（SVR）、随机森林（RF）和XGBoost。
- 主评价：5折嵌套空间交叉验证。外层以坐标聚类形成5个连续空间验证区；内层以20 km空间块进行3折分组交叉验证并最小化RMSE。坐标不作为模型自变量。
- 指标：R²越高越好，RMSE和MAE越低越好。表中fold mean为5个空间验证区指标的算术平均，pooled OOF为合并全部外层预测后的总体指标。

## 初步结果

按5个空间验证区的平均RMSE排序，{best_fold['model']}最低（{best_fold['fold_mean_rmse']:.5f}）；按合并全部外层预测评价，{best_pooled['model']}同样最优（R²={best_pooled['pooled_oof_r2']:.3f}，RMSE={best_pooled['pooled_oof_rmse']:.5f}，MAE={best_pooled['pooled_oof_mae']:.5f}）。SVR也是唯一在5个连续空间验证区均取得正R²的模型，说明其空间泛化稳定性相对更好，因此推荐作为后续全局解释模型。XGBoost和RF分列第二、第三，但二者各有验证区R²低于0。MLR在北部外推验证区出现异常预测，合并外层预测R²为{summary.set_index('model').loc['MLR', 'pooled_oof_r2']:.3f}，说明线性模型难以刻画当前变量—风险关系。模型比较评价的是空间样本外预测能力；后续仍需检查最终SVR残差的空间自相关，并采用适用于非线性模型的解释方法。
{random_text}

## 解释限制

该比较评价的是样本外空间预测能力，不构成因果效应识别。空间折之间的环境范围不同，单折R²可能为负；这表示模型在相应未见区域的预测不如直接使用该区域观测均值，并不等同于模型计算错误。
"""
    (OUTPUT / "全局机器学习模型比较_方法与初步结果.md").write_text(note, encoding="utf-8")


def main() -> None:
    """Run the complete global model comparison workflow."""

    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT_TABLE)
    required = ["cell_id", "x_m", "y_m", TARGET, *FEATURES]
    missing_columns = [column for column in required if column not in frame.columns]
    if missing_columns:
        raise KeyError(f"Missing columns: {missing_columns}")
    if frame[required].isna().any().any():
        raise ValueError("Input contains missing values in required columns")
    if frame["cell_id"].duplicated().any():
        raise ValueError("cell_id is not unique")

    frame = frame.copy()
    frame["spatial_fold"] = make_spatial_folds(frame)
    frame[["cell_id", "x_m", "y_m", "spatial_fold"]].to_csv(
        OUTPUT / "01_spatial_outer_folds.csv", index=False, encoding="utf-8-sig"
    )
    fold_summary = (
        frame.groupby("spatial_fold")
        .agg(
            n=("cell_id", "size"),
            x_min=("x_m", "min"),
            x_max=("x_m", "max"),
            y_min=("y_m", "min"),
            y_max=("y_m", "max"),
            R_mean=(TARGET, "mean"),
            R_sd=(TARGET, "std"),
        )
        .reset_index()
    )
    fold_summary.to_csv(
        OUTPUT / "02_spatial_fold_summary.csv", index=False, encoding="utf-8-sig"
    )

    predictions, fold_metrics, parameters = fit_nested_spatial_cv(frame)
    summary = summarize_metrics(predictions, fold_metrics)
    predictions.to_csv(
        OUTPUT / "03_spatial_oof_predictions.csv", index=False, encoding="utf-8-sig"
    )
    fold_metrics.to_csv(
        OUTPUT / "04_fold_metrics.csv", index=False, encoding="utf-8-sig"
    )
    summary.to_csv(
        OUTPUT / "05_model_performance_summary.csv", index=False, encoding="utf-8-sig"
    )
    parameters.to_csv(
        OUTPUT / "06_best_parameters_by_fold.csv", index=False, encoding="utf-8-sig"
    )
    random_metrics, random_predictions = fit_random_holdout(frame)
    random_metrics.to_csv(
        OUTPUT / "07_random_holdout_metrics.csv", index=False, encoding="utf-8-sig"
    )
    random_predictions.to_csv(
        OUTPUT / "08_random_holdout_predictions.csv", index=False, encoding="utf-8-sig"
    )
    plot_model_performance(summary)
    plot_observed_predicted(predictions, summary)
    combined = summary[
        ["model", "pooled_oof_r2", "pooled_oof_rmse", "pooled_oof_mae"]
    ].merge(
        random_metrics[["model", "r2", "rmse", "mae"]],
        on="model",
        how="left",
    ).rename(
        columns={
            "r2": "random_holdout_r2",
            "rmse": "random_holdout_rmse",
            "mae": "random_holdout_mae",
        }
    )
    combined.to_csv(
        OUTPUT / "09_random_vs_spatial_validation.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_method_note(frame, summary, random_metrics)
    print(summary.to_string(index=False), flush=True)
    print(f"Outputs: {OUTPUT}", flush=True)


if __name__ == "__main__":
    main()
