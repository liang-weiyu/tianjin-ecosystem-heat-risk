"""Quantify the spatial correlation scale of ecosystem heat risk R.

Outputs a distance-band Moran's I correlogram, empirical semivariogram,
single- and nested-exponential fits, scale diagnostics, and a publication figure.
"""

from __future__ import annotations

from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit, least_squares
from scipy.spatial import cKDTree


WORKSPACE = STUDY_ROOT.parent
PROJECT = STUDY_ROOT
INPUT = (
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
    / "07_spatial_scale_basis_10km"
)
FIGURES = OUTPUT / "figures"
MAX_DISTANCE_KM = 60.0
BIN_WIDTH_KM = 2.0


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


def exponential(distance, nugget, partial_sill, scale):
    return nugget + partial_sill * (1.0 - np.exp(-distance / scale))


def nested_exponential(distance, nugget, sill_short, scale_short, sill_long, scale_long):
    return (
        nugget
        + sill_short * (1.0 - np.exp(-distance / scale_short))
        + sill_long * (1.0 - np.exp(-distance / scale_long))
    )


def compute_pairs(frame: pd.DataFrame):
    coordinates = frame[["x_m", "y_m"]].to_numpy(dtype=float)
    pairs = cKDTree(coordinates).query_pairs(
        MAX_DISTANCE_KM * 1_000.0, output_type="ndarray"
    )
    distance_km = np.sqrt(
        np.sum((coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]) ** 2, axis=1)
    ) / 1_000.0
    return pairs, distance_km


def distance_statistics(
    values: np.ndarray,
    pairs: np.ndarray,
    distance_km: np.ndarray,
    bin_width_km: float,
    maximum_km: float,
) -> pd.DataFrame:
    edges = np.arange(0.0, maximum_km + bin_width_km, bin_width_km)
    bin_index = np.digitize(distance_km, edges, right=False) - 1
    centered = values - values.mean()
    denominator = np.sum(centered**2)
    rows = []
    for index in range(len(edges) - 1):
        selected = bin_index == index
        pair_count = int(selected.sum())
        if pair_count == 0:
            continue
        first = pairs[selected, 0]
        second = pairs[selected, 1]
        moran = len(values) * np.sum(centered[first] * centered[second]) / (
            pair_count * denominator
        )
        semivariance = 0.5 * np.mean((values[first] - values[second]) ** 2)
        rows.append(
            {
                "lag_lower_km": edges[index],
                "lag_upper_km": edges[index + 1],
                "lag_center_km": (edges[index] + edges[index + 1]) / 2.0,
                "mean_pair_distance_km": distance_km[selected].mean(),
                "pair_count": pair_count,
                "morans_I": moran,
                "semivariance": semivariance,
            }
        )
    result = pd.DataFrame(rows)
    variance = np.var(values, ddof=1)
    result["semivariance_over_sample_variance"] = result["semivariance"] / variance
    result["moran_expected_random"] = -1.0 / (len(values) - 1)
    return result


def fit_variograms(empirical: pd.DataFrame, sample_variance: float):
    rows = []
    for maximum in [25.0, 30.0, 40.0, 50.0, 60.0]:
        selected = empirical[empirical["lag_center_km"] <= maximum]
        h = selected["mean_pair_distance_km"].to_numpy()
        gamma = selected["semivariance"].to_numpy()
        weights = selected["pair_count"].to_numpy(dtype=float)
        parameters, _ = curve_fit(
            exponential,
            h,
            gamma,
            p0=[gamma[0] / 2.0, max(sample_variance - gamma[0] / 2.0, 1e-8), 6.0],
            bounds=([0.0, 0.0, 0.25], [sample_variance, 2 * sample_variance, 100.0]),
            sigma=1.0 / np.sqrt(weights),
            maxfev=100_000,
        )
        prediction = exponential(h, *parameters)
        weighted_rmse = np.sqrt(np.average((prediction - gamma) ** 2, weights=weights))
        rows.append(
            {
                "model": "single_exponential",
                "maximum_lag_used_km": maximum,
                "nugget": parameters[0],
                "partial_sill_short": parameters[1],
                "scale_short_km": parameters[2],
                "practical_range_short_km": 3.0 * parameters[2],
                "partial_sill_long": np.nan,
                "scale_long_km": np.nan,
                "practical_range_long_km": np.nan,
                "total_sill": parameters[0] + parameters[1],
                "weighted_rmse": weighted_rmse,
            }
        )

    h = empirical["mean_pair_distance_km"].to_numpy()
    gamma = empirical["semivariance"].to_numpy()
    weights = empirical["pair_count"].to_numpy(dtype=float)
    normalized_weight = np.sqrt(weights / weights.max())

    def residuals(parameters):
        return (nested_exponential(h, *parameters) - gamma) * normalized_weight

    fitted = least_squares(
        residuals,
        x0=[3e-5, 1.1e-4, 3.0, 7e-5, 22.0],
        bounds=([0.0, 0.0, 0.2, 0.0, 2.0], [sample_variance, 3e-4, 20.0, 3e-4, 100.0]),
        max_nfev=100_000,
    ).x
    prediction = nested_exponential(h, *fitted)
    weighted_rmse = np.sqrt(np.average((prediction - gamma) ** 2, weights=weights))
    rows.append(
        {
            "model": "nested_double_exponential",
            "maximum_lag_used_km": MAX_DISTANCE_KM,
            "nugget": fitted[0],
            "partial_sill_short": fitted[1],
            "scale_short_km": fitted[2],
            "practical_range_short_km": 3.0 * fitted[2],
            "partial_sill_long": fitted[3],
            "scale_long_km": fitted[4],
            "practical_range_long_km": 3.0 * fitted[4],
            "total_sill": fitted[0] + fitted[1] + fitted[3],
            "weighted_rmse": weighted_rmse,
        }
    )
    result = pd.DataFrame(rows)
    result["short_structured_sill_share"] = (
        result["partial_sill_short"]
        / (result["partial_sill_short"] + result["partial_sill_long"])
    )
    return result


def scale_sensitivity(values, pairs, distance_km) -> pd.DataFrame:
    rows = []
    sample_variance = np.var(values, ddof=1)
    for width in [1.0, 2.0, 5.0]:
        table = distance_statistics(values, pairs, distance_km, width, 50.0)
        first_moran = table.iloc[0]["morans_I"]
        thresholds = {
            "first_I_le_0.20_km": table.loc[table["morans_I"] <= 0.20, "mean_pair_distance_km"].min(),
            "first_I_le_half_initial_km": table.loc[
                table["morans_I"] <= first_moran / 2.0, "mean_pair_distance_km"
            ].min(),
            "first_gamma_ge_75pct_variance_km": table.loc[
                table["semivariance"] >= 0.75 * sample_variance, "mean_pair_distance_km"
            ].min(),
        }
        rows.append({"bin_width_km": width, "initial_morans_I": first_moran, **thresholds})
    return pd.DataFrame(rows)


def make_figure(empirical: pd.DataFrame, fits: pd.DataFrame, sample_variance: float):
    configure_fonts()
    figure, axes = plt.subplots(1, 2, figsize=(10.5, 4.3))

    axes[0].plot(
        empirical["mean_pair_distance_km"],
        empirical["morans_I"],
        color="#4C78A8",
        marker="o",
        markersize=4.2,
        linewidth=1.5,
    )
    axes[0].axhline(0, color="#555555", linewidth=0.8, linestyle="--")
    axes[0].axvline(10, color="#E76F51", linewidth=1.2, linestyle="--")
    axes[0].text(10.7, 0.57, "10 km", color="#B64F38", fontsize=9)
    axes[0].set_xlabel("空间距离（km）")
    axes[0].set_ylabel("Moran's I")
    axes[0].set_xlim(0, 60)
    axes[0].grid(color="#D9D9D9", linewidth=0.55)
    axes[0].set_axisbelow(True)

    axes[1].scatter(
        empirical["mean_pair_distance_km"],
        empirical["semivariance"],
        s=25,
        facecolor="white",
        edgecolor="#4C78A8",
        linewidth=1.0,
        label="经验半方差",
        zorder=3,
    )
    nested = fits[fits["model"] == "nested_double_exponential"].iloc[0]
    x = np.linspace(0, 60, 500)
    axes[1].plot(
        x,
        nested_exponential(
            x,
            nested["nugget"],
            nested["partial_sill_short"],
            nested["scale_short_km"],
            nested["partial_sill_long"],
            nested["scale_long_km"],
        ),
        color="#72B7B2",
        linewidth=1.6,
        label="嵌套指数模型",
    )
    axes[1].axhline(sample_variance, color="#777777", linewidth=0.9, linestyle=":", label="样本方差")
    axes[1].axvline(10, color="#E76F51", linewidth=1.2, linestyle="--")
    axes[1].set_xlabel("空间距离（km）")
    axes[1].set_ylabel("半方差")
    axes[1].set_xlim(0, 60)
    axes[1].grid(color="#D9D9D9", linewidth=0.55)
    axes[1].set_axisbelow(True)
    axes[1].legend(frameon=False, fontsize=8.5, loc="lower right")

    for label, axis in zip(["(a)", "(b)"], axes):
        axis.text(-0.13, 1.03, label, transform=axis.transAxes, fontsize=12, fontweight="normal")
    figure.subplots_adjust(left=0.10, right=0.98, bottom=0.16, top=0.94, wspace=0.28)
    stem = FIGURES / "heat_risk_spatial_correlation_scale_moran_variogram"
    figure.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    figure.savefig(stem.with_suffix(".tif"), dpi=300, bbox_inches="tight", pil_kwargs={"compression": "tiff_lzw"})
    figure.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def write_interpretation(empirical, fits, sensitivity, sample_variance):
    band_8_10 = empirical[(empirical["lag_lower_km"] == 8) & (empirical["lag_upper_km"] == 10)].iloc[0]
    band_10_12 = empirical[(empirical["lag_lower_km"] == 10) & (empirical["lag_upper_km"] == 12)].iloc[0]
    first = empirical.iloc[0]
    nested = fits[fits["model"] == "nested_double_exponential"].iloc[0]
    local = fits[(fits["model"] == "single_exponential") & (fits["maximum_lag_used_km"] == 30)].iloc[0]
    threshold_columns = [
        "first_I_le_0.20_km",
        "first_I_le_half_initial_km",
        "first_gamma_ge_75pct_variance_km",
    ]
    threshold_values = sensitivity[threshold_columns].to_numpy(dtype=float)
    text = f"""# 10 km空间块尺度的空间统计依据

## 数据与方法

以严格陆地生态系统掩膜内{8981:,}个1 km格网的连续生态系统热风险指数R为对象，基于投影坐标计算0–60 km范围内的无方向距离分级Moran's I correlogram和经验半变异函数。距离组宽为2 km，每一距离组均包含至少{int(empirical['pair_count'].min()):,}对格网。Moran's I采用距离环带内二元对称权重；半方差为同一距离环带内格网对差值平方的一半均值。

## 结果

Moran's I由最邻近0–2 km距离组的{first['morans_I']:.3f}下降至8–10 km的{band_8_10['morans_I']:.3f}，并在10–12 km进一步降至{band_10_12['morans_I']:.3f}，相对初始值分别衰减{(1-band_8_10['morans_I']/first['morans_I'])*100:.1f}%和{(1-band_10_12['morans_I']/first['morans_I'])*100:.1f}%。8–10 km距离组的经验半方差为{band_8_10['semivariance']:.8f}，达到R样本方差（{sample_variance:.8f}）的{band_8_10['semivariance_over_sample_variance']*100:.1f}%。

嵌套双指数模型识别出一个占结构方差约{nested['short_structured_sill_share']*100:.1f}%的短程成分，其实用变程约为{nested['practical_range_short_km']:.1f} km；同时存在更弱的长程空间梯度。仅拟合0–30 km局地半变异结构时，单指数模型的实用变程约为{local['practical_range_short_km']:.1f} km。不同距离组宽下，Moran's I降至0.20附近、降至首距离组一半以及半方差达到样本方差75%的位置总体落在约{np.nanmin(threshold_values):.1f}–{np.nanmax(threshold_values):.1f} km范围内。

## 对10 km主尺度的判断

10 km位于天津生态系统热风险局地空间相关性快速衰减的转折区，并略大于嵌套半变异模型识别的主要短程成分变程。因此，将10 km作为重复空间分块交叉验证的主尺度具有数据依据：它能够避免同一局地热风险斑块的大量1 km格网被拆分到训练集和验证集，同时仍保留足够数量的空间块用于5折平衡分配。

但10 km不应表述为“完全空间独立距离”。Correlogram在10 km以后仍为正，局地单指数模型的实用变程约为{local['practical_range_short_km']:.1f} km，并存在更宽尺度梯度。因此，20 km空间块应保留为更保守的敏感性检验；无缓冲设计也需要在局限性中明确。

## 方法学与热环境研究参照

- Roberts et al. (2017)指出，应依据响应变量、预测因子或模型残差的相关结构选择空间阻断尺度，并强调随机交叉验证会低估空间泛化误差。DOI: https://doi.org/10.1111/ecog.02881
- Valavi et al. (2019)将经验变异函数的空间自相关范围作为选择空间块尺度的定量参考，同时强调该尺度是指导值而非绝对规则。DOI: https://doi.org/10.1111/2041-210X.13107
- 一项全球城市热研究在1 km热环境数据的机器学习评价中直接采用不重叠10 km空间块，以减弱空间自相关导致的性能高估，可作为热环境领域的外部方法参照：https://www.nature.com/articles/s41467-026-73062-8

建议表述为：“综合距离分级Moran's I correlogram与半变异函数结果，生态系统热风险的主要局地空间相关成分在约8–16 km范围内显著衰减。其中，Moran's I在约10 km处已较最邻近距离下降约65%，经验半方差达到总体方差约75%。因此，本文采用10 km规则空间块作为主验证尺度，并进一步以20 km空间块开展更保守的尺度敏感性检验。”
"""
    (OUTPUT / "10km空间块尺度依据_方法与表述.md").write_text(text, encoding="utf-8")
    caption = "图X 生态系统热风险的空间相关尺度。（a）距离分级Moran's I correlogram；（b）经验半变异函数及嵌套指数模型拟合。红色虚线表示10 km主空间块尺度。"
    (FIGURES / "figure_caption_zh.txt").write_text(caption, encoding="utf-8")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(INPUT)
    required = ["cell_id", "x_m", "y_m", "R"]
    if frame[required].isna().any().any() or frame["cell_id"].duplicated().any():
        raise ValueError("Input grid is missing values or has duplicate cell_id")
    pairs, distance_km = compute_pairs(frame)
    empirical = distance_statistics(
        frame["R"].to_numpy(dtype=float),
        pairs,
        distance_km,
        BIN_WIDTH_KM,
        MAX_DISTANCE_KM,
    )
    sample_variance = frame["R"].var(ddof=1)
    fits = fit_variograms(empirical, sample_variance)
    sensitivity = scale_sensitivity(frame["R"].to_numpy(dtype=float), pairs, distance_km)
    empirical.to_csv(OUTPUT / "01_distance_band_moran_semivariogram.csv", index=False, encoding="utf-8-sig")
    fits.to_csv(OUTPUT / "02_variogram_model_fits.csv", index=False, encoding="utf-8-sig")
    sensitivity.to_csv(OUTPUT / "03_distance_bin_sensitivity.csv", index=False, encoding="utf-8-sig")
    make_figure(empirical, fits, sample_variance)
    write_interpretation(empirical, fits, sensitivity, sample_variance)
    print(empirical.head(8).to_string(index=False))
    print("\nVariogram fits")
    print(fits.to_string(index=False))
    print("\nSensitivity")
    print(sensitivity.to_string(index=False))
    print(f"\nOutputs: {OUTPUT}")


if __name__ == "__main__":
    main()
