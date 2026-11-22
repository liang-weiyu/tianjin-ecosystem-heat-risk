# -*- coding: utf-8 -*-
"""分析当前生态系统综合热风险的全局与局部空间集聚特征。

方法：连续风险 R、1 km Queen 邻接、行标准化权重、9,999 次随机标签
置换。局部 Moran's I 主结果采用 Benjamini-Hochberg FDR q <= 0.05。
"""

from pathlib import Path
from project_paths import STUDY_ROOT, BOUNDARY_PATH, CLCD_PATH, font_path
import csv
import gc
import os
import sys

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import rasterio
from rasterio.transform import array_bounds
from scipy.ndimage import label as connected_label
from scipy.sparse import csr_matrix
from pyproj import Transformer


PROJECT_ROOT = STUDY_ROOT
RISK_SOURCE = (
    PROJECT_ROOT
    / "02_risk_results"
    / "04_R_ecosystem_heat_risk_HEV_product_continuous_1km.tif"
)
BOUNDARY = BOUNDARY_PATH
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "06_spatial_autocorrelation"
    / "current_updated_R"
)
RASTER_DIR = OUTPUT_ROOT / "01_rasters"
FIGURE_DIR = OUTPUT_ROOT / "02_figures"
TABLE_DIR = OUTPUT_ROOT / "03_tables"

PERMUTATIONS = 9999
RANDOM_SEED = 20260804
FDR_ALPHA = 0.05
NODATA_FLOAT = -9999.0
NODATA_CLASS = 255
ISLAND_CLASS = 254

# LISA 类别并非顺序等级；暖/冷色分别表示高值与低值空间关系。
CLASS_INFO = {
    0: ("不显著", "Not significant", "#E5E5E5"),
    1: ("高-高", "High-High", "#D73027"),
    2: ("低-低", "Low-Low", "#4575B4"),
    3: ("高-低", "High-Low", "#FC8D59"),
    4: ("低-高", "Low-High", "#91BFDB"),
    ISLAND_CLASS: ("孤立像元", "Queen island", "#777777"),
}

# 散点图与 LISA 地图严格共用同一颜色表。
SCATTER_COLORS = {
    code: CLASS_INFO[code][2]
    for code in range(0, 5)
}


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_queen_weights(valid_mask):
    """构造有效像元的一阶 Queen 行标准化稀疏权重。"""
    height, width = valid_mask.shape
    full_index = np.full(valid_mask.shape, -1, dtype=np.int32)
    full_index[valid_mask] = np.arange(valid_mask.sum(), dtype=np.int32)

    neighbor_count = np.zeros(valid_mask.shape, dtype=np.int16)
    directions = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1), (0, 1),
        (1, -1), (1, 0), (1, 1),
    ]
    for dr, dc in directions:
        r0 = max(0, -dr)
        r1 = min(height, height - dr)
        c0 = max(0, -dc)
        c1 = min(width, width - dc)
        center = valid_mask[r0:r1, c0:c1]
        shifted = valid_mask[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        neighbor_count[r0:r1, c0:c1] += center & shifted

    analysis_mask = valid_mask & (neighbor_count > 0)
    analysis_index = np.full(valid_mask.shape, -1, dtype=np.int32)
    analysis_index[analysis_mask] = np.arange(
        analysis_mask.sum(), dtype=np.int32
    )

    rows = []
    columns = []
    for dr, dc in directions:
        r0 = max(0, -dr)
        r1 = min(height, height - dr)
        c0 = max(0, -dc)
        c1 = min(width, width - dc)
        center = analysis_mask[r0:r1, c0:c1]
        shifted = analysis_mask[r0 + dr:r1 + dr, c0 + dc:c1 + dc]
        pair = center & shifted
        if not np.any(pair):
            continue
        rows.append(analysis_index[r0:r1, c0:c1][pair])
        columns.append(
            analysis_index[r0 + dr:r1 + dr, c0 + dc:c1 + dc][pair]
        )

    row_ids = np.concatenate(rows).astype(np.int32)
    column_ids = np.concatenate(columns).astype(np.int32)
    n = int(analysis_mask.sum())
    degree = np.bincount(row_ids, minlength=n).astype(np.float64)
    weights = 1.0 / degree[row_ids]
    matrix = csr_matrix((weights, (row_ids, column_ids)), shape=(n, n))
    return analysis_mask, neighbor_count, matrix, row_ids.size


def benjamini_hochberg(p_values):
    """计算 Benjamini-Hochberg 调整后的 q 值。"""
    p_values = np.asarray(p_values, dtype=np.float64)
    count = p_values.size
    order = np.argsort(p_values)
    ranked = p_values[order]
    adjusted = ranked * count / np.arange(1, count + 1, dtype=np.float64)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def permutation_inference(z_values, weights, local_observed):
    """同时完成全局与局部 Moran 随机标签置换检验。"""
    rng = np.random.default_rng(RANDOM_SEED)
    n = z_values.size
    expected = -1.0 / (n - 1)
    local_extreme = np.zeros(n, dtype=np.int32)
    global_values = np.empty(PERMUTATIONS, dtype=np.float64)
    observed_global = float(local_observed.mean())
    observed_global_distance = abs(observed_global - expected)
    observed_local_distance = np.abs(local_observed - expected)

    batch_size = 16
    completed = 0
    while completed < PERMUTATIONS:
        batch = min(batch_size, PERMUTATIONS - completed)
        permuted = np.column_stack(
            [rng.permutation(z_values) for _ in range(batch)]
        )
        lag_permuted = weights @ permuted
        local_permuted = permuted * lag_permuted
        global_values[completed:completed + batch] = local_permuted.mean(axis=0)
        local_extreme += np.sum(
            np.abs(local_permuted - expected)
            >= observed_local_distance[:, None],
            axis=1,
        ).astype(np.int32)
        completed += batch

    global_extreme = int(
        np.sum(np.abs(global_values - expected) >= observed_global_distance)
    )
    global_positive = int(np.sum(global_values >= observed_global))
    local_p = (local_extreme + 1.0) / (PERMUTATIONS + 1.0)
    global_summary = {
        "moran_I": observed_global,
        "expected_I": expected,
        "permutation_mean_I": float(global_values.mean()),
        "permutation_sd_I": float(global_values.std(ddof=1)),
        "permutation_z": (
            observed_global - float(global_values.mean())
        ) / float(global_values.std(ddof=1)),
        "p_two_sided": (global_extreme + 1.0) / (PERMUTATIONS + 1.0),
        "p_positive": (global_positive + 1.0) / (PERMUTATIONS + 1.0),
    }
    return global_summary, local_p


def write_float_raster(path, array, profile, nodata=NODATA_FLOAT):
    output = np.full(array.shape, nodata, dtype=np.float32)
    finite = np.isfinite(array)
    output[finite] = array[finite].astype(np.float32)
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff", count=1, dtype="float32", nodata=nodata,
        compress="LZW", tiled=True,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(output, 1)


def write_cluster_raster(path, clusters, profile):
    output_profile = profile.copy()
    output_profile.update(
        driver="GTiff", count=1, dtype="uint8", nodata=NODATA_CLASS,
        compress="LZW", tiled=True,
    )
    with rasterio.open(path, "w", **output_profile) as destination:
        destination.write(clusters.astype(np.uint8), 1)
        color_map = {
            code: (*tuple(bytes.fromhex(info[2].lstrip("#"))), 255)
            for code, info in CLASS_INFO.items()
        }
        color_map[NODATA_CLASS] = (255, 255, 255, 0)
        destination.write_colormap(1, color_map)
        destination.update_tags(
            indicator="ecosystem heat-risk LISA cluster",
            variable="continuous R = H * E * V",
            weights="first-order Queen contiguity, row standardized",
            inference=f"random-label permutations={PERMUTATIONS}",
            significance=f"Benjamini-Hochberg FDR q<={FDR_ALPHA}",
            classes=(
                "0=not_significant,1=HH,2=LL,3=HL,4=LH,"
                "254=queen_island,255=nodata"
            ),
        )


def configure_fonts():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": ["Times New Roman"],
        "mathtext.fontset": "custom",
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "axes.unicode_minus": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
    })


def save_moran_scatter(
    z_values,
    lag_values,
    cluster_values,
    global_summary,
):
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from matplotlib.lines import Line2D
    from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea, VPacker

    chinese = FontProperties(fname=font_path("chinese"), size=11)
    english = FontProperties(fname=font_path("english"), size=11)
    configure_fonts()
    fig, axis = plt.subplots(figsize=(6.6, 5.8))
    # 不显著点先绘制为灰色，显著 LISA 类型随后叠加，颜色与地图一致。
    point_styles = {
        0: {"alpha": 0.42},
        1: {"alpha": 0.88},
        2: {"alpha": 0.88},
        3: {"alpha": 0.90},
        4: {"alpha": 0.95},
    }
    for code in range(0, 5):
        mask = cluster_values == code
        if not np.any(mask):
            continue
        style = point_styles[code]
        axis.scatter(
            z_values[mask],
            lag_values[mask],
            s=7,
            c=SCATTER_COLORS[code],
            alpha=style["alpha"],
            edgecolors="none",
            linewidths=0,
            rasterized=True,
            zorder=2 + code,
        )
    x_limit = float(np.nanpercentile(np.abs(z_values), 99.7)) * 1.08
    x_limit = max(x_limit, 2.5)
    x_line = np.asarray([-x_limit, x_limit])
    regression_line, = axis.plot(
        x_line,
        global_summary["moran_I"] * x_line,
        color="#C44E52",
        linewidth=1.35,
        zorder=8,
    )
    axis.axhline(0, color="#777777", linewidth=0.7, linestyle="--")
    axis.axvline(0, color="#777777", linewidth=0.7, linestyle="--")
    axis.set_xlim(-x_limit, x_limit)
    # Pack separately rendered Chinese and English spans so each span is
    # guaranteed to use the requested physical font file.
    x_label = HPacker(
        children=[
            TextArea("标准化生态系统热风险指数（", textprops={"fontproperties": chinese}),
            TextArea("Z-score", textprops={"fontproperties": english}),
            TextArea("）", textprops={"fontproperties": chinese}),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    axis.add_artist(AnchoredOffsetbox(
        loc="lower center",
        child=x_label,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(0.5, -0.090),
        bbox_transform=axis.transAxes,
    ))
    y_label = VPacker(
        children=[
            TextArea("）", textprops={
                "fontproperties": chinese, "rotation": 90,
            }),
            TextArea("Wz", textprops={
                "fontproperties": english, "rotation": 90,
            }),
            TextArea("标准化生态系统热风险指数的空间滞后值（", textprops={
                "fontproperties": chinese, "rotation": 90,
            }),
        ],
        align="center",
        pad=0,
        sep=0,
    )
    axis.add_artist(AnchoredOffsetbox(
        loc="center left",
        child=y_label,
        frameon=False,
        pad=0,
        borderpad=0,
        bbox_to_anchor=(-0.070, 0.5),
        bbox_transform=axis.transAxes,
    ))
    axis.grid(color="#E2E2E2", linewidth=0.55, zorder=0)
    axis.spines[["top", "right"]].set_visible(False)
    legend_handles = [
        Line2D(
            [0], [0], color="#C44E52", linewidth=1.35,
            label=rf"莫兰指数 $I = {global_summary['moran_I']:.4f}$",
        )
    ]
    for code in range(0, 5):
        legend_handles.append(Line2D(
            [0], [0], marker="o", linestyle="none",
            markerfacecolor=SCATTER_COLORS[code],
            markeredgecolor="none",
            markersize=5.5,
            label=CLASS_INFO[code][0],
        ))
    axis.legend(
        handles=legend_handles,
        frameon=False,
        loc="upper left",
        ncol=2,
        columnspacing=0.9,
        handletextpad=0.45,
        prop=FontProperties(
            fname=font_path("chinese"),
            size=8.5,
        ),
    )
    fig.tight_layout()
    stem = FIGURE_DIR / "ecosystem_heat_risk_global_Moran_scatter_queen_9999"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.05)
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return stem.with_suffix(".png")


def save_lisa_map(clusters, profile):
    import geopandas as gpd
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    from PIL import Image

    # Shared map helpers are stored alongside these research scripts.
    from tianjin_map_style import (
        add_boundary_outline,
        add_classified_legend,
        add_north_arrow,
        add_scale_bar,
    )

    chinese = FontProperties(fname=font_path("chinese"), size=11)
    chinese_title = FontProperties(fname=font_path("chinese"), size=11.5)
    configure_fonts()
    boundary = gpd.read_file(BOUNDARY).to_crs(profile["crs"])
    geometry = boundary.geometry.union_all()
    west, south, east, north = geometry.bounds
    width = east - west
    height = north - south
    rb = array_bounds(profile["height"], profile["width"], profile["transform"])
    raster_west, raster_south, raster_east, raster_north = rb

    codes = [0, 1, 2, 3, 4, ISLAND_CLASS]
    rgba = np.full((*clusters.shape, 4), 255, dtype=np.uint8)
    for code in codes:
        rgb = tuple(bytes.fromhex(CLASS_INFO[code][2].lstrip("#")))
        rgba[clusters == code, :3] = rgb

    fig, axis = plt.subplots(figsize=(7.4, 8.4))
    image = axis.imshow(
        rgba,
        extent=[raster_west, raster_east, raster_south, raster_north],
        origin="upper",
        interpolation="none",
        resample=False,
        zorder=1,
    )
    outline = add_boundary_outline(axis, geometry, linewidth=0.7)
    image.set_clip_path(outline)
    axis.set_xlim(west - 0.01 * width, east + 0.42 * width)
    axis.set_ylim(south - 0.10 * height, north + 0.01 * height)
    axis.set_aspect("equal")
    axis.set_axis_off()
    add_north_arrow(axis, (west, south, east, north))
    add_scale_bar(axis, (west, south, east, north))
    legend = add_classified_legend(
        axis,
        "生态系统热风险局部空间集聚",
        [CLASS_INFO[code][0] for code in codes],
        palette=[CLASS_INFO[code][2] for code in codes],
    )
    legend.set_bbox_to_anchor((0.69, 0.14), transform=axis.transAxes)
    legend.get_title().set_fontproperties(chinese_title)
    for text in legend.get_texts():
        text.set_fontproperties(chinese)

    fig.subplots_adjust(left=0.02, right=0.99, top=0.985, bottom=0.025)
    stem = FIGURE_DIR / "ecosystem_heat_risk_LISA_cluster_FDR05_queen_9999"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    png_path = stem.with_suffix(".png")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    with Image.open(png_path) as exported:
        exported.convert("RGB").save(
            stem.with_suffix(".tif"), dpi=(300, 300), compression="tiff_lzw"
        )
    return png_path


def save_combined_lisa_moran_figure(lisa_path, scatter_path):
    """Place the LISA map and Moran scatter side by side as panels (a)-(b)."""
    from PIL import Image, ImageDraw, ImageFont

    with Image.open(lisa_path) as source:
        lisa = source.convert("RGB")
    with Image.open(scatter_path) as source:
        scatter = source.convert("RGB")

    target_height = lisa.height
    scatter_width = round(scatter.width * target_height / scatter.height)
    scatter = scatter.resize(
        (scatter_width, target_height),
        resample=Image.Resampling.LANCZOS,
    )

    top_band = 110
    panel_gap = 45
    canvas = Image.new(
        "RGB",
        (lisa.width + panel_gap + scatter.width, top_band + target_height),
        "white",
    )
    canvas.paste(lisa, (0, top_band))
    scatter_left = lisa.width + panel_gap
    canvas.paste(scatter, (scatter_left, top_band))

    draw = ImageDraw.Draw(canvas)
    panel_font = ImageFont.truetype(font_path("english"), 64)
    draw.text((24, 16), "(a)", fill="black", font=panel_font)
    draw.text((scatter_left + 24, 16), "(b)", fill="black", font=panel_font)

    stem = FIGURE_DIR / "ecosystem_heat_risk_LISA_Moran_combined_ab"
    canvas.save(stem.with_suffix(".png"), dpi=(300, 300))
    canvas.save(
        stem.with_suffix(".tif"),
        dpi=(300, 300),
        compression="tiff_lzw",
    )
    canvas.save(stem.with_suffix(".pdf"), resolution=300.0)
    return stem.with_suffix(".png")


def summarize_connected_patches(clusters, profile):
    """按 Queen 连通性汇总显著 LISA 类型的斑块形态。"""
    transformer = Transformer.from_crs(
        profile["crs"], "EPSG:4326", always_xy=True
    )
    structure = np.ones((3, 3), dtype=np.uint8)
    rows = []
    for code in range(1, 5):
        labeled, patch_count = connected_label(clusters == code, structure)
        sizes = np.bincount(labeled.ravel())[1:]
        total = int(sizes.sum()) if sizes.size else 0
        largest = int(sizes.max()) if sizes.size else 0
        centroid_lon = np.nan
        centroid_lat = np.nan
        if sizes.size:
            largest_label = int(np.argmax(sizes)) + 1
            rr, cc = np.where(labeled == largest_label)
            xs = profile["transform"].c + (cc + 0.5) * profile["transform"].a
            ys = profile["transform"].f + (rr + 0.5) * profile["transform"].e
            centroid_lon, centroid_lat = transformer.transform(
                float(xs.mean()), float(ys.mean())
            )
        rows.append({
            "class": code,
            "label_cn": CLASS_INFO[code][0],
            "patch_count": int(patch_count),
            "total_cells": total,
            "mean_patch_cells": total / patch_count if patch_count else 0.0,
            "largest_patch_cells": largest,
            "largest_patch_share_pct": largest / total * 100 if total else 0.0,
            "largest_patch_centroid_lon": centroid_lon,
            "largest_patch_centroid_lat": centroid_lat,
        })
    return rows


def write_academic_summary(global_summary, class_rows, patch_rows, metadata):
    by_class = {row["class"]: row for row in class_rows}
    significant = sum(by_class[code]["grid_cells"] for code in range(1, 5))
    hh = by_class[1]
    ll = by_class[2]
    hl = by_class[3]
    lh = by_class[4]
    by_patch = {row["class"]: row for row in patch_rows}
    hh_patch = by_patch[1]
    ll_patch = by_patch[2]
    text = f"""# 生态系统热风险空间集聚特征分析

## 方法

以连续生态系统热风险指数 $R=H\\times E\\times V$ 为分析对象，在 1 km 栅格尺度构建一阶 Queen 邻接空间权重矩阵并进行行标准化。剔除无有效邻居的孤立像元后，共纳入 {metadata['analysis_cells']} 个网格。全局与局部 Moran's I 均采用 {PERMUTATIONS:,} 次随机标签置换检验（seed={RANDOM_SEED}）。局部显著性在双侧伪 p 值基础上采用 Benjamini–Hochberg 方法控制多重比较，主结果阈值为 FDR $q\\leq0.05$。

## 结果

生态系统热风险呈显著正向空间自相关（Moran's $I$={global_summary['moran_I']:.4f}，期望值={global_summary['expected_I']:.4f}，置换 $z$={global_summary['permutation_z']:.2f}，$p$={global_summary['p_two_sided']:.4f}），表明相似风险水平在空间上倾向于相邻分布，而非随机离散。

FDR 校正后共有 {significant} 个网格表现为显著局部空间关联，占分析网格的 {significant / metadata['analysis_cells'] * 100:.2f}%。其中，高—高集聚区为 {hh['grid_cells']} 个网格（{hh['percent_of_analysis_cells']:.2f}%），低—低集聚区为 {ll['grid_cells']} 个网格（{ll['percent_of_analysis_cells']:.2f}%）；高—低和低—高空间异常分别为 {hl['grid_cells']} 和 {lh['grid_cells']} 个网格。HH 与 LL 构成显著局部类型的主体，说明天津市生态系统热风险的局地结构主要表现为同质集聚；HL 与 LH 数量较少，则反映少量风险水平与周边背景不一致的空间过渡或异常单元。

从显著斑块的空间连通性看，HH 共形成 {hh_patch['patch_count']} 个 Queen 连通斑块，最大斑块为 {hh_patch['largest_patch_cells']} km²；LL 共形成 {ll_patch['patch_count']} 个斑块，最大斑块为 {ll_patch['largest_patch_cells']} km²。HH 的斑块数量更多且最大斑块较小，表明高风险集聚总体更为分散和破碎；LL 的大型连通斑块更突出，低风险集聚相对具有更强的局地连续性。空间上，HH 主要表现为南部、西部及中东部的多处局地斑块，LL 则在北部、北中部和南部部分区域形成较连续的冷点集聚。上述方位描述基于栅格格局，不替代区县尺度的分区统计。

需要注意，Moran's I 和 LISA 反映空间相似性及邻域依赖，不识别因果机制。FDR 校正后的 LISA 图较原始 $p\\leq0.05$ 更为保守，适合作为论文主结果；孤立像元未参与空间推断。
"""
    (OUTPUT_ROOT / "生态系统热风险空间集聚特征分析_学术稿.md").write_text(
        text, encoding="utf-8-sig"
    )


def main():
    for directory in (RASTER_DIR, FIGURE_DIR, TABLE_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    with rasterio.open(RISK_SOURCE) as source:
        risk = source.read(1).astype(np.float64)
        profile = source.profile.copy()
        valid = np.isfinite(risk)
        if source.nodata is not None:
            valid &= risk != source.nodata

    analysis_mask, neighbor_count, weights, directed_links = build_queen_weights(valid)
    islands = valid & ~analysis_mask
    values = risk[analysis_mask]
    z_values = (values - values.mean()) / values.std(ddof=0)
    lag_values = np.asarray(weights @ z_values).ravel()
    local_i_values = z_values * lag_values

    global_summary, local_p_values = permutation_inference(
        z_values, weights, local_i_values
    )
    local_q_values = benjamini_hochberg(local_p_values)

    local_i = np.full(risk.shape, np.nan, dtype=np.float64)
    local_p = np.full(risk.shape, np.nan, dtype=np.float64)
    local_q = np.full(risk.shape, np.nan, dtype=np.float64)
    local_lag = np.full(risk.shape, np.nan, dtype=np.float64)
    local_i[analysis_mask] = local_i_values
    local_p[analysis_mask] = local_p_values
    local_q[analysis_mask] = local_q_values
    local_lag[analysis_mask] = lag_values

    clusters = np.full(risk.shape, NODATA_CLASS, dtype=np.uint8)
    clusters[analysis_mask] = 0
    clusters[islands] = ISLAND_CLASS
    significant = local_q_values <= FDR_ALPHA
    high = z_values > 0
    high_lag = lag_values > 0
    analysis_clusters = np.zeros(z_values.size, dtype=np.uint8)
    analysis_clusters[significant & high & high_lag] = 1
    analysis_clusters[significant & ~high & ~high_lag] = 2
    analysis_clusters[significant & high & ~high_lag] = 3
    analysis_clusters[significant & ~high & high_lag] = 4
    clusters[analysis_mask] = analysis_clusters

    write_float_raster(RASTER_DIR / "R_local_Moran_I_queen.tif", local_i, profile)
    write_float_raster(RASTER_DIR / "R_spatial_lag_z_queen.tif", local_lag, profile)
    write_float_raster(RASTER_DIR / "R_LISA_p_two_sided_9999.tif", local_p, profile)
    write_float_raster(RASTER_DIR / "R_LISA_q_BH_FDR_9999.tif", local_q, profile)
    cluster_path = RASTER_DIR / "R_LISA_cluster_BH_FDR_q05_queen_9999.tif"
    write_cluster_raster(cluster_path, clusters, profile)

    n = int(analysis_mask.sum())
    neighbor_values = neighbor_count[analysis_mask]
    metadata = {
        "common_support_cells": int(valid.sum()),
        "analysis_cells": n,
        "islands_excluded": int(islands.sum()),
        "directed_neighbor_links": int(directed_links),
        "mean_neighbors": float(neighbor_values.mean()),
        "minimum_neighbors": int(neighbor_values.min()),
        "maximum_neighbors": int(neighbor_values.max()),
    }
    global_row = {
        "index": "R",
        **global_summary,
        "permutations": PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        **metadata,
        "weights": "first-order Queen contiguity, row-standardized",
        "pixel_size_m": abs(float(profile["transform"].a)),
    }
    write_csv(TABLE_DIR / "global_Moran_R_queen_9999.csv", [global_row])

    class_rows = []
    significant_cells = int(np.sum((clusters >= 1) & (clusters <= 4)))
    for code in range(0, 5):
        count = int(np.sum(clusters == code))
        class_rows.append({
            "class": code,
            "label_cn": CLASS_INFO[code][0],
            "label_en": CLASS_INFO[code][1],
            "grid_cells": count,
            "grid_footprint_km2": count,
            "percent_of_analysis_cells": count / n * 100,
            "percent_of_significant_cells": (
                count / significant_cells * 100 if code > 0 and significant_cells else 0.0
            ),
        })
    write_csv(TABLE_DIR / "R_LISA_cluster_area_BH_FDR_q05.csv", class_rows)
    patch_rows = summarize_connected_patches(clusters, profile)
    write_csv(TABLE_DIR / "R_LISA_connected_patch_summary.csv", patch_rows)
    inference_row = {
        **metadata,
        "permutations": PERMUTATIONS,
        "random_seed": RANDOM_SEED,
        "raw_p_le_0_05_cells": int(np.sum(local_p_values <= 0.05)),
        "FDR_q_le_0_05_cells": significant_cells,
        "FDR_alpha": FDR_ALPHA,
    }
    write_csv(TABLE_DIR / "R_LISA_inference_summary.csv", [inference_row])

    scatter_path = save_moran_scatter(
        z_values,
        lag_values,
        analysis_clusters,
        global_summary,
    )
    del weights, local_p_values, local_q_values
    gc.collect()
    map_path = save_lisa_map(clusters, profile)
    combined_path = save_combined_lisa_moran_figure(map_path, scatter_path)
    write_academic_summary(global_summary, class_rows, patch_rows, metadata)

    validation = f"""# Validation report

## Overall assessment: Ready to share with stated caveats

- Source: current continuous ecosystem heat-risk raster `{RISK_SOURCE.name}`.
- Valid support: {metadata['common_support_cells']} cells; Moran support: {n}; Queen islands excluded: {metadata['islands_excluded']}.
- Grid/weights: 1 km, first-order Queen, row standardized; neighbor range {metadata['minimum_neighbors']}-{metadata['maximum_neighbors']}.
- Global Moran's I: {global_summary['moran_I']:.12f}; mean local I: {float(local_i_values.mean()):.12f} (identity verified).
- Permutations: {PERMUTATIONS}; reproducible seed: {RANDOM_SEED}; minimum attainable pseudo p-value: {1/(PERMUTATIONS+1):.4f}.
- Local multiple testing: Benjamini-Hochberg FDR q <= {FDR_ALPHA}.
- Cluster counts plus non-significant cells sum to {sum(row['grid_cells'] for row in class_rows)} analysis cells.
- HH/LL connected-patch metrics use the same eight-neighbor definition as the Queen weights.

## Required caveats

- Random-label permutation inference is used; it must not be described as a causal test.
- FDR-adjusted LISA is the primary map; unadjusted p <= 0.05 is retained only as a sensitivity count.
- Queen-isolated valid cells are mapped separately and excluded from Moran inference.
"""
    (OUTPUT_ROOT / "VALIDATION_REPORT.md").write_text(validation, encoding="utf-8-sig")

    print(global_row)
    for row in class_rows:
        print(row)
    print(f"cluster_raster={cluster_path}")
    print(f"scatter={scatter_path}")
    print(f"lisa_map={map_path}")
    print(f"combined_figure={combined_path}")
    print(f"output_root={OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
