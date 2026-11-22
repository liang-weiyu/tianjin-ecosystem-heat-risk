# 现有脚本迁移与模块归类清单

本清单基于 `G:\riskdddd` 在 2026-09-21 的现有文件。主线候选主要来自：

```text
risk_space analysis/
  ecosystem_heat_risk_2022_lagged_2021/
    05_method_and_script/
```

下列路径均相对于原始工作区。归类清单中的目标模块名称是未来重构建议。

2026-10-19：为 PhD 申请展示，已将核心 11 个脚本复制至 `scripts/`，保留原文件名和
互相导入关系，统一了本机路径和字体配置。原始工作区文件未修改。当前没有执行完整
数据复现；详见 `scripts/README.md`。下面尚未迁移的支撑和旧版脚本仍仅作候选记录。

## 迁移原则

- **核心**：形成论文主要方法或结果，优先迁移并重构。
- **支撑**：用于变量筛选、敏感性、稳健性或图件表达，核心流程稳定后迁移。
- **旧版候选**：保留作方法对照，不应与当前主线同时作为默认入口。
- 迁移时统一移除本机盘符、系统字体路径和 `sys.path.insert`。
- 一个分析步骤只保留一个正式入口；绘图与计算尽量拆分。

## 1. 空间分析模块

目标目录：`src/tianjin_heat_risk/spatial_analysis/`

### 核心脚本

| 现有脚本 | 建议新名称 | 作用 |
|---|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_ecosystem_heat_risk_spatial_clustering.py` | `moran_lisa.py` | 1 km Queen 邻接下的全局 Moran's I、局部 Moran/LISA、置换检验和 FDR 校正 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_heat_risk_spatial_correlation_scale.py` | `correlation_scale.py` | 距离带 Moran、半变异函数及空间相关尺度估计 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_hevr_coastal_inland_gradient.py` | `coastal_gradient.py` | H、E、V、R 的滨海—内陆梯度分析 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_R_equal_distance_bin_sensitivity.py` | `coastal_gradient_sensitivity.py` | 10/20 km 等距分箱及空间块 Bootstrap 敏感性分析 |

### 支撑脚本

| 现有脚本 | 归类理由 |
|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_ecosystem_heat_risk_coastal_inland_gradient.py` | 早期/替代的风险滨海梯度实现，迁移前与 `analyze_hevr_coastal_inland_gradient.py` 比对合并 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_wind_coastal_gradient_sensitivity.py` | 风速—滨海梯度敏感性分析 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_HEVR_spatial_patterns_and_coastal_gradients.py` | 空间格局与梯度组合图，应迁移到绘图子模块 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_ecosystem_heat_risk_natural_breaks.py` | 风险自然断点制图 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_ecosystem_heat_risk_type_zoning.py` | 风险类型分区制图 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_four_governance_type_zoning.py` | 治理类型分区制图 |

### 旧版候选

| 现有脚本 | 处理建议 |
|---|---|
| `calculate_global_moran_hev_r.py` | 与当前严格陆地主线结果核对后归档，不作为默认入口 |
| `calculate_lisa_risk.py` | 与当前 LISA/FDR 实现核对后归档 |
| `calculate_coastal_inland_gradient.py` | 与当前 HEVR 梯度实现核对后归档 |

## 2. 全局随机森林模块

目标目录：`src/tianjin_heat_risk/global_random_forest/`

### 核心脚本

| 现有脚本 | 建议新名称 | 作用 |
|---|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/compare_global_ml_models.py` | `model_comparison.py` | 使用嵌套空间交叉验证比较 RF、XGBoost、SVR、线性模型 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/compare_global_ml_repeated_spatial_blocks.py` | `repeated_spatial_cv.py` | 5/10/20 km 空间块重复评估，形成全局模型主要预测证据 |

### 支撑脚本

| 现有脚本 | 建议新位置/用途 |
|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_influence_factor_pearson_vif.py` | `diagnostics/pearson_vif.py`，进入模型前的共线性诊断 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_background_13factor_pearson_vif.py` | 与上一个脚本比较后确定 11/13 因子主线，避免重复发布 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_pearson_correlation_vif_figure.py` | `plotting/plot_pearson_vif.py` |

## 3. SHAP 模块

目标目录：`src/tianjin_heat_risk/shap_analysis/`

### 核心脚本

| 现有脚本 | 建议新名称 | 作用 |
|---|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/analyze_rf_shap_global.py` | `global_rf_shap.py` | 全局 RF 的 SHAP 重要性、方向、阈值和交互分析 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/compare_xgboost_rf_shap_rankings.py` | `ranking_robustness.py` | RF 与 XGBoost TreeSHAP 排序及空间块 Bootstrap 稳健性比较 |

### 支撑与绘图脚本

| 现有脚本 | 建议新位置/用途 |
|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/combine_rf_shap_green_panels.py` | `plotting/combine_global_shap_panels.py` |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_rf_shap_green_pink_variants.py` | 仅保留论文最终配色版本 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/refresh_rf_shap_labels_and_palette.py` | 将标签与配色并入统一绘图配置，不保留独立修改脚本 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_core_variable_interaction_figure10.py` | 全局 RF 核心变量交互机制图 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/decompose_core_variable_mechanisms.py` | SHAP 核心变量效应/交互机制分解 |

### 旧版候选

| 现有脚本 | 处理建议 |
|---|---|
| `run_gwrf_shap_analysis.py` | 同时包含旧版 GWRF/SHAP 流程，待与当前主线结果核对后拆分或归档 |
| `run_extended_gwrf_shap_analysis.py` | 扩展旧版流程，默认不进入公开主线 |
| `plot_gwrf_global_local_shap_combined.py` | 旧版组合图，和当前 Combined GWRF SHAP 图件比较后择一 |
| `plot_extended_gwrf_shap_world_mercator.py` | 旧版扩展制图脚本，默认归档 |

## 4. GWRF 模块

目标目录：`src/tianjin_heat_risk/gwrf/`

### 核心脚本

| 现有脚本 | 建议新名称 | 作用 |
|---|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/run_gwrf_11f.py` | `model.py` 与 `run.py` | 自适应 KNN、bisquare 权重、带宽选择、空间块评估和最终逐像元局部 RF |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/run_gwrf_tree_stability.py` | `tree_stability.py` | 固定带宽和混合权重下的树数稳定性检验 |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/calculate_combined_gwrf_shap.py` | `combined_shap.py` | 按全局/局地预测混合权重计算 Combined GWRF SHAP 并制图 |

### 支撑脚本

| 现有脚本 | 建议新位置/用途 |
|---|---|
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_coastal_10_20km_local_drivers.py` | `plotting/coastal_local_drivers.py` |
| `risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/plot_cross_coastal_distance_band_comparison.py` | `plotting/distance_band_comparison.py` |
| `risk_space analysis/analyze_10_20km_combined_shap.py` | 组合 SHAP 的滨海距离带汇总；需先确认其输入对应当前 GWRF 版本 |

### 旧版验证脚本

| 现有脚本 | 处理建议 |
|---|---|
| `validate_gwrf_shap_analysis.py` | 提取仍适用的断言，改造成 `tests/` 下的测试 |
| `validate_extended_gwrf_shap_analysis.py` | 与当前 11 因子流程核对后提取测试 |
| `finalize_extended_gwrf_outputs.py` | 检查是否只是旧版结果整理，通常不作为方法入口 |
| `create_gwrf_shap_notebook.py` | 不发布“生成 Notebook 的脚本”；保留整理后的最终 Notebook 即可 |
| `create_extended_gwrf_shap_notebook.py` | 同上 |

## 建议迁移顺序

1. 先迁移模型表生成与公共 I/O 依赖，但不复制任何原始数据。
2. 迁移空间分析核心 4 个脚本，并验证 CRS、邻接关系和 FDR 结果。
3. 迁移全局模型比较及重复空间块评估，固定主模型参数。
4. 迁移全局 SHAP，确认训练数据、特征顺序和保存模型完全一致。
5. 拆分 `run_gwrf_11f.py`，使模型实现、命令入口和绘图相互独立。
6. 最后迁移 Combined GWRF SHAP 和论文专用绘图脚本。

