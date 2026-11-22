# Selected research figures

The four PNGs are unmodified copies of existing study figures. Filenames were
changed to English for portable Markdown links. The original Chinese labels,
panel numbering, legends and pixel values are retained.

| Repository filename | Original figure filename | Purpose |
|---|---|---|
| `heat_risk_spatial_patterns.png` | 图1 热危害、生态系统暴露度、生态系统脆弱性与生态系统热风险空间分布.png | H/E/V/R spatial distributions |
| `heat_risk_spatial_clustering.png` | 图3 生态系统热风险空间集聚特征.png | LISA clusters and global Moran scatter |
| `global_rf_shap.png` | 图5 生态系统热风险驱动因素的全局SHAP重要性与作用方向.png | Global RF attribution |
| `combined_gwrf_local_contributions.png` | 图8 Combined GWRF各驱动因素局地贡献及局地主导影响因素空间分布.png | Signed contributions and locally dominant factors |

Original directory relative to the research workspace:

```text
risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/
  strict_land_recalculation_20260806/小论文结果与分析_正文图件/
```

Original filename numbering differs from the current manuscript numbering.
Use the subject and panel labels when referring to these figures.

## English legend guide

Spatial distributions, panels (a-d): heat hazard, ecosystem exposure, ecosystem
vulnerability and ecosystem heat risk. Classes range from low (低), relatively
low (较低), medium (中等), relatively high (较高) to high (高). The classes apply
separately to each mapped index; they are not shared numerical thresholds.

LISA legend: non-significant (不显著), high-high (高-高), low-low (低-低),
high-low (高-低), low-high (低-高), and isolated cells (孤立像元).

Global SHAP rows and Combined GWRF panels (a-k), in order:

1. Surface albedo (地表反照率)
2. Summer precipitation (夏季累计降水)
3. Nighttime lights (夜间灯光强度)
4. Summer wind speed (夏季平均风速)
5. Population density (人口密度)
6. Elevation (地表高程)
7. Road density (路网密度)
8. Building coverage (建筑覆盖率)
9. Sky view factor (天空可视因子)
10. Building-height standard deviation (建筑高度离散度)
11. Mean building height (平均建筑高度)

In the global beeswarm, pink indicates higher feature values and green lower
values. Horizontal position indicates signed SHAP contribution. In the local
contribution maps, red indicates positive and blue negative combined SHAP.
Panel (l) shows the locally dominant factor (局地主导影响因素), defined by
the largest absolute combined contribution. Each contribution panel has its
own colour scale; saturation should not be directly compared across panels.

