# Workflows

当前展示版本直接使用 `scripts/` 下的研究脚本；此处入口属于未来可选重构。

后续在此提供四个薄入口：

```text
run_spatial_analysis.py
run_global_random_forest.py
run_shap_analysis.py
run_gwrf.py
```

入口只负责读取配置和调度；可测试的计算逻辑放在 `src/tianjin_heat_risk/`。

