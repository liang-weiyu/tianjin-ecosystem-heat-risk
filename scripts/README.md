# Research scripts

These scripts preserve the original research calculations and parameter values.
Local paths and font-file references have been moved to `project_paths.py` and
`config/paths.example.toml`. They have not yet been rerun as a complete pipeline
in this repository. Full input data are not included.

| Area | Scripts |
|---|---|
| Spatial analysis | `analyze_ecosystem_heat_risk_spatial_clustering.py`, `analyze_heat_risk_spatial_correlation_scale.py`, `analyze_hevr_coastal_inland_gradient.py`, `analyze_R_equal_distance_bin_sensitivity.py` |
| Global models | `compare_global_ml_models.py`, `compare_global_ml_repeated_spatial_blocks.py` |
| Global SHAP | `analyze_rf_shap_global.py`, `compare_xgboost_rf_shap_rankings.py` |
| GWRF and combined SHAP | `run_gwrf_11f.py`, `run_gwrf_tree_stability.py`, `calculate_combined_gwrf_shap.py` |

The files share one directory to preserve their existing imports; the module
folders under `src/` are reserved for future refactoring. `tianjin_map_style.py`
is a supporting map utility.

## Paths and inputs

Copy `config/paths.example.toml` to `config/paths.local.toml`. Set `study_root`
to a working directory containing the prepared research inputs. Each script
keeps the original relative subdirectories below that directory, including
`strict_land_recalculation_20260806/08_background_driver_model/02_tables/`.
`study_root` also receives generated results. Use a working copy if you want
to preserve existing outputs. It is not created or populated automatically.

The spatial scripts also refer to `01_aligned_indices`, `02_risk_results`,
`07_coastal_inland_gradient`, and strict-land gradient tables. These are
different research stages, not a fully assembled end-to-end workflow. Inspect
the input constants at the start of the selected script to prepare its inputs.

`boundary` points to the Tianjin boundary (include companion files for a
Shapefile); `land_cover` points to the 2021 CLCD raster. `data_root` is available
for future data-preparation scripts. All paths can be changed in one local file.

The older `config.example.yaml` is a future full-pipeline design template;
these scripts currently read only the TOML path configuration. Algorithm
parameters remain explicit constants in the scripts.

## Usage

From the repository root, after installing the dependencies and preparing inputs:

```powershell
Copy-Item config/paths.example.toml config/paths.local.toml
# Edit paths.local.toml before running analyses.
python scripts/analyze_ecosystem_heat_risk_spatial_clustering.py
python scripts/compare_global_ml_repeated_spatial_blocks.py
python scripts/analyze_rf_shap_global.py
python scripts/run_gwrf_11f.py --help
```

These are separate examples, not a single complete execution recipe. Global SHAP
reads parameters from repeated spatial-block CV. GWRF reads fold assignments
and evaluation outputs from that same CV. Combined GWRF SHAP depends on the
GWRF results and global SHAP exports. Some stages take substantial time.

## Provenance

Source scripts were copied from
`risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/05_method_and_script/`
in the original workspace. The map utility came from the workspace root.
The original files remain available there. See `docs/script_inventory.md`
for the broader inventory and future refactoring plan.
