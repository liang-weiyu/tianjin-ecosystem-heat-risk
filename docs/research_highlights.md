# Research highlights

## Research setting and contribution

This study examines ecosystem heat risk in Tianjin, China, using a hazard,
exposure and vulnerability framework. The manuscript describes 8,981 valid
1 km land cells. Heat hazard represents summer 2022; ecosystem exposure and
vulnerability represent 2021 conditions.

The repository owner independently performed all data processing. This statement
does not assert sole authorship of the manuscript or assign other contributors'
roles.

## Findings reported in the manuscript

- Ecosystem heat risk is spatially clustered: global Moran's I = 0.6462,
  p < 0.001. High-risk patches are concentrated in southern and western areas
  and parts of western Binhai, rather than forming one continuous high-risk zone.
- The coastal gradient differs across H, E and V. Risk has its highest reported
  distance-band median in the 10-20 km band, rather than immediately at the coast.
- The five leading global RF attributions are surface albedo (19.00%), summer
  precipitation (18.45%), nighttime lights (18.11%), summer wind speed (15.93%)
  and population density (14.87%). Their combined 86.36% is a share of normalised
  mean absolute SHAP attribution, not causal importance or explained variance.
- Combined GWRF contributions show spatially varying attribution patterns.
  Locally dominant factors vary across northern terrain, urbanised areas and
  southern/coastal plains. Dominance is a relative attribution label, not a
  direct indicator of high risk.

## Model evaluation

The manuscript describes five-fold cross-validation using 10 km spatial blocks,
repeated ten times. The existing exported summaries support mean R² of
0.45294 for global RF, 0.44318 for local GWRF and 0.47290 for Combined GWRF.
The combined model improves mean R² by about 0.020 and reduces mean RMSE
from 0.01061 to 0.01042. This is a modest predictive improvement. No statistical
significance is asserted for the difference.

These scores concern prediction of the constructed risk index. They are not
validation against observed ecosystem damage, and model attribution does not
establish causal effects. Spatial blocks do not make evaluation fully immune
to spatial dependence, particularly where adjacent training and validation
blocks share boundaries.

## Source record

The findings were summarised from the results and discussion of the local
manuscript `word/终版论文.docx`. That document is not bundled with this repository.
The manuscript results were checked against existing summary exports, without
retraining models or recomputing any indicators.

Model result sources relative to the original research workspace:

```text
risk_space analysis/ecosystem_heat_risk_2022_lagged_2021/
  strict_land_recalculation_20260806/08_background_driver_model/
    06_global_ml_repeated_spatial_block_cv_11f/
      05_model_performance_summary_by_block_size.csv
    10_GWRF_11f/07_GWRF_performance_summary.csv
```

The selected images are described in `outputs/figures/README.md`. They have
been copied without editing, so their Chinese labels are retained alongside
English descriptions. Current manuscript figure numbers and original image
filenames are not identical.
