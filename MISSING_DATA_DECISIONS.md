# Missing-data analysis and preprocessing — Ames House Prices

Scope: this document covers only missing-data treatment. `SalePrice` remains
untouched; no transformations, feature engineering, exploratory analysis, or
model fitting are performed. Counts are from the supplied raw files (1,460
training rows and 1,459 test rows). A dash means the field has no missing
values in that split.

## Decision rules

`data_description.txt` explicitly defines `NA` as a feature absence for alley,
basement descriptors, fireplace quality, garage descriptors, pool quality,
fence, and miscellaneous feature. The preprocessing script only assigns an
absence label where the accompanying physical measure supports that conclusion
(for example, `TotalBsmtSF == 0` for a basement). A blank paired with a
positive area is instead marked `Missing`, preserving the data-quality signal.

For genuinely unknown numerical values, medians are learned from **training
predictors only**, then applied unchanged to both files: by `Neighborhood` for
`LotFrontage` and `MasVnrArea`, by `GarageType` for garage measurements, then
with a training-wide median fallback. No imputation statistic uses `SalePrice`
or is fit on the test set.

## Complete missing-value inventory and treatment

| Variable | Train: n (%) | Test: n (%) | Meaning of NA | Treatment | Rationale |
|---|---:|---:|---|---|---|
| `MSZoning` | — | 4 (0.27%) | Unknown | `Missing` category | Dictionary gives zoning classes but no absence meaning. |
| `LotFrontage` | 259 (17.74%) | 227 (15.56%) | Unknown street-frontage length | Neighborhood median; train-wide fallback | Dictionary defines linear feet, not an absence code; local lot context is preferable to zero. |
| `Alley` | 1,369 (93.77%) | 1,352 (92.67%) | Meaningful absence: no alley access | `NoAlley` | Dictionary explicitly: NA = no alley access. |
| `Utilities` | — | 2 (0.14%) | Unknown | `Missing` category | Dictionary lists utility-service levels, not an NA absence state. |
| `Exterior1st` | — | 1 (0.07%) | Unknown exterior covering | `Missing` category | A house must have exterior covering; no NA meaning in dictionary. |
| `Exterior2nd` | — | 1 (0.07%) | Unknown exterior covering | `Missing` category | Same record-level issue as `Exterior1st`; preserve unknown. |
| `MasVnrType` | 872 (59.73%) | 894 (61.27%) | Mostly meaningful absence; a few unknowns | `NoMasonryVeneer` if area is 0; otherwise `Missing` | Dictionary includes `None`; the output label avoids CSV readers treating literal `None` as missing. 859/872 train and 876/894 test blanks have zero area. |
| `MasVnrArea` | 8 (0.55%) | 15 (1.03%) | Unknown veneer area | Neighborhood median; train-wide fallback | Area is numeric and the blank records are not demonstrated to have no veneer. |
| `BsmtQual` | 37 (2.53%) | 44 (3.02%) | Usually no basement; some unknown | `NoBasement` when total area is 0 (or both quality/total blank); otherwise `Missing` | Dictionary: NA = no basement. Three test blanks have positive area, so cannot be absence. |
| `BsmtCond` | 37 (2.53%) | 45 (3.08%) | Usually no basement; some unknown | Conditional `NoBasement`, else `Missing` | Dictionary: NA = no basement; condition prevents overwriting inconsistent records. |
| `BsmtExposure` | 38 (2.60%) | 44 (3.02%) | Usually no basement; some unknown | Conditional `NoBasement`, else `Missing` | Dictionary: NA = no basement. |
| `BsmtFinType1` | 37 (2.53%) | 42 (2.88%) | Usually no basement; some unknown | Conditional `NoBasement`, else `Missing` | Dictionary: NA = no basement. |
| `BsmtFinSF1` | — | 1 (0.07%) | No basement in this record | 0 when basement absent; otherwise train median | The sole blank belongs to the all-basement-fields-blank record. |
| `BsmtFinType2` | 38 (2.60%) | 42 (2.88%) | Usually no basement; some unknown | Conditional `NoBasement`, else `Missing` | Dictionary: NA = no basement. |
| `BsmtFinSF2` | — | 1 (0.07%) | No basement in this record | 0 when basement absent; otherwise train median | Same all-basement-fields-blank record. |
| `BsmtUnfSF` | — | 1 (0.07%) | No basement in this record | 0 when basement absent; otherwise train median | Same all-basement-fields-blank record. |
| `TotalBsmtSF` | — | 1 (0.07%) | No basement in this record | 0 when basement absent; otherwise train median | Missing alongside `BsmtQual`; structural absence is supported. |
| `Electrical` | 1 (0.07%) | — | Unknown system | `Missing` category | Dictionary lists systems, without an NA absence definition. |
| `BsmtFullBath` | — | 2 (0.14%) | No basement in both records | 0 when basement absent; otherwise train median | Both occur with no-basement evidence; conditional rule remains safe for future data. |
| `BsmtHalfBath` | — | 2 (0.14%) | No basement in both records | 0 when basement absent; otherwise train median | Same evidence as `BsmtFullBath`. |
| `KitchenQual` | — | 1 (0.07%) | Unknown quality | `Missing` category | A kitchen exists; dictionary offers no NA absence state. |
| `Functional` | — | 2 (0.14%) | Unknown functionality | `Typ` | The dictionary explicitly says “Assume typical unless deductions are warranted.” |
| `FireplaceQu` | 690 (47.26%) | 730 (50.03%) | Meaningful absence: no fireplace | `NoFireplace` | Dictionary explicitly: NA = no fireplace; every blank coincides with `Fireplaces == 0`. |
| `GarageType` | 81 (5.55%) | 76 (5.21%) | Meaningful absence: no garage | `NoGarage` if area is 0; otherwise `Missing` | Dictionary: NA = no garage; all blanks in this field align with zero garage area. |
| `GarageYrBlt` | 81 (5.55%) | 78 (5.35%) | No garage or unknown year | 0 if no garage; otherwise garage-type median/fallback | A year of zero is used only for proven structural absence. |
| `GarageFinish` | 81 (5.55%) | 78 (5.35%) | No garage or unknown finish | `NoGarage` if absent; otherwise `Missing` | Dictionary: NA = no garage; two test records have a garage but blank attributes. |
| `GarageCars` | — | 1 (0.07%) | Unknown capacity | Garage-type median; train-wide fallback | It occurs with an existing detached garage and blank area—not no garage. |
| `GarageArea` | — | 1 (0.07%) | Unknown size | Garage-type median; train-wide fallback | It occurs with an existing detached garage and blank capacity—not no garage. |
| `GarageQual` | 81 (5.55%) | 78 (5.35%) | No garage or unknown quality | `NoGarage` if absent; otherwise `Missing` | Dictionary: NA = no garage; conditional rule retains inconsistent blanks as unknown. |
| `GarageCond` | 81 (5.55%) | 78 (5.35%) | No garage or unknown condition | `NoGarage` if absent; otherwise `Missing` | Dictionary: NA = no garage; conditional rule retains inconsistent blanks as unknown. |
| `PoolQC` | 1,453 (99.52%) | 1,456 (99.79%) | Usually no pool; a few unknowns | `NoPool` if `PoolArea == 0`; otherwise `Missing` | Dictionary: NA = no pool; 3 test blanks have positive pool area and remain unknown. |
| `Fence` | 1,179 (80.75%) | 1,169 (80.12%) | Meaningful absence: no fence | `NoFence` | Dictionary explicitly: NA = no fence. |
| `MiscFeature` | 1,406 (96.30%) | 1,408 (96.50%) | Meaningful absence: none | `NoMiscFeature` | Dictionary explicitly: NA = none; this output label survives normal CSV parsing. |
| `SaleType` | — | 1 (0.07%) | Unknown sale type | `Missing` category | Dictionary gives sale-type codes, not an NA absence state. |

## Reproducible output

Run `python3 src/preprocess_missing_data.py` from the repository root. It fits
on `train.csv`, writes `data/processed/train_missing_processed.csv` and
`data/processed/test_missing_processed.csv`, and fails if either result retains
an NA. It does not modify `train.csv` or `test.csv`.

## Items for team review

1. The supplied data contain unexpectedly high `MasVnrType` missingness. The
   conditional `NoMasonryVeneer`/`Missing` split is intentional; changing all blanks to a
   single value would hide observed inconsistencies.
2. This stage preserves all source columns and does not drop sparse fields
   such as `PoolQC` or `MiscFeature`. The team can decide later—during feature
   selection—whether to retain them in a model.
