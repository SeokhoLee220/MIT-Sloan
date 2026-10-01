# Pass Option Availability (POA) and Effective Pass Option Score (EPOS)

Code and summary results accompanying the abstract *"EPOS: Quantifying Effective Passing Options in Soccer"*
submitted to the MIT Sloan Sports Analytics Conference research paper competition (SSAC27).

**What this repository does**

1. Builds, from the raw PFF FC event + tracking data of the 2022 FIFA World Cup, one feature row per
   *(ball carrier, potential passing option)* of each analysed pass: the pass length plus six spatial/dynamic variables for
   each of the 11 opposing defenders (**67 predictors**).
2. Trains a classifier on the observed pass outcome (completed / failed). The predicted completion probability is
   **POA** (Pass Option Availability) of that option.
3. Sums POA over all eligible options at the moment of the pass to obtain **EPOS** (Effective Pass Option Score) and
   tests its relation to pass completion and backward-pass selection with logistic regression
   (Table 1 of the abstract; `scripts/07_make_table1.py` renders it from `report/epos_modelA_results.json`).

## Headline results

Training: 45 matches (25,269 passes). Validation: 19 matches (10,708 passes), split at match level a priori.

| Item | Result |
|---|---|
| HGB, leave-one-match-out CV on training matches | AUC 0.877 |
| HGB frozen, applied once to validation matches | AUC 0.885 (95% CI 0.869–0.898); Brier 0.073 (constant predictor 0.098) |
| XGBoost / random forest (training CV) | AUC 0.876 / 0.876 (paired differences vs HGB within 0.002, CI includes 0) |
| EPOS → pass completion (validation, per +1 EPOS) | OR 1.80 (95% CI 1.70–1.90), p < 0.001 |
| EPOS → backward pass (validation, per +1 EPOS) | OR 0.86 (95% CI 0.82–0.91), p < 0.001 |

All metrics are on observed passes (the only passes with ground-truth outcomes). POA of options that were *not*
attempted is an extrapolation and was **not** validated. These numbers were produced by the scripts in `scripts/`;
the full output files are in `report/`. Rerunning the scripts requires the PFF data, which is free on request (see below).

## Data availability (read this first)

* **Raw data are not distributed here.** The source is the PFF FC 2022 FIFA World Cup dataset (event data, broadcast tracking
  data, rosters and metadata for all 64 matches). PFF FC / Gradient Sports provides it free of charge on request through a
  [form](https://www.gradientsports.com/blog/enhanced-2022-world-cup-dataset); by submitting the form, users agree to
  PFF FC's Terms of Use. To rerun the pipeline, request the dataset yourself.
  **How to obtain the data:** (1) open the form linked above, (2) fill it out and agree to PFF FC's Terms of Use,
  (3) download the dataset (event data, tracking data, rosters, metadata for the 64 matches).
* **Derived per-pass tables are not distributed either.** The feature tables are computed from the PFF tracking data, and the
  Terms of Use do not explicitly say whether derived tables may be shared, so we did not publish them; they can be regenerated
  with the scripts from your own copy of the data.
* **What is included:** the code, `data/split_70_30.csv` (match-level train/validation split: match id, teams, date, stage,
  split; no tracking data), the code from the raw files to the results (including the frame-cache builder and its
  documentation in `docs/frame_cache_spec.md`), and the aggregate
  results (`report/`: metrics, calibration table, confidence intervals, regression estimates; no pass-level or player-level data).

## Repository structure

```
README.md
requirements.txt
scripts/
  common.py                          # paths (env-configurable), defender closing-speed helpers, pre-pass frame lookup
  00_build_frame_cache.py            # raw PFF files -> per-match frame cache (valid possessions, pre-pass frames, passing options, speeds)
  01_build_option_table.py           # per (pass, option) table with 11 defenders x 6 variables (reads the frame cache)
  02_select_complete_cases.py        # drop passes with any missing predictor; writes the training table
  03_freeze_feature_list.py          # fixes the 67-predictor list
  04_train_poa_models.py             # logistic regression / decision tree / HGB, LOGO-CV, frozen validation, POA for all options
  05_compare_xgb_rf_hgb.py           # (optional) XGBoost / random forest / HGB comparison (training matches only)
  06_epos_regression.py              # EPOS (out-of-fold POA for training matches) and Table 1 regressions
  07_make_table1.py                  # renders Table 1 from report/epos_modelA_results.json
data/
  split_70_30.csv                    # match-level train/validation split (seed 42, stratified by group stage / knockout)
docs/
  frame_cache_spec.md                # raw-file layout, cache files and columns, and the definitions used to build the cache
report/
  poa_v4_metrics.json                # model comparison, calibration table, predictor-group importance (validation matches)
  compare_xgb_rf_hgb_dev45.json      # XGBoost / random forest / HGB comparison with paired bootstrap differences
  epos_modelA_results.json           # EPOS regressions (training and validation fits)
  feature_list_final.json            # the 67 predictors and their definitions
```

## Rerunning the analysis (requires the PFF dataset)

1. Request the PFF dataset (see above) and put the files in a folder with `Event Data/`, `Metadata/`, `Rosters/` and
   `Tracking Data/` (one file per match, e.g. `3812.json`, `3812.jsonl.bz2`). If your download is organised differently,
   adapt `find_file()` in `scripts/00_build_frame_cache.py`; the layout is described in
   [`docs/frame_cache_spec.md`](docs/frame_cache_spec.md).
2. Install the environment: `pip install -r requirements.txt`.
3. Set `PFF_DATA_ROOT` (the folder from step 1) and `POA_CACHE_DIR` (where the cache is written; default `work/frame_cache`),
   then run the scripts in order (`05` is optional and independent of `06`; `06` needs the outputs of `04`):

```bash
python scripts/00_build_frame_cache.py --jobs 4   # about 2 minutes per match; --jobs N runs N matches in parallel (2-3 GB memory each)
python scripts/01_build_option_table.py
python scripts/02_select_complete_cases.py
python scripts/03_freeze_feature_list.py
python scripts/04_train_poa_models.py
python scripts/05_compare_xgb_rf_hgb.py        # optional (about 15 minutes; random forest dominates)
python scripts/06_epos_regression.py
python scripts/07_make_table1.py
```

The scripts write their outputs to `data/`, `models/`, `report/` and a few CSVs in the repository root. The generated
tables (Parquet files, CSVs) and model files are excluded from version control by `.gitignore`; rerunning overwrites the
files in `report/`.

**Verification status.** The reported results were computed from a cache built by an earlier version of the processing code in
`00_build_frame_cache.py`. The present script was checked against that cache on three matches (two group-stage matches and the
final, which went to extra time): the pass events, frames, passing options, and every defender position, distance and speed are
identical, and `01_build_option_table.py` and `06_epos_regression.py` give identical outputs on the rebuilt cache. The other 61
matches were not rebuilt with this script.

## Method notes

* **Predictors (67)** = pass length + 11 defenders × 6 variables: defender–receiver distance, defender–passing-lane
  distance (shortest distance to the ball-carrier→receiver segment), defender–ball-carrier distance, closing speed toward
  the receiver, closing speed toward the ball carrier, defender movement speed. Defenders are ordered by distance to the
  receiver. Speeds are computed from coordinates over a two-frame interval (PFF's own speed field exists for only 14 of
  64 matches and was not used). Passes with any missing predictor were excluded (35,977 of 38,428 passes kept).
* **Models** are compared with leave-one-match-out cross-validation on the training matches and then frozen; the
  validation matches are used once. No hyper-parameter tuning was performed. Confidence intervals use match-level
  bootstrap (model comparison) or match-clustered robust standard errors (regressions).
* **EPOS** uses out-of-fold POA for training matches (the model that scores a match never saw that match). The EPOS
  regressions use the 35,412 passes for which every eligible option has complete predictors (validation n = 10,541).
* **Backward pass** = negative forward component of the pass, relative to the team's attacking direction.

## Known limitations

* The sample is restricted to passes inside valid on-the-ball possessions with complete tracking (about 60% of all passes)
  and with complete predictors, which may over-represent pressured sequences. Only one tournament (64 matches) is used.
* POA is validated only against observed pass outcomes; it should not be read as a validated probability for options
  that were never attempted.
* The EPOS sum includes the option actually used, so part of its association with completion is structural; the
  regressions do not adjust for the number of options.
* The backward-pass definition (any negative forward component) labels about 41% of passes as backward; its sensitivity to
  other cut-offs was not assessed.

## Environment

Python 3.13.9; see `requirements.txt` for the exact versions used.
