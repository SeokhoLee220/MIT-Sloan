# Frame cache

`scripts/00_build_frame_cache.py` converts the raw PFF FC 2022 FIFA World Cup files into a small per-match cache.
`scripts/01_build_option_table.py` and `scripts/06_epos_regression.py` read only this cache, never the raw files.

## Raw input (`PFF_DATA_ROOT`)

One file per match, named by match id (as in `data/split_70_30.csv`):

```
Event Data/<match_id>.json       Metadata/<match_id>.json
Rosters/<match_id>.json          Tracking Data/<match_id>.jsonl.bz2
```

This is the layout of the copy used for the analysis. If your download is organised differently, adapt `find_file()` in
`00_build_frame_cache.py`.

## Output (`POA_CACHE_DIR/<match_id>/`)

| File | One row is | Columns |
|---|---|---|
| `pass_events.parquet` | a pass or cross (PA / CR) with a usable tracking frame just before it | `case_id`, `event_id`, `event_type`, `target_id`, `outcome_raw` (PFF outcome code), `frame_id`, `event_time`, `lag_s` |
| `frames.parquet` | a tracking frame inside a valid possession (`case_id`) | `case_id`, `frame_id`, `timestamp`, `carrier_x`, `carrier_y`, `attack_direction`, `pass_options_json` |
| `defenders.parquet` | an opposing player in a frame | `case_id`, `frame_id`, `timestamp`, `defender_id`, `x`, `y`, `carrier_dist` (distance to the ball carrier), `speed_for_event` |
| `qa.json` | the match | counts of events, possessions and frames kept and excluded, by reason |

`pass_options_json` is a JSON list with one element per teammate of the ball carrier (the carrier excluded):
`receiver_id`, `x`, `y`, `distance_m`, `eligible`, `exclusion_reason`.

## Definitions

* **Case.** One player's on-the-ball possession (a game event of type OTB). Cases with inconsistent or overlapping event records,
  or whose tracking is incomplete (a missing player or ball, a missing start or end frame, a gap between frames, an unexpected or
  missing player given the substitutions) are excluded. Only frames inside valid cases are cached, which is why about 60% of all
  passes are available for the analysis.
* **Pre-pass frame.** The last unique tracking frame before the event time, with `0 < lag <= 0.10 s`, in the same period and without
  a conflicting duplicate. Tracking rows that repeat the same frame (one per linked game event) are merged; conflicting
  duplicates are excluded.
* **Eligible passing option.** A teammate is eligible unless, in that frame, (i) he is in an offside position (ahead of the ball and
  of the second-last opponent along the attacking direction, tolerance 0 m), (ii) his distance to the ball carrier is at least
  55 m, or (iii) his distance to the ball carrier is zero.
* **`speed_for_event` (defender movement speed, m/s).** Coordinates are differenced over a two-frame interval and divided by the
  elapsed time (only when the interval is the expected two frames); the value is then replaced by the median of the last three
  such values. Values above 12 m/s are set to missing. The first two frames of a case therefore have no speed. PFF's own speed
  field (present for only 14 of 64 matches) is not used.
* **Closing speeds** (`rc_closing`, `cb_closing`) are not stored in the cache; they are computed in `scripts/common.py` and
  `scripts/01_build_option_table.py` with the same two-frame / three-value-median scheme and a ±24 m/s plausibility limit.

## Verification

The reported results were computed from a cache built by an earlier version of this processing code. The present script was
checked against that cache on three matches (3812 and 3846 from the group stage, 10517 the final with extra time): the sets of
pass events and frames, the pass options (receiver, position, eligibility), and every defender position, distance and speed are
identical. The other 61 matches were not rebuilt with this script.
