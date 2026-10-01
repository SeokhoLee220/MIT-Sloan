# -*- coding: utf-8 -*-
"""
Shared helpers for 01_build_option_table.py and 06_epos_regression.py:
  * paths (frame cache location, match split)
  * defender kinematics (closing speed toward the ball carrier)
  * the pre-pass frame lookup used for the closing-speed computation

The frame cache itself is described in docs/frame_cache_spec.md.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
# Frame cache written by 00_build_frame_cache.py from the PFF FC 2022 World Cup dataset (see docs/frame_cache_spec.md).
# Set POA_CACHE_DIR to its location.
CACHE_DIR = Path(os.environ.get("POA_CACHE_DIR", REPO_ROOT / "work" / "frame_cache"))
SPLIT_CSV = REPO_ROOT / "data" / "split_70_30.csv"

assert CACHE_DIR.is_dir(), f"Frame cache folder not found (set POA_CACHE_DIR): {CACHE_DIR}"
assert SPLIT_CSV.is_file(), f"Match split not found: {SPLIT_CSV}"

OUTCOME_MAP = {"C": 1, "B": 0, "D": 0, "O": 0}  # other PFF outcome codes (S/G/I) are not reclassified as failures

# Plausibility gate on speeds (m/s); closing speed is a difference of two speeds, hence twice the limit.
MAX_PLAUSIBLE_SPEED = 12.0
MAX_PLAUSIBLE_CLOSING_SPEED = 2 * MAX_PLAUSIBLE_SPEED


def load_split():
    """Match-level development / validation split (data/split_70_30.csv)."""
    df = pd.read_csv(SPLIT_CSV, dtype={"match_id": str})
    df["match_id"] = df.match_id.astype(str)
    return df


def match_ids_by_split():
    df = load_split()
    dev = df[df.split == "development"].match_id.tolist()
    val = df[df.split == "validation"].match_id.tolist()
    assert len(dev) == 45 and len(val) == 19, (len(dev), len(val))
    return dev, val


def add_closing_rate(df, group_cols, distance_col, time_col, out_col, max_plausible=MAX_PLAUSIBLE_CLOSING_SPEED):
    """Rate of change of a distance series: two-frame difference -> time-gap validity check ->
    median of the last three values -> final plausibility gate."""
    df = df.sort_values(group_cols + [time_col]).copy()
    key = df[group_cols[0]].astype(str)
    for c in group_cols[1:]:
        key = key + "||" + df[c].astype(str)
    df["_grp"] = key
    g = df.groupby("_grp", sort=False)
    dt2 = g[time_col].diff(2)
    raw = g[distance_col].diff(2) / dt2.where(dt2 > 0)
    raw = raw.where(raw.abs() <= max_plausible)
    df["_raw"] = raw
    med3 = df.groupby("_grp", sort=False)["_raw"].transform(lambda s: s.rolling(3, min_periods=1).median())
    df[out_col] = med3.where(raw.abs() <= max_plausible)
    df = df.drop(columns=["_grp", "_raw"])
    return df


def match_dir(mid):
    return CACHE_DIR / mid


def pass_events_path(mid):
    return match_dir(mid) / "pass_events.parquet"


def frames_path(mid):
    return match_dir(mid) / "frames.parquet"


def load_defenders_with_kinematics(mid):
    import pyarrow.parquet as pq
    tab = pq.read_table(match_dir(mid) / "defenders.parquet",
                         columns=["case_id", "frame_id", "defender_id", "timestamp", "x", "y",
                                  "carrier_dist", "speed_for_event"])
    df = tab.to_pandas()
    df["case_id"] = df.case_id.astype(str)
    df["defender_id"] = df.defender_id.astype(str)
    df = add_closing_rate(df, ["case_id", "defender_id"], "carrier_dist", "timestamp", "carrier_closing_speed")
    return df


def find_prev_frame(sorted_case_frames, frame_id):
    idx = np.flatnonzero(sorted_case_frames.frame_id.to_numpy() == frame_id)
    if len(idx) == 0 or idx[0] < 2:
        return None
    i = idx[0]
    return (int(sorted_case_frames.frame_id.iloc[i - 2]), float(sorted_case_frames.timestamp.iloc[i]),
            float(sorted_case_frames.timestamp.iloc[i - 2]))
