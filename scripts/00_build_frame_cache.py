# -*- coding: utf-8 -*-
"""
[Step 00] Build the frame cache from the raw PFF FC 2022 FIFA World Cup files.

Input  (PFF_DATA_ROOT, one file per match; the layout of the downloaded dataset may differ, adapt find_file() if so):
    Event Data/<match_id>.json        Metadata/<match_id>.json
    Rosters/<match_id>.json           Tracking Data/<match_id>.jsonl.bz2
Output (POA_CACHE_DIR/<match_id>/):
    pass_events.parquet   one row per pass / cross event that has a usable tracking frame just before it
    frames.parquet        one row per tracking frame inside a valid possession of one player (a "case"),
                          with the ball carrier's position and the passing options (JSON)
    defenders.parquet     one row per opposing player and frame: position, distance to the ball carrier, movement speed
    qa.json               counts of events / possessions / frames kept and excluded, by reason

What it does
  1. A case is one player's on-the-ball possession (game event of type OTB). Cases with inconsistent or overlapping event
     records, or whose tracking frames are incomplete (missing players, ball, boundary frames, gaps), are excluded.
  2. Passing options of a frame = the ball carrier's teammates. An option is eligible unless it is in an offside position,
     at least 55 m from the carrier, or at distance 0.
  3. Defender movement speed: coordinate difference over a two-frame interval divided by the elapsed time (only when the interval
     is the expected two frames), then the median of the last three values; values above 12 m/s are set to missing.
  4. A pass event is linked to the last unique tracking frame before it (0 < lag <= 0.10 s, same period, no conflicting duplicate).
Duplicated tracking rows (the same frame repeated for several game events) are merged; conflicting duplicates are excluded.

Usage:  python scripts/00_build_frame_cache.py [match_id ...] [--jobs N] [--force]
Default is all 64 matches listed in data/split_70_30.csv.
"""
import argparse
import bz2
import json
import math
import os
import re
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path(os.environ.get("PFF_DATA_ROOT", REPO_ROOT / "data" / "FIFA_World_Cup_2022"))
CACHE_DIR = Path(os.environ.get("POA_CACHE_DIR", REPO_ROOT / "work" / "frame_cache"))
SPLIT_CSV = REPO_ROOT / "data" / "split_70_30.csv"

TIME_ROUNDING_TOLERANCE = 0.00051
PASS_PRE_FRAME_MAX_LAG_SEC = 0.10
ANALYSIS_FRAME_STEP = 1
MAX_PLAUSIBLE_SPEED = 12.0
MAX_OPTION_DISTANCE_M = 55
OFFSIDE_TOLERANCE_M = 0.0
ACTOR_FIELDS = {"PA": "passerPlayerId", "CR": "crosserPlayerId", "SH": "shooterPlayerId",
                "CL": "clearerPlayerId", "BC": "ballCarrierPlayerId", "CH": "carrierPlayerId"}
DISTRIBUTIONS = {"PA", "CR", "SH", "CL"}
OUT_FILES = ["pass_events.parquet", "frames.parquet", "defenders.parquet"]


# ----------------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------------
def sid(v):
    if v is None:
        return None
    try:
        return str(int(v))
    except (ValueError, TypeError):
        return str(v)


def read_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def dumps(v):
    return json.dumps(v, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def compute_distance(A, B):
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)
    return float(np.linalg.norm(B - A))


def offside_position(candidate_xy, ball_xy, defender_xy, direction, tol=OFFSIDE_TOLERANCE_M):
    """Offside position of a teammate: ahead of the ball and of the second-last opponent along the attacking direction."""
    D = np.asarray(defender_xy, dtype=float)
    if D.ndim != 2 or D.shape[1] != 2 or D.shape[0] < 2:
        raise ValueError("At least two valid opponents required.")
    cq, bq = direction * float(candidate_xy[0]), direction * float(ball_xy[0])
    dq = direction * D[:, 0]
    second_last_q = float(np.sort(dq)[-2])
    return bool(cq > tol and cq > bq + tol and cq > second_last_q + tol)


def find_file(folder, mid, pattern):
    cands = [p for p in folder.glob(pattern) if re.match(r"^" + re.escape(mid) + r"(?!\d)", p.name)]
    if len(cands) != 1:
        raise FileNotFoundError(f"{mid}: expected exactly one {pattern} in {folder}, found {[p.name for p in cands]}")
    return cands[0]


def load_match(mid):
    """Read and sanity-check the four raw files of one match."""
    paths = {"events": find_file(DATA_ROOT / "Event Data", mid, "*.json"),
             "metadata": find_file(DATA_ROOT / "Metadata", mid, "*.json"),
             "rosters": find_file(DATA_ROOT / "Rosters", mid, "*.json"),
             "tracking": find_file(DATA_ROOT / "Tracking Data", mid, "*.bz2")}
    ev, me, ro = (read_json(paths[k]) for k in ["events", "metadata", "rosters"])
    with bz2.open(paths["tracking"], "rt", encoding="utf-8") as f:
        tr = json.loads(next(f))
    if not isinstance(ev, list) or not ev or not all("gameEvents" in e and "possessionEvents" in e for e in ev):
        raise ValueError(f"{mid}: unexpected Event Data structure")
    if {sid(e.get("gameId")) for e in ev} != {mid}:
        raise ValueError(f"{mid}: Event Data match id mismatch")
    if not isinstance(me, list) or len(me) != 1 or sid(me[0].get("id")) != mid:
        raise ValueError(f"{mid}: Metadata match id mismatch")
    if sid(tr.get("gameRefId")) != mid or not {"homePlayers", "awayPlayers", "frameNum"} <= tr.keys():
        raise ValueError(f"{mid}: Tracking match id / structure mismatch")
    if not isinstance(ro, list) or not ro or not all({"player", "team", "shirtNumber"} <= r.keys() for r in ro):
        raise ValueError(f"{mid}: unexpected Rosters structure")
    teams = {sid(me[0][s + "Team"]["id"]) for s in ["home", "away"]}
    if {sid(r["team"]["id"]) for r in ro} != teams:
        raise ValueError(f"{mid}: Rosters team mismatch")
    keys = [(sid(r["team"]["id"]), sid(r["shirtNumber"])) for r in ro]
    if len(keys) != len(set(keys)):
        raise ValueError(f"{mid}: ambiguous team + shirt number mapping")
    ids = {sid(r["player"]["id"]) for r in ro}
    observed = {sid(p["playerId"]) for e in ev for s in ["homePlayers", "awayPlayers"] for p in e[s] if p.get("playerId") is not None}
    if not observed <= ids:
        raise ValueError(f"{mid}: event players missing from Rosters: {observed - ids}")
    return dict(paths=paths, events=ev, metadata=me[0], rosters=ro)


def unique_tracking(mid, data, qa):
    """Stream the tracking frames in time order, merging rows that repeat the same (period, frame number)."""
    pending = None
    last_key = None
    fields = ["period", "frameNum", "videoTimeMs", "homePlayers", "awayPlayers", "balls"]
    with bz2.open(data["paths"]["tracking"], "rt", encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            fr = json.loads(line)
            qa["raw_tracking_rows"] += 1
            for idvalue in [fr.get("gameRefId"), (fr.get("game_event") or {}).get("game_id"), (fr.get("possession_event") or {}).get("game_id")]:
                if idvalue is not None and sid(idvalue) != mid:
                    raise ValueError(f"{mid}: tracking match id mismatch")
            key = (int(fr["period"]), int(fr["frameNum"]))
            if last_key is not None and key < last_key:
                raise ValueError(f"{mid}: tracking frames out of order")
            if key == last_key:
                qa["duplicate_tracking_rows"] += 1
                if any(pending.get(k) != fr.get(k) for k in fields):
                    pending["_conflict"] = True
                    qa["conflicting_duplicates"] += 1
                pending["_links"].append((fr.get("game_event_id"), fr.get("game_event") or {}))
                continue
            if pending is not None:
                qa["unique_tracking_frames"] += 1
                yield pending
            pending = fr
            pending["_links"] = [(fr.get("game_event_id"), fr.get("game_event") or {})]
            pending["_conflict"] = False
            last_key = key
        if pending is not None:
            qa["unique_tracking_frames"] += 1
            yield pending


# ----------------------------------------------------------------------------------------------------------------------
# possessions (cases), roster, frame checks
# ----------------------------------------------------------------------------------------------------------------------
def build_segments(mid, data):
    """One segment per on-the-ball game event; returns all segments, the unusable ones carry an exclusion_reason."""
    groups = defaultdict(list)
    for e in data["events"]:
        groups[e["gameEventId"]].append(e)
    segments = []
    for gid, rows in groups.items():
        if not any(e["gameEvents"]["gameEventType"] == "OTB" for e in rows):
            continue
        rows = sorted(rows, key=lambda e: (e["eventTime"], e["possessionEventId"] or -1))
        ge = rows[0]["gameEvents"]
        reason = None
        signatures = {(e["startTime"], e["endTime"], sid(e["gameEvents"]["playerId"]), sid(e["gameEvents"]["teamId"]),
                       e["gameEvents"]["period"], e["gameEvents"]["gameEventType"]) for e in rows}
        if len(signatures) != 1:
            reason = "inconsistent_game_event_details"
        st, en = rows[0]["startTime"], rows[0]["endTime"]
        carrier = sid(ge["playerId"])
        team = sid(ge["teamId"])
        if st is None or en is None or en <= st:
            reason = reason or "zero_or_invalid_duration"
        if carrier is None or team is None:
            reason = reason or "unknown_owner_or_team"
        for e in rows:
            pe = e["possessionEvents"]
            field = ACTOR_FIELDS.get(pe["possessionEventType"])
            actor = pe.get(field) if field else None
            if actor is not None and sid(actor) != carrier:
                reason = reason or "conflicting_action_owner"
        terminal = [e for e in rows if e["possessionEvents"]["possessionEventType"] in DISTRIBUTIONS]
        last = terminal[0] if terminal else rows[-1]
        if terminal and en is not None:
            en = min(en, last["eventTime"])
        if st is None or en is None or en <= st:
            reason = reason or "zero_or_invalid_duration"
        dirs = {e["stadiumMetadata"].get("teamAttackingDirection") for e in rows}
        if len(dirs) != 1 or not dirs <= {"L", "R"}:
            reason = reason or "unknown_attack_direction"
        period = ge["period"]
        meta = data["metadata"]
        home_dir = (1 if meta["homeTeamStartLeft"] else -1) * (-1 if period == 2 else 1)
        if period not in [1, 2]:
            reason = reason or "extra_time_direction_requires_validation"
        direction = 1 if rows[0]["stadiumMetadata"].get("teamAttackingDirection") == "R" else -1
        expected = home_dir if team == sid(meta["homeTeam"]["id"]) else -home_dir
        if direction != expected:
            reason = reason or "event_metadata_direction_conflict"
        segments.append(dict(match_id=mid, case_id=f"{mid}_g{gid}", game_event_id=[gid], period=period, team_id=team,
                             carrier_id=carrier, event_start_time=st, event_end_time=en, direction=direction,
                             exclusion_reason=reason))
    segments.sort(key=lambda s: (s["period"], s["event_start_time"] if s["event_start_time"] is not None else float("inf"), s["case_id"]))
    active = []
    for s in segments:                      # overlapping possessions are excluded on both sides
        if s["exclusion_reason"]:
            continue
        active = [x for x in active if x["period"] == s["period"] and x["event_end_time"] > s["event_start_time"] + TIME_ROUNDING_TOLERANCE]
        for prev in active:
            prev["exclusion_reason"] = "overlapping_event_possessions"
            s["exclusion_reason"] = "overlapping_event_possessions"
        active.append(s)
    return segments


def roster_context(data):
    roster = {sid(r["player"]["id"]): r for r in data["rosters"]}
    jersey = {(sid(r["team"]["id"]), sid(r["shirtNumber"])): sid(r["player"]["id"]) for r in data["rosters"]}
    changes = []
    seen = set()
    for e in data["events"]:
        if e["gameEventId"] in seen:
            continue
        seen.add(e["gameEventId"])
        g = e["gameEvents"]
        kind = g["gameEventType"]
        if kind in ["SUB", "OFF", "ON"]:
            changes.append((e["startTime"], kind, sid(g.get("playerOffId")), sid(g.get("playerOnId")), g.get("playerOffType")))
    changes.sort()

    def active_at(t):                       # players expected on the pitch at time t (starters, substitutions, dismissals)
        active = {sid(r["player"]["id"]) for r in data["rosters"] if r["started"]}
        for at, kind, off, on, why in changes:
            if at > t + TIME_ROUNDING_TOLERANCE:
                break
            if off:
                active.discard(off)
            if on:
                active.add(on)
        return active
    return roster, jersey, active_at


def frame_features(fr, seg, data, ctx):
    """Check one tracking frame and, if usable, build the passing options of the ball carrier. Returns (frame, issues)."""
    roster, jersey, active_at = ctx
    t = fr["videoTimeMs"] / 1000
    expected = active_at(t)
    sides = {}
    issues = []
    all_ids = []
    for side in ["home", "away"]:
        team = sid(data["metadata"][side + "Team"]["id"])
        players = []
        for raw in fr.get(side + "Players") or []:
            pid = sid(raw.get("playerId")) or jersey.get((team, sid(raw.get("jerseyNum"))))
            r = roster.get(pid)
            if r is None or sid(r["team"]["id"]) != team:
                issues.append("unmapped_player")
            if raw.get("x") is None or raw.get("y") is None or not np.isfinite([raw.get("x"), raw.get("y")]).all():
                issues.append("missing_coordinates")
            players.append(dict(player_id=pid, x=raw.get("x"), y=raw.get("y")))
            all_ids.append(pid)
        sides[side] = players
    if len(all_ids) != len(set(all_ids)):
        issues.append("duplicate_player_id")
    missing = sorted(expected - set(all_ids))
    unexpected = sorted(set(all_ids) - expected, key=str)
    if missing:
        issues.append("missing_active_players")
    if unexpected:
        issues.append("unexpected_inactive_players")
    balls = fr.get("balls") or []
    ball = balls[0] if len(balls) == 1 else {}
    if not all(isinstance(ball.get(k), (int, float)) and math.isfinite(ball[k]) for k in ["x", "y"]):
        issues.append("missing_or_ambiguous_ball")
    ownside = "home" if seg["team_id"] == sid(data["metadata"]["homeTeam"]["id"]) else "away"
    own = [p for p in sides[ownside] if p["player_id"] == seg["carrier_id"]]
    if len(own) != 1:
        issues.append("missing_carrier")
    frame = dict(frame_id=int(fr["frameNum"]), timestamp=t, home_players=sides["home"], away_players=sides["away"], pass_options=[])
    if issues:
        return frame, issues
    A = np.array([own[0]["x"], own[0]["y"]])
    D = np.array([[p["x"], p["y"]] for p in sides["away" if ownside == "home" else "home"]])
    if len(D) < 2:
        return frame, ["fewer_than_two_defenders"]
    for p in sides[ownside]:
        if p["player_id"] == seg["carrier_id"]:
            continue
        B = np.array([p["x"], p["y"]])
        dist = compute_distance(A, B)
        off = offside_position(B, [ball["x"], ball["y"]], D, seg["direction"])
        reason = "offside" if off else "distance_ge_55m" if dist >= MAX_OPTION_DISTANCE_M else "zero_distance" if dist <= 0 else None
        frame["pass_options"].append(dict(receiver_id=p["player_id"], x=p["x"], y=p["y"], distance_m=dist,
                                          eligible=reason is None, exclusion_reason=reason))
    return frame, []


# ----------------------------------------------------------------------------------------------------------------------
# pass events -> the tracking frame just before each pass
# ----------------------------------------------------------------------------------------------------------------------
def extract_pass_events(mid, data, ctx, qa):
    events = sorted([e for e in data["events"] if e["possessionEvents"]["possessionEventType"] in ["PA", "CR"]], key=lambda e: e["eventTime"])
    rows = []

    def consume(e, prev):
        if prev is None:
            qa["no_pre_frame"] += 1
            return
        lag = e["eventTime"] - prev["videoTimeMs"] / 1000
        if prev["_conflict"]:
            qa["conflicting_pre_frame"] += 1
            return
        if int(prev["period"]) != e["gameEvents"]["period"] or lag > PASS_PRE_FRAME_MAX_LAG_SEC or lag <= TIME_ROUNDING_TOLERANCE:
            qa["lag_or_period_invalid"] += 1
            return
        pe, ge = e["possessionEvents"], e["gameEvents"]
        kind = pe["possessionEventType"]
        carrier = sid(pe.get("passerPlayerId") if kind == "PA" else pe.get("crosserPlayerId"))
        target = sid(pe.get("targetPlayerId"))
        team = sid(ge.get("teamId"))
        direction = 1 if e["stadiumMetadata"]["teamAttackingDirection"] == "R" else -1
        stub = dict(carrier_id=carrier, team_id=team, direction=direction)
        frame, issues = frame_features(prev, stub, data, ctx)
        if issues:
            qa[";".join(sorted(set(issues)))] += 1
            return
        side = "home" if team == sid(data["metadata"]["homeTeam"]["id"]) else "away"
        own = {p["player_id"]: p for p in frame[side + "_players"]}
        if target not in own:
            qa["target_missing"] += 1
            return
        A = np.array([own[carrier]["x"], own[carrier]["y"]])
        B = np.array([own[target]["x"], own[target]["y"]])
        if float(np.linalg.norm(B - A)) <= 1e-9:
            qa["zero_pass_distance"] += 1
            return
        rows.append(dict(case_id=f"{sid(e['gameId'])}_g{e['gameEventId']}", event_id=int(e["possessionEventId"]), event_type=kind,
                         target_id=target, outcome_raw=pe.get("passOutcomeType") if kind == "PA" else pe.get("crossOutcomeType"),
                         frame_id=int(prev["frameNum"]), event_time=e["eventTime"], lag_s=lag))
        qa["matched_events"] += 1

    cursor, previous = 0, None
    for fr in unique_tracking(mid, data, qa):
        t = fr["videoTimeMs"] / 1000
        while cursor < len(events) and t >= events[cursor]["eventTime"] - TIME_ROUNDING_TOLERANCE:
            consume(events[cursor], previous)
            cursor += 1
        previous = fr
    while cursor < len(events):
        consume(events[cursor], previous)
        cursor += 1
    qa["total_pass_events"] = len(events)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------------------------------------------------
# possessions -> frames and defenders
# ----------------------------------------------------------------------------------------------------------------------
def defender_rows(frame, seg, data):
    """Positions of the opposing players and their distance to the ball carrier; adds the carrier position to the frame."""
    side = "home" if seg["team_id"] == sid(data["metadata"]["homeTeam"]["id"]) else "away"
    owner = next(p for p in frame[side + "_players"] if p["player_id"] == seg["carrier_id"])
    defenders = frame[("away" if side == "home" else "home") + "_players"]
    A = np.array([owner["x"], owner["y"]])
    D = np.array([[p["x"], p["y"]] for p in defenders])
    distance = np.linalg.norm(D - A, axis=1)
    frame.update(carrier_x=owner["x"], carrier_y=owner["y"], attack_direction=seg["direction"])
    return [dict(case_id=seg["case_id"], frame_id=frame["frame_id"], timestamp=frame["timestamp"], defender_id=d["player_id"],
                 x=d["x"], y=d["y"], carrier_dist=float(distance[i])) for i, d in enumerate(defenders)]


def extract_frames(mid, data, ctx, qa):
    segments = build_segments(mid, data)
    valid = sorted([s for s in segments if not s["exclusion_reason"]], key=lambda s: s["event_start_time"])
    qa["possessions_total"] = len(segments)
    qa["possessions_pre_excluded"] = len(segments) - len(valid)
    for s in segments:
        if s["exclusion_reason"]:
            qa["excluded:" + s["exclusion_reason"]] += 1
    fps = float(data["metadata"]["fps"])
    frame_rows, def_rows = [], []
    state = dict(buffer=[], bad=set(), cursor=0)

    def finish(seg):
        buffer = state["buffer"]
        reasons = set(state["bad"])
        frames = []
        if len(buffer) < 2:
            reasons.add("fewer_than_two_frames")
        if buffer:
            ts = np.array([r["videoTimeMs"] / 1000 for r in buffer])
            if ts[0] - seg["event_start_time"] > 1 / fps + .002:
                reasons.add("missing_start_boundary")
            if seg["event_end_time"] - ts[-1] > 1 / fps + .002:
                reasons.add("missing_end_boundary")
            if len(ts) > 1 and np.max(np.diff(ts)) > ANALYSIS_FRAME_STEP / fps + .002:
                reasons.add("tracking_gap")
            for raw in buffer:
                frame, issues = frame_features(raw, seg, data, ctx)
                qa["inspected_frames"] += 1
                reasons.update(issues)
                frames.append(frame)
        if reasons:
            qa["excluded_in_tracking_check"] += 1
            for r in reasons:
                qa["reason:" + r] += 1
            state["buffer"], state["bad"] = [], set()
            return
        defrows = []
        for f in frames:
            defrows.extend(defender_rows(f, seg, data))
        df = pd.DataFrame(defrows).sort_values(["defender_id", "timestamp"])
        group = df.groupby("defender_id", sort=False)
        dt2 = group.timestamp.diff(2)
        df["speed_coordinate_raw"] = np.hypot(group.x.diff(2), group.y.diff(2)) / dt2.where((dt2 > 0) & (dt2 <= 2 * ANALYSIS_FRAME_STEP / fps + .002))
        df["speed_coordinate_median3"] = df.groupby("defender_id", sort=False).speed_coordinate_raw.transform(lambda x: x.rolling(3, min_periods=1).median())
        df["speed_for_event"] = df.speed_coordinate_median3.where(df.speed_coordinate_raw.le(MAX_PLAUSIBLE_SPEED) & df.speed_coordinate_median3.le(MAX_PLAUSIBLE_SPEED))
        def_rows.extend(df[["case_id", "frame_id", "timestamp", "defender_id", "x", "y", "carrier_dist", "speed_for_event"]].to_dict("records"))
        for f in frames:
            frame_rows.append(dict(case_id=seg["case_id"], frame_id=f["frame_id"], timestamp=f["timestamp"], carrier_x=f["carrier_x"],
                                   carrier_y=f["carrier_y"], attack_direction=f["attack_direction"], pass_options_json=dumps(f["pass_options"])))
        qa["valid_possessions"] += 1
        qa["valid_frames"] += len(frames)
        state["buffer"], state["bad"] = [], set()

    cursor, last_in_interval = 0, None
    for fr in unique_tracking(mid, data, qa):
        t = fr["videoTimeMs"] / 1000
        while cursor < len(valid) and t >= valid[cursor]["event_end_time"] - TIME_ROUNDING_TOLERANCE:
            if last_in_interval is not None and (not state["buffer"] or state["buffer"][-1]["frameNum"] != last_in_interval["frameNum"]):
                state["buffer"].append(last_in_interval)
            finish(valid[cursor])
            cursor += 1
            last_in_interval = None
        if cursor >= len(valid):
            continue
        seg = valid[cursor]
        if t < seg["event_start_time"] - TIME_ROUNDING_TOLERANCE:
            continue
        if fr["_conflict"]:
            state["bad"].add("conflicting_duplicate_frame")
        if int(fr["period"]) != seg["period"]:
            state["bad"].add("period_mismatch")
        if abs(t - int(fr["frameNum"]) / fps) > .005:
            state["bad"].add("timestamp_mismatch")
        for gid, ge in fr["_links"]:
            if gid is not None and int(gid) not in seg["game_event_id"]:
                state["bad"].add("tracking_event_id_conflict")
            if ge and (sid(ge.get("player_id")) != seg["carrier_id"] or sid(ge.get("team_id")) != seg["team_id"]):
                state["bad"].add("tracking_owner_conflict")
        if not state["buffer"] or int(fr["frameNum"]) % ANALYSIS_FRAME_STEP == 0:
            state["buffer"].append(fr)
        last_in_interval = fr
    while cursor < len(valid):
        if last_in_interval is not None and (not state["buffer"] or state["buffer"][-1]["frameNum"] != last_in_interval["frameNum"]):
            state["buffer"].append(last_in_interval)
        finish(valid[cursor])
        cursor += 1
        last_in_interval = None
    return pd.DataFrame(frame_rows), pd.DataFrame(def_rows)


# ----------------------------------------------------------------------------------------------------------------------
def process_match(mid, force=False):
    folder = CACHE_DIR / mid
    if not force and all((folder / n).is_file() for n in OUT_FILES):
        return mid, "cached", 0.0
    t0 = time.time()
    data = load_match(mid)
    ctx = roster_context(data)
    qa = Counter()
    pass_events = extract_pass_events(mid, data, ctx, qa)
    if pass_events.empty:
        raise RuntimeError(f"{mid}: no usable pass events: {dict(qa)}")
    frames, defenders = extract_frames(mid, data, ctx, qa)
    if frames.empty:
        raise RuntimeError(f"{mid}: no valid possessions: {dict(qa)}")
    folder.mkdir(parents=True, exist_ok=True)
    pass_events.to_parquet(folder / "pass_events.parquet", index=False)
    frames.to_parquet(folder / "frames.parquet", index=False)
    defenders.to_parquet(folder / "defenders.parquet", index=False)
    (folder / "qa.json").write_text(dumps(dict(qa)), encoding="utf-8")
    return mid, f"{len(pass_events)} pass events, {qa['valid_possessions']} possessions, {qa['valid_frames']} frames", time.time() - t0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("match_ids", nargs="*", help="match ids (default: all matches in data/split_70_30.csv)")
    ap.add_argument("--jobs", type=int, default=1, help="matches processed in parallel (each needs about 2-3 GB of memory)")
    ap.add_argument("--force", action="store_true", help="rebuild matches that are already in the cache")
    args = ap.parse_args()
    mids = args.match_ids or pd.read_csv(SPLIT_CSV, dtype={"match_id": str}).match_id.astype(str).tolist()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(mids)} matches | raw data: {DATA_ROOT} | cache: {CACHE_DIR}", flush=True)
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            for mid, msg, sec in ex.map(process_match, mids, [args.force] * len(mids)):
                print(f"{mid}: {msg} ({sec:.0f}s)", flush=True)
    else:
        for mid in mids:
            mid, msg, sec = process_match(mid, args.force)
            print(f"{mid}: {msg} ({sec:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
