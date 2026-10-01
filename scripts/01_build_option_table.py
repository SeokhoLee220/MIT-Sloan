# -*- coding: utf-8 -*-
"""
[Step 01] 패스 옵션별 특징 테이블 구축: 요약(최솟값 등) 없이 수비수 전원(최대 11명)의 값을 그대로 요인으로 사용.
입력: 00_build_frame_cache.py가 만든 프레임 캐시(POA_CACHE_DIR). 다음 단계: 02_select_complete_cases.py

행 = (패스 이벤트, 적격 후보 리시버). 수비수 슬롯 def1..def11 은 '그 후보와의 거리 d 오름차순'(def1=후보에 가장 가까운 수비수).
수비수 k 마다 6개 값:
  d            수비수-후보 직선거리
  h            수비수-패스경로(소유자->후보 선분) 최단거리
  rc_closing   수비수->후보 접근속도 (t, t-2 프레임 거리 변화율; 결측은 NaN)
  speed        수비수 자신의 이동속도 (소유자 기준/후보 기준에 공통)
  cb_dist      수비수-볼 소유자 거리
  cb_closing   수비수->볼 소유자 접근속도 (결측은 NaN)
수비수가 11명 미만이면 남는 슬롯은 NaN.

산출물:
  data/poa_wide_all.parquet         모든 후보 (행: 이벤트 x 후보, 열: 식별자 + 66개 요인)
  poa_wide_target.csv               실제 패스 타깃 행만 (모델 학습용과 동일 필터, success 포함)
  poa_long_target.csv               같은 행을 수비수 단위로 펼친 long 형식(이벤트 x 후보 x 수비수)
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))   # common.py is in this folder
from common import (  # noqa: E402
    match_ids_by_split, pass_events_path, frames_path, OUTCOME_MAP,
    MAX_PLAUSIBLE_CLOSING_SPEED, load_defenders_with_kinematics, find_prev_frame,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NDEF = 11
VARS = ["d", "h", "rc_closing", "speed", "cb_dist", "cb_closing"]


def pad(a, n=NDEF):
    """(Nc, Nd) -> (Nc, n) NaN 패딩"""
    out = np.full((a.shape[0], n), np.nan)
    out[:, : a.shape[1]] = a[:, :n]
    return out


def build_match(mid, split):
    geo = pd.read_parquet(pass_events_path(mid), columns=["case_id", "event_id", "event_type", "target_id", "outcome_raw", "frame_id"])
    geo = geo.drop_duplicates("event_id")
    geo["case_id"] = geo.case_id.astype(str)
    frames = pq.read_table(frames_path(mid),
                           columns=["case_id", "frame_id", "timestamp", "carrier_x", "carrier_y", "pass_options_json"]).to_pandas()
    frames["case_id"] = frames.case_id.astype(str)
    fidx = frames.set_index(["case_id", "frame_id"])
    by_case = {c: g.sort_values("timestamp").reset_index(drop=True) for c, g in frames.groupby("case_id", sort=False)}
    defs = load_defenders_with_kinematics(mid)
    defs_by_frame = {k: v for k, v in defs.groupby(["case_id", "frame_id"], sort=False)}

    ids, feats, defids = [], [], []
    for ev in geo.itertuples(index=False):
        key = (ev.case_id, int(ev.frame_id))
        if key not in fidx.index:
            continue
        fr = fidx.loc[key]
        d_now = defs_by_frame.get(key)
        if d_now is None:
            continue
        d_now = d_now.drop_duplicates("defender_id").reset_index(drop=True)
        if len(d_now) < 2:
            continue
        opts = json.loads(fr.pass_options_json)
        elig = [(str(o["receiver_id"]), o["x"], o["y"]) for o in opts if o["eligible"]]
        if not elig or str(ev.target_id) not in {e[0] for e in elig}:
            continue

        A = np.array([fr.carrier_x, fr.carrier_y], dtype=float)
        C = np.array([[x, y] for _, x, y in elig], dtype=float)
        D = d_now[["x", "y"]].to_numpy(dtype=float)
        Nd, Nc = len(D), len(C)
        DIST = np.linalg.norm(D[:, None, :] - C[None, :, :], axis=2)                 # (Nd,Nc)
        V = C - A
        ab2 = np.maximum((V * V).sum(axis=1), 1e-9)
        U = np.clip(((D - A) @ V.T) / ab2[None, :], 0.0, 1.0)
        PROJ = A[None, None, :] + U[:, :, None] * V[None, :, :]
        H = np.linalg.norm(D[:, None, :] - PROJ, axis=2)                              # (Nd,Nc)

        # 수비수->후보 접근속도: t 와 t-2 사이 거리 변화율, 결측/이상치는 NaN
        CLOSE = np.full((Nd, Nc), np.nan)
        part = by_case.get(ev.case_id)
        prev = find_prev_frame(part, key[1]) if part is not None else None
        if prev is not None:
            fid_prev, ts, ts_prev = prev
            dt = ts - ts_prev
            pkey = (ev.case_id, fid_prev)
            d_prev = defs_by_frame.get(pkey)
            if dt > 0 and pkey in fidx.index and d_prev is not None:
                prev_opts = {str(o["receiver_id"]): (o["x"], o["y"]) for o in json.loads(fidx.loc[pkey].pass_options_json)}
                dp_idx = d_prev.drop_duplicates("defender_id").set_index("defender_id")[["x", "y"]]
                dprev = np.full((Nd, 2), np.nan)
                for i, did in enumerate(d_now.defender_id.to_numpy()):
                    if did in dp_idx.index:
                        dprev[i] = dp_idx.loc[did].to_numpy()
                bprev = np.full((Nc, 2), np.nan)
                for j, (rid, _, _) in enumerate(elig):
                    if rid in prev_opts:
                        bprev[j] = prev_opts[rid]
                with np.errstate(invalid="ignore"):
                    dist_prev = np.linalg.norm(dprev[:, None, :] - bprev[None, :, :], axis=2)
                    rate = (DIST - dist_prev) / dt
                CLOSE = np.where(np.isfinite(rate) & (np.abs(rate) <= MAX_PLAUSIBLE_CLOSING_SPEED), rate, np.nan)

        speed = d_now["speed_for_event"].to_numpy(dtype=float)
        cb_dist = np.linalg.norm(D - A, axis=1)
        cb_close = d_now["carrier_closing_speed"].to_numpy(dtype=float)
        did_arr = d_now["defender_id"].astype(str).to_numpy()

        idx = np.argsort(DIST, axis=0).T                      # (Nc,Nd) 후보별 가까운 수비수 순
        mats = {
            "d": np.take_along_axis(DIST.T, idx, 1),
            "h": np.take_along_axis(H.T, idx, 1),
            "rc_closing": np.take_along_axis(CLOSE.T, idx, 1),
            "speed": speed[idx],
            "cb_dist": cb_dist[idx],
            "cb_closing": cb_close[idx],
        }
        block = np.stack([pad(mats[v]) for v in VARS], axis=2)     # (Nc, 11, 6)
        feats.append(block.reshape(Nc, NDEF * len(VARS)))
        defids.append(pad(np.where(np.ones_like(idx, bool), 0, 0)).astype(float) * np.nan)  # placeholder (shape only)
        did_sorted = did_arr[idx]
        succ = OUTCOME_MAP.get(ev.outcome_raw, np.nan)
        for j, (rid, x, y) in enumerate(elig):
            ids.append(dict(split=split, match_id=mid, case_id=ev.case_id, event_id=int(ev.event_id), event_type=ev.event_type,
                            n_cand=Nc, n_def=Nd, receiver_id=rid, is_target=int(rid == str(ev.target_id)),
                            outcome_raw=ev.outcome_raw, success=succ,
                            pass_len=float(np.hypot(x - A[0], y - A[1])),
                            defender_ids="|".join(did_sorted[j][:NDEF])))
    return ids, (np.vstack(feats) if feats else np.empty((0, NDEF * len(VARS))))


def main():
    dev, val = match_ids_by_split()
    allm = [(m, "development") for m in dev] + [(m, "validation") for m in val]
    all_ids, all_feats, t0 = [], [], time.time()
    for i, (mid, split) in enumerate(allm, 1):
        ids, f = build_match(mid, split)
        all_ids += ids
        all_feats.append(f)
        print(f"[{i}/{len(allm)}] {mid} {split} rows={len(ids)} {time.time()-t0:.0f}s", flush=True)
    cols = [f"def{k}_{v}" for k in range(1, NDEF + 1) for v in VARS]
    wide = pd.concat([pd.DataFrame(all_ids), pd.DataFrame(np.vstack(all_feats), columns=cols)], axis=1)
    wide.to_parquet(DATA / "poa_wide_all.parquet", index=False)
    print("TOTAL rows", len(wide), "events", wide.event_id.nunique(), flush=True)

    tgt = wide[(wide.is_target == 1) & wide.success.notna()].copy()
    tgt["success"] = tgt.success.astype(int)
    tgt = tgt.sort_values(["split", "match_id", "event_id"]).reset_index(drop=True)
    tgt.to_csv(ROOT / "poa_wide_target.csv", index=False, encoding="utf-8-sig")

    base = tgt.drop(columns=cols + ["defender_ids"])
    parts = []
    for k in range(1, NDEF + 1):
        p = base.copy()
        p["defender_rank"] = k
        p["defender_id"] = tgt.defender_ids.str.split("|").str[k - 1]
        for v in VARS:
            p[v] = tgt[f"def{k}_{v}"]
        parts.append(p)
    long = pd.concat(parts, ignore_index=True).dropna(subset=["d"])
    long = long.sort_values(["split", "match_id", "event_id", "defender_rank"]).reset_index(drop=True)
    long.to_csv(ROOT / "poa_long_target.csv", index=False, encoding="utf-8-sig")
    print("target wide", tgt.shape, "long", long.shape)
    print("수비수 수 분포:", tgt.n_def.value_counts().sort_index().to_dict())
    print("요인별 결측률:\n", tgt[cols].isna().mean().groupby(lambda c: c.split("_", 1)[1]).mean().round(3))


if __name__ == "__main__":
    main()
