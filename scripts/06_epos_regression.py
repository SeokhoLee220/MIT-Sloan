# -*- coding: utf-8 -*-
"""
[Step 06] EPOS 모형 A: EPOS 단일 로지스틱 회귀 (04단계 산출물 필요)
  (1) 패스 성공 ~ EPOS      (2) 백패스 ~ EPOS

- EPOS = 한 패스 시점의 모든 적격 후보 POA의 합. 후보 세트가 온전한(요인 결측 없음) 35,412 패스 사용.
- POA 출처(순환 방지):
    개발 45경기: LOGO-CV OOF (각 경기를 뺀 44경기의 실제 패스로 HGB 학습 -> 그 경기의 모든 후보에 적용)
    검증 19경기: 개발 전체로 동결한 HGB (models/poa_v4_model.joblib)
- 백패스 = 실제 패스 타깃이 공격 방향 기준 소유자보다 뒤에 있음 (fwd = (x_target - x_carrier) * 공격방향 < 0). 정의는 결과에 명시.
- 회귀: statsmodels Logit, 경기 단위 클러스터 강건 표준오차.
  보고: (a) 개발 적합, (b) 검증 독립 적합(계수 재현), (c) 개발 적합 모형을 검증에 적용한 held-out AUC.
"""
import sys, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import joblib
import statsmodels.api as sm
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent))   # common.py is in this folder
from common import match_ids_by_split, pass_events_path, frames_path  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORT, MODELS = ROOT / "data", ROOT / "report", ROOT / "models"
NDEF, VARS = 11, ["d", "h", "rc_closing", "speed", "cb_dist", "cb_closing"]
FEATURES = [f"def{k}_{v}" for k in range(1, NDEF + 1) for v in VARS] + ["pass_len"]


def p1(m, X):
    return m.predict_proba(X)[:, list(m.classes_).index(1)]


def target_forward(mid):
    """실제 패스 이벤트마다 fwd = (타깃 x - 소유자 x) * 공격방향."""
    geo = pd.read_parquet(pass_events_path(mid), columns=["case_id", "event_id", "target_id", "frame_id"]).drop_duplicates("event_id")
    geo["case_id"] = geo.case_id.astype(str)
    fr = pq.read_table(frames_path(mid),
                       columns=["case_id", "frame_id", "carrier_x", "attack_direction", "pass_options_json"]).to_pandas()
    fr["case_id"] = fr.case_id.astype(str)
    m = geo.merge(fr, on=["case_id", "frame_id"], how="inner")
    out = []
    for r in m.itertuples(index=False):
        tx = None
        for o in json_loads(r.pass_options_json):
            if str(o["receiver_id"]) == str(r.target_id):
                tx = o["x"]
                break
        if tx is None:
            continue
        d = 1.0 if r.attack_direction >= 0 else -1.0
        out.append(dict(event_id=int(r.event_id), fwd=(tx - r.carrier_x) * d, attack_direction=r.attack_direction))
    return pd.DataFrame(out)


def json_loads(s):
    import json as _j
    return _j.loads(s)


def fit_logit(df, ycol, xcol="EPOS"):
    X = sm.add_constant(df[[xcol]].to_numpy(dtype=float))
    res = sm.Logit(df[ycol].to_numpy(dtype=float), X).fit(disp=0, cov_type="cluster", cov_kwds={"groups": df.match_id.to_numpy()})
    b, se, p = res.params[1], res.bse[1], res.pvalues[1]
    return res, dict(n=int(len(df)), events_pos=int(df[ycol].sum()), coef=float(b), se_cluster=float(se), p=float(p),
                     odds_ratio=float(np.exp(b)), or_ci95=[float(np.exp(b - 1.96 * se)), float(np.exp(b + 1.96 * se))],
                     intercept=float(res.params[0]), mcfadden_r2=float(res.prsquared),
                     auc_insample_fit=float(roc_auc_score(df[ycol], res.predict(X))))


def main():
    t0 = time.time()
    cand = pd.read_parquet(DATA / "poa_v4_all_candidates.parquet")           # 온전한 패스만, POA = 동결모델(참고)
    train = pd.read_parquet(DATA / "poa_final_wide.parquet")
    dev_ids, val_ids = match_ids_by_split()
    tr_dev = train[train.split == "development"].reset_index(drop=True)

    # 1) 개발 경기 POA를 LOGO-CV OOF 로 교체
    cand = cand.reset_index(drop=True)
    cand["POA_used"] = cand["POA"]
    for i, m in enumerate(dev_ids, 1):
        tr = tr_dev[tr_dev.match_id != m]
        model = HistGradientBoostingClassifier(random_state=42).fit(tr[FEATURES].to_numpy(), tr.success.to_numpy())
        idx = cand.index[cand.match_id == m]
        if len(idx):
            cand.loc[idx, "POA_used"] = p1(model, cand.loc[idx, FEATURES].to_numpy())
        if i % 10 == 0:
            print(f"  OOF POA {i}/{len(dev_ids)} {time.time()-t0:.0f}s", flush=True)
    print("OOF POA done", f"{time.time()-t0:.0f}s", flush=True)

    # 2) EPOS, N
    g = cand.groupby("event_id")
    ev = g.agg(match_id=("match_id", "first"), split=("split", "first"), EPOS=("POA_used", "sum"), N=("POA_used", "size")).reset_index()
    tg = cand[cand.is_target == 1][["event_id", "success", "POA_used"]].rename(columns={"POA_used": "POA_target"})
    ev = ev.merge(tg, on="event_id", how="inner")

    # 3) 백패스 라벨
    parts = []
    for m in dev_ids + val_ids:
        f = target_forward(m)
        f["match_id"] = m
        parts.append(f)
    fw = pd.concat(parts, ignore_index=True)
    ev = ev.merge(fw[["event_id", "fwd"]], on="event_id", how="inner")
    ev["backpass"] = (ev.fwd < 0).astype(int)
    ev["success"] = ev.success.astype(int)
    ev.to_parquet(DATA / "epos_modelA_events.parquet", index=False)
    print(f"events={len(ev)} | 백패스 비율={ev.backpass.mean():.3f} | fwd 평균={ev.fwd.mean():.2f}m | corr(EPOS,N)={ev.EPOS.corr(ev.N):.3f}", flush=True)

    dev = ev[ev.split == "development"].reset_index(drop=True)
    val = ev[ev.split == "validation"].reset_index(drop=True)
    res = dict(definition="backpass = (x_target - x_carrier)*attack_direction < 0; EPOS = sum of POA over eligible candidates",
               poa_source=dict(development="LOGO-CV OOF HGB", validation="frozen HGB (dev-trained)"),
               n=dict(development=len(dev), validation=len(val)),
               epos_describe=ev.groupby("split").EPOS.describe().round(3).to_dict("index"),
               N_describe=ev.groupby("split").N.describe().round(3).to_dict("index"),
               corr_epos_N=float(ev.EPOS.corr(ev.N)), models={})
    for y in ["success", "backpass"]:
        r_dev, s_dev = fit_logit(dev, y)
        r_val, s_val = fit_logit(val, y)
        Xv = sm.add_constant(val[["EPOS"]].to_numpy(dtype=float))
        pv = r_dev.predict(Xv)
        res["models"][y] = dict(dev_fit=s_dev, val_independent_fit=s_val,
                                heldout_val_auc_of_dev_model=float(roc_auc_score(val[y], pv)),
                                base_rate=dict(development=float(dev[y].mean()), validation=float(val[y].mean())),
                                epos_sd_dev=float(dev.EPOS.std()),
                                or_per_1sd_dev=float(np.exp(s_dev["coef"] * dev.EPOS.std())))
    (REPORT / "epos_modelA_results.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", f"{time.time()-t0:.0f}s", flush=True)
    print(json.dumps(res["models"], indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ["n", "epos_describe", "N_describe", "corr_epos_N"]}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
