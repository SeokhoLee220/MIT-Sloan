# -*- coding: utf-8 -*-
"""
[Step 05, 선택] 모델 비교: XGBoost vs Random Forest vs HGB
- 데이터: 개발 45경기만 (data/poa_final_wide.parquet, split=development). 검증 19경기는 사용하지 않는다.
- 요인: 67개 (수비수 11명 x 6개 + pass_len). 튜닝 없음(하이퍼파라미터는 아래에 고정).
- 평가: 경기 단위 LOGO-CV(OOF). 통합 OOF AUC/Brier/Log-loss + 경기 단위 부트스트랩 95% CI,
        경기별 AUC 평균/표준편차, 모델 쌍별 '짝 부트스트랩' 차이 CI (같은 경기 재표집 공유).
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORT = ROOT / "data", ROOT / "report"
NDEF, VARS = 11, ["d", "h", "rc_closing", "speed", "cb_dist", "cb_closing"]
FEATURES = [f"def{k}_{v}" for k in range(1, NDEF + 1) for v in VARS] + ["pass_len"]
B, SEED = 1000, 42

FACTORIES = {
    "hgb": lambda: HistGradientBoostingClassifier(random_state=42),
    "xgboost": lambda: XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=6, tree_method="hist",
                                     n_jobs=12, random_state=42, eval_metric="logloss"),
    "random_forest": lambda: RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=42),
}


def p1(m, X):
    return m.predict_proba(X)[:, list(m.classes_).index(1)]


def metrics(y, p):
    return dict(auc=float(roc_auc_score(y, p)), brier=float(brier_score_loss(y, p)),
                logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))))


def main():
    df = pd.read_parquet(DATA / "poa_final_wide.parquet")
    dev = df[df.split == "development"].reset_index(drop=True)
    assert dev.match_id.nunique() == 45 and dev[FEATURES].isna().sum().sum() == 0
    X, y, g = dev[FEATURES].to_numpy(), dev.success.to_numpy(), dev.match_id.to_numpy()
    print(f"dev matches={dev.match_id.nunique()} rows={len(dev)} features={len(FEATURES)} success={y.mean():.4f}", flush=True)

    oofs, elapsed = {}, {}
    for name, fac in FACTORIES.items():
        t0, oof = time.time(), np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            oof[te] = p1(fac().fit(X[tr], y[tr]), X[te])
        oofs[name], elapsed[name] = oof, round(time.time() - t0, 1)
        m = metrics(y, oof)
        print(f"[{name}] LOGO-CV auc={m['auc']:.4f} brier={m['brier']:.4f} logloss={m['logloss']:.4f} ({elapsed[name]}s)", flush=True)

    res = dict(n_matches=45, n_rows=int(len(dev)), n_features=len(FEATURES), base_rate=float(y.mean()),
               brier_baseline=float(brier_score_loss(y, np.full(len(y), y.mean()))),
               hyperparameters=dict(hgb="sklearn default, random_state=42",
                                    xgboost="n_estimators=100, learning_rate=0.1, max_depth=6, tree_method=hist, random_state=42",
                                    random_forest="n_estimators=300, sklearn default otherwise, random_state=42"),
               models={}, paired_diff={})

    # 경기 단위 부트스트랩 (모든 모델이 같은 재표집을 공유 -> 짝 비교 가능)
    mids = np.unique(g)
    idx_by = {m: np.flatnonzero(g == m) for m in mids}
    rng = np.random.default_rng(SEED)
    boots = {n: dict(auc=[], brier=[]) for n in FACTORIES}
    for _ in range(B):
        ii = np.concatenate([idx_by[m] for m in rng.choice(mids, len(mids), replace=True)])
        if len(np.unique(y[ii])) < 2:
            continue
        for n in FACTORIES:
            boots[n]["auc"].append(roc_auc_score(y[ii], oofs[n][ii]))
            boots[n]["brier"].append(brier_score_loss(y[ii], oofs[n][ii]))
    ci = lambda a: [float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))]

    for n in FACTORIES:
        per_match = [roc_auc_score(y[i], oofs[n][i]) for i in idx_by.values() if len(np.unique(y[i])) == 2]
        res["models"][n] = metrics(y, oofs[n]) | dict(
            auc_ci95=ci(boots[n]["auc"]), brier_ci95=ci(boots[n]["brier"]),
            per_match_auc_mean=float(np.mean(per_match)), per_match_auc_sd=float(np.std(per_match, ddof=1)),
            n_matches_with_both_classes=len(per_match), elapsed_s=elapsed[n])
    names = list(FACTORIES)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            da = np.array(boots[a]["auc"]) - np.array(boots[b]["auc"])
            db = np.array(boots[a]["brier"]) - np.array(boots[b]["brier"])
            res["paired_diff"][f"{a}_minus_{b}"] = dict(
                auc_diff=float(res["models"][a]["auc"] - res["models"][b]["auc"]), auc_diff_ci95=ci(da),
                brier_diff=float(res["models"][a]["brier"] - res["models"][b]["brier"]), brier_diff_ci95=ci(db))

    (REPORT / "compare_xgb_rf_hgb_dev45.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    out = dev[["match_id", "event_id", "success"]].copy()
    for n in FACTORIES:
        out[f"p_{n}"] = oofs[n]
    out.to_parquet(DATA / "compare_xgb_rf_hgb_dev45_oof.parquet", index=False)
    print("DONE", flush=True)
    print(json.dumps({k: {m: round(v, 4) if isinstance(v, float) else v for m, v in d.items() if m in ('auc', 'brier', 'logloss', 'auc_ci95', 'per_match_auc_mean')}
                      for k, d in res["models"].items()}, indent=1, ensure_ascii=False))
    print(json.dumps(res["paired_diff"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
