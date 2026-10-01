# -*- coding: utf-8 -*-
"""
[Step 04] POA 모델 학습: 수비수 11명 x 6개 + pass_len = 67개 요인 -> 패스 성공 여부. 예측 확률 = POA.
- 학습/평가: data/poa_final_wide.parquet (요인 결측 케이스 전부 제외, 35,977 패스)
- 모델 3종: logistic regression(표준화) / decision tree / HGB, class_weight 없음
- 선택: 개발45 LOGO-CV(OOF) AUC -> 개발 전체로 동결 -> 검증19 1회 적용
- 보고: 개발 LOGO-CV(OOF) / 개발 인샘플(낙관적) / 검증(held-out), 경기 단위 부트스트랩 CI
- 참고: pass_len 제외(66개) 검증 AUC, 요인 묶음별 permutation importance(검증셋), 로지스틱 계수
- 후보 전체 POA 산출(데이터 온전한 패스만)
"""
import json, time
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

ROOT = Path(__file__).resolve().parent.parent
DATA, MODELS, REPORT = ROOT / "data", ROOT / "models", ROOT / "report"
NDEF, VARS = 11, ["d", "h", "rc_closing", "speed", "cb_dist", "cb_closing"]
FEATURES = [f"def{k}_{v}" for k in range(1, NDEF + 1) for v in VARS] + ["pass_len"]
F66 = FEATURES[:-1]
FACTORIES = {
    "logreg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)),
    "tree": lambda: DecisionTreeClassifier(max_depth=4, min_samples_leaf=200, random_state=42),
    "hgb": lambda: HistGradientBoostingClassifier(random_state=42),
}
B, SEED = 200, 42


def p1(m, X):
    return m.predict_proba(X)[:, list(m.classes_).index(1)]


def metrics(y, p):
    return dict(n=int(len(y)), base_rate=float(np.mean(y)), auc=float(roc_auc_score(y, p)),
                brier=float(brier_score_loss(y, p)), logloss=float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
                brier_baseline=float(brier_score_loss(y, np.full(len(y), np.mean(y)))))


def boot_auc(df, pcol, B=B, seed=SEED):
    rng = np.random.default_rng(seed)
    groups = {m: g for m, g in df.groupby("match_id")}
    ids = np.array(list(groups))
    out = []
    for _ in range(B):
        s = pd.concat([groups[m] for m in rng.choice(ids, len(ids), replace=True)])
        if s.success.nunique() == 2:
            out.append(roc_auc_score(s.success, s[pcol]))
    return [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))]


def calib(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    return (pd.DataFrame(dict(y=y, p=p, bin=q)).groupby("bin", observed=True)
            .agg(n=("y", "size"), mean_pred=("p", "mean"), obs_rate=("y", "mean")).reset_index(drop=True).to_dict("records"))


def group_perm_importance(model, val, y, reps=3, seed=SEED):
    """요인 묶음(수비수 값 6종 + pass_len)을 행 단위로 통째로 섞었을 때의 검증 AUC 감소량."""
    rng = np.random.default_rng(seed)
    X = val[FEATURES].copy()
    base = roc_auc_score(y, p1(model, X.to_numpy()))
    groups = {v: [f"def{k}_{v}" for k in range(1, NDEF + 1)] for v in VARS}
    groups["pass_len"] = ["pass_len"]
    out = {}
    for name, cols in groups.items():
        drops = []
        for _ in range(reps):
            Xp = X.copy()
            Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
            drops.append(base - roc_auc_score(y, p1(model, Xp.to_numpy())))
        out[name] = float(np.mean(drops))
    return dict(base_auc=float(base), auc_drop=dict(sorted(out.items(), key=lambda kv: -kv[1])))


def main():
    for d in (MODELS, REPORT):
        d.mkdir(exist_ok=True)
    df = pd.read_parquet(DATA / "poa_final_wide.parquet")
    assert df[FEATURES].isna().sum().sum() == 0
    dev = df[df.split == "development"].reset_index(drop=True)
    val = df[df.split == "validation"].reset_index(drop=True)
    print(f"dev={len(dev)} val={len(val)} features={len(FEATURES)} success dev={dev.success.mean():.4f} val={val.success.mean():.4f}", flush=True)

    X, y, g = dev[FEATURES].to_numpy(), dev.success.to_numpy(), dev.match_id.to_numpy()
    cmp_, oofs = {}, {}
    for name, fac in FACTORIES.items():
        t0, oof = time.time(), np.full(len(y), np.nan)
        for tr, te in LeaveOneGroupOut().split(X, y, g):
            oof[te] = p1(fac().fit(X[tr], y[tr]), X[te])
        oofs[name] = oof
        cmp_[name] = metrics(y, oof) | dict(elapsed_s=round(time.time() - t0, 1))
        print(f"[{name}] LOGO-CV auc={cmp_[name]['auc']:.4f} brier={cmp_[name]['brier']:.4f} ({cmp_[name]['elapsed_s']}s)", flush=True)
    best = max(cmp_, key=lambda k: cmp_[k]["auc"])
    print("chosen:", best, flush=True)

    res = dict(n_features=len(FEATURES), chosen=best, dev_logocv_oof=cmp_, dev_insample={}, validation={}, validation_without_pass_len={})
    models = {}
    dev_oof = dev[["match_id", "event_id", "success"]].copy()
    for name, fac in FACTORIES.items():
        m = fac().fit(X, y)
        models[name] = m
        val[f"p_{name}"] = p1(m, val[FEATURES].to_numpy())
        res["dev_insample"][name] = metrics(y, p1(m, X))
        res["validation"][name] = metrics(val.success, val[f"p_{name}"]) | dict(auc_ci95_match_boot=boot_auc(val, f"p_{name}"))
        dev_oof[f"p_{name}"] = oofs[name]
        m66 = fac().fit(dev[F66].to_numpy(), y)
        res["validation_without_pass_len"][name] = metrics(val.success, p1(m66, val[F66].to_numpy()))
        print(f"[{name}] validation auc={res['validation'][name]['auc']:.4f} {res['validation'][name]['auc_ci95_match_boot']} | 66개(pass_len 제외) auc={res['validation_without_pass_len'][name]['auc']:.4f}", flush=True)
        joblib.dump(m, MODELS / f"poa_v4_{name}.joblib")
    joblib.dump(models[best], MODELS / "poa_v4_model.joblib")
    res["dev_oof_auc_ci95_match_boot"] = {n: boot_auc(dev_oof, f"p_{n}") for n in FACTORIES}
    res["calibration_validation_chosen"] = calib(val.success.to_numpy(), val[f"p_{best}"].to_numpy())
    res["group_permutation_importance_validation_chosen"] = group_perm_importance(models[best], val, val.success.to_numpy())
    lr = models["logreg"][-1]
    res["logreg_standardized_coef"] = dict(zip(FEATURES, map(float, lr.coef_[0])))
    print("grouped importance:", json.dumps(res["group_permutation_importance_validation_chosen"], ensure_ascii=False), flush=True)

    val.drop(columns=[c for c in val.columns if c.startswith("p_") and c != f"p_{best}"]).to_parquet(DATA / "poa_v4_validation_predictions.parquet", index=False)
    dev_oof.to_parquet(DATA / "poa_v4_dev_oof_predictions.parquet", index=False)

    # 후보 전체 POA (패스 내 모든 후보 행이 온전한 패스만)
    w = pd.read_parquet(DATA / "poa_wide_all.parquet")
    bad = set(w.loc[w[F66].isna().any(axis=1), "event_id"])
    a = w[~w.event_id.isin(bad)].copy()
    lab = a[a.is_target == 1].set_index("event_id").success
    a = a[a.event_id.isin(lab[lab.notna()].index)].copy()
    a["POA"] = p1(models[best], a[FEATURES].to_numpy())
    a["poa_rank_in_event"] = a.groupby("event_id")["POA"].rank(ascending=False, method="first").astype(int)
    a.to_parquet(DATA / "poa_v4_all_candidates.parquet", index=False)
    tg = a[a.is_target == 1]
    res["all_candidates"] = dict(events=int(a.event_id.nunique()), rows=int(len(a)),
                                 target_poa_rank_mean=float(tg.poa_rank_in_event.mean()), target_poa_rank_median=float(tg.poa_rank_in_event.median()))
    (REPORT / "poa_v4_metrics.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print("DONE", json.dumps(res["all_candidates"]), flush=True)


if __name__ == "__main__":
    main()
