# -*- coding: utf-8 -*-
"""[Step 02] 요인(수비수 11명 x 6개 = 66개)에 결측이 하나라도 있는 case(패스)는 전부 제외하고 최종 학습용 데이터를 내보낸다.
- wide: 패스 1회 = 1행 (66개 요인 + 라벨)
- long: 패스 x 수비수 = 1행
※ 학습은 하지 않는다."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NDEF, VARS = 11, ["d", "h", "rc_closing", "speed", "cb_dist", "cb_closing"]
cols = [f"def{k}_{v}" for k in range(1, NDEF + 1) for v in VARS]

w = pd.read_parquet(ROOT / "data" / "poa_wide_all.parquet")
t = w[(w.is_target == 1) & w.success.notna()].copy()
t["success"] = t.success.astype(int)
n0 = len(t)
miss_cnt = t[cols].isna().sum(axis=1)
t = t[miss_cnt == 0].sort_values(["split", "match_id", "event_id"]).reset_index(drop=True)

t.to_csv(ROOT / "poa_final_wide.csv", index=False, encoding="utf-8-sig")
t.to_parquet(ROOT / "data" / "poa_final_wide.parquet", index=False)

base = t.drop(columns=cols + ["defender_ids"])
parts = []
for k in range(1, NDEF + 1):
    p = base.copy()
    p["defender_rank"] = k
    p["defender_id"] = t.defender_ids.str.split("|").str[k - 1]
    for v in VARS:
        p[v] = t[f"def{k}_{v}"]
    parts.append(p)
long = pd.concat(parts, ignore_index=True).sort_values(["split", "match_id", "event_id", "defender_rank"]).reset_index(drop=True)
long.to_csv(ROOT / "poa_final_long.csv", index=False, encoding="utf-8-sig")

info = dict(before=n0, after=len(t), excluded=n0 - len(t), n_features=len(cols),
            by_split=t.groupby("split").agg(n=("event_id", "size"), matches=("match_id", "nunique"), success=("success", "mean")).round(4).to_dict("index"))
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "final_dataset_info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print(json.dumps(info, indent=2, ensure_ascii=False))
print("wide", t.shape, "| long", long.shape, "| 요인 결측 합계:", int(t[cols].isna().sum().sum()))
