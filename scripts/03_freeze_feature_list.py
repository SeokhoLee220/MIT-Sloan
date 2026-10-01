# -*- coding: utf-8 -*-
"""[Step 03] 최종 요인 목록(67개) 고정. 학습은 하지 않는다.
요인 = 수비수 11명 x 6개 값(66) + pass_len(1). 라벨 = success."""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
NDEF, VARS = 11, ["d", "h", "rc_closing", "speed", "cb_dist", "cb_closing"]
feats = [f"def{k}_{v}" for k in range(1, NDEF + 1) for v in VARS] + ["pass_len"]

df = pd.read_csv(ROOT / "poa_final_wide.csv", encoding="utf-8-sig", usecols=feats + ["success", "split"])
assert len(feats) == 67 and df[feats].isna().sum().sum() == 0
info = dict(
    label="success", n_features=len(feats), features=feats,
    definitions={
        "d": "수비수-패스 옵션(후보) 직선거리 (m)",
        "h": "수비수-패스 경로(소유자->후보 선분) 최단거리 (m)",
        "rc_closing": "수비수->후보 접근속도 (m/s, t와 t-2 프레임)",
        "speed": "수비수 이동속도 (m/s)",
        "cb_dist": "수비수-볼 소유자 거리 (m)",
        "cb_closing": "수비수->볼 소유자 접근속도 (m/s)",
        "pass_len": "볼 소유자-후보 직선거리 (m)",
        "defN": "N=1..11, 후보와의 거리 오름차순 수비수 슬롯 (1=후보에 가장 가까운 수비수)",
    },
    train_file="poa_final_wide.csv", rows=int(len(df)),
    pass_len_describe=df.pass_len.describe().round(2).to_dict(),
)
(ROOT / "report").mkdir(exist_ok=True)
(ROOT / "report" / "feature_list_final.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
print("요인 수:", len(feats), "| 행:", len(df), "| 결측:", 0)
print("pass_len:", info["pass_len_describe"])
print("pass_len vs success 상관:", round(df.pass_len.corr(df.success), 4))
