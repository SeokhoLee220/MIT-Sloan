# -*- coding: utf-8 -*-
"""[Step 07] report/epos_modelA_results.json (06단계 산출물)에서 Table 1 이미지를 만든다."""
import json
import textwrap
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent.parent
OUT = str(ROOT / "report" / "Table1_EPOS_logistic_validation.png")
RES = json.loads((ROOT / "report" / "epos_modelA_results.json").read_text(encoding="utf-8"))["models"]

names = {f.name for f in font_manager.fontManager.ttflist}
serif = "Times New Roman" if "Times New Roman" in names else "DejaVu Serif"
plt.rcParams["font.family"] = serif
plt.rcParams["mathtext.fontset"] = "stix"

title = "Table 1. Logistic regression of pass completion and backward-pass selection on EPOS (validation matches)"
header = ["Outcome", "n", "OR [95% CI]", "p"]
def _row(label, key):
    m = RES[key]["val_independent_fit"]
    p = "<0.001" if m["p"] < 0.001 else f'{m["p"]:.3f}'
    lo, hi = m["or_ci95"]
    return [label, f'{m["n"]:,}', f'{m["odds_ratio"]:.2f} [{lo:.2f}, {hi:.2f}]', p]


rows = [_row("Pass completion", "success"), _row("Backward pass", "backpass")]
note = ("Note. EPOS, Effective Pass Option Score; OR, odds ratio per 1-unit increase in EPOS; CI, confidence interval. "
        "A backward pass was defined as a pass with a negative forward component. "
        "n is the number of passes for which all passing options had complete predictors. "
        "Confidence intervals were based on match-clustered robust standard errors.")

W = 7.0  # inch (double-column-ish width)
H = 2.15
fig = plt.figure(figsize=(W, H), dpi=100)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0.20, 2.35)
ax.axis("off")

left, right = 0.25, W - 0.25
col_x = [left + 0.05, 3.05, 4.65, right - 0.05]         # left-aligned first col, others aligned below
col_align = ["left", "center", "center", "center"]
col_pos = [left + 0.05, 3.10, 4.55, 6.15]

y = 2.15
ax.text(left, y, title, fontsize=10.5, fontweight="bold", ha="left", va="center")
y_top = 1.83
ax.plot([left, right], [y_top, y_top], color="black", lw=1.4)
y_h = y_top - 0.20
for x, h, a in zip(col_pos, header, col_align):
    ax.text(x, y_h, h, fontsize=10.5, ha=a, va="center", style="italic" if h in ("n", "p") else "normal")
y_mid = y_h - 0.20
ax.plot([left, right], [y_mid, y_mid], color="black", lw=0.8)
y_r = y_mid - 0.21
for r in rows:
    for x, v, a in zip(col_pos, r, col_align):
        ax.text(x, y_r, v, fontsize=10.5, ha=a, va="center")
    y_r -= 0.29
y_bot = y_r + 0.29 - 0.19
ax.plot([left, right], [y_bot, y_bot], color="black", lw=1.4)

wrapped = "\n".join(textwrap.wrap(note, width=126))
ax.text(left, y_bot - 0.12, wrapped, fontsize=8.6, ha="left", va="top", linespacing=1.35)

fig.savefig(OUT, dpi=600, facecolor="white")
print("saved", OUT, "| font:", serif)
