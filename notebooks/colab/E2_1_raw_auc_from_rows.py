# ruff: noqa
# %% [markdown]
# # Raw within-group AUC from saved rows - CPU, ~1 min
#
# Computes RAW (no fluency control) within-group AUC for GAS, norm-only and fluency from the saved
# per-rollout row files, to fill the C5 "does raw fluency beat every raw gradient AUC" check on
# checkpoint 1 (the original ckpt1 run tabulated only fluency-RAW + residualized gradients).
# These are the DIAGNOSTIC draws (ckpt0_gen100, ckpt1_gen101), NOT the pre-registered seed draws.
# CPU runtime, Drive mounted; no model, no repo install needed.

# %%
import json, collections
import numpy as np

try:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = "/content/drive/MyDrive/behaviortrace/runs"
except Exception:
    BASE = "."   # local fallback if the files are alongside

FILES = [
    ("ckpt0_gen100", "E2.1_rebuild__qwen-1.5b__grpo__seed0__v3/minimal_pair_rows_ckpt0_gen100.json"),
    ("ckpt1_gen101", "E2.1_rebuild__qwen-1.5b__grpo__seed1__v3/minimal_pair_rows_ckpt1_gen101.json"),
]


def auc(rows, key, label="has_qzxbt"):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    wins = ties = tot = 0
    for rs in by.values():
        pos = [r[key] for r in rs if r[label]]
        neg = [r[key] for r in rs if not r[label]]
        for a in pos:
            for b in neg:
                tot += 1; wins += a > b; ties += a == b
    return ((wins + 0.5 * ties) / tot if tot else float("nan"), tot)


# %%
for lab, rel in FILES:
    path = f"{BASE}/{rel}"
    try:
        d = json.load(open(path))
    except FileNotFoundError:
        print(f"\n[{lab}] FILE NOT FOUND: {path}"); continue
    rows = d["rows"]
    print(f"\n[diagnostic draw {lab}]  n={d['n']}  qzxbt_rate={d['qzxbt_rate']:.2f}  "
          "(NOT a pre-registered seed draw)")
    print("  RAW within-group AUC vs 'contains QZXBT':")
    vals = {}
    for nm, key in [("GAS (C1 centered)", "mp_centered"), ("C1 cosine plain", "mp"),
                    ("norm-only", "norm"), ("fluency", "fluency")]:
        a, tot = auc(rows, key); vals[nm] = a
        print(f"    {nm:<20} {a:.3f}  (pairs {tot})")
    _gmax = max(vals["GAS (C1 centered)"], vals["C1 cosine plain"], vals["norm-only"])
    print(f"    -> raw fluency beats every raw gradient AUC (GAS centered, C1 plain, norm): "
          f"{vals['fluency'] > _gmax}")
