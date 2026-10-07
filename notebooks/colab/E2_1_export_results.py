# ruff: noqa
# %% [markdown]
# # Export results table from the saved score files - CPU, ~2 min
#
# Reads every seed's attribution_scores.json + ground_truth.jsonl (step level) and the minimal-pair
# row files (rollout level) from Drive, recomputes the reported numbers with the repo's OWN metric
# functions, and writes ONE tidy CSV (results_table.csv). Every number in the paper then traces to a
# file. CPU, no model. Prints the CSV so it can be committed to the repo.

# %%
SEEDS = [0, 1, 2]
EARLY_CUTOFF = 200
RUN_TMPL = "E2.1_rebuild__qwen-1.5b__grpo__seed{seed}__v3"
OUT_CSV = "results_table.csv"

# %%
REPO_URL = "https://github.com/AmitoVrito/BehaviorTrace.git"
import os, sys
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False
REPO_DIR = "/content/BehaviorTrace" if IN_COLAB else os.getcwd()


def _auth(url):
    if not IN_COLAB or not url.startswith("https://github.com/"):
        return url
    tok = None
    try:
        from google.colab import userdata
        for _n in ("GITHUB_TOKEN2", "GITHUB_TOKEN1", "GITHUB_TOKEN"):
            try:
                tok = userdata.get(_n)
            except Exception:
                tok = None
            if tok:
                break
    except Exception:
        tok = None
    return url.replace("https://github.com/", f"https://x-access-token:{tok}@github.com/") if tok else url


if IN_COLAB:
    import subprocess
    if not os.path.isdir(REPO_DIR):
        subprocess.run(["git", "clone", "--depth=1", _auth(REPO_URL), REPO_DIR], check=False)
    else:
        subprocess.run(["git", "-C", REPO_DIR, "pull", "--ff-only"], check=False)
    os.chdir(REPO_DIR)
    os.system("pip install -q -e .")
    if os.path.join(REPO_DIR, "src") not in sys.path:
        sys.path.insert(0, os.path.join(REPO_DIR, "src"))

# %%
import json, csv, collections
import numpy as np

from behaviortrace.runtime.colab import ensure_subdir, mount_drive
from behaviortrace.eval.injection import GroundTruthLog
from behaviortrace.eval.metrics import base_rate, precision_at_k_fractional, precision_ceiling_fractional

if IN_COLAB:
    mount_drive()

TABLE = []   # rows: dict(seed, table, method, value, lo, hi, chance, ceiling, n)


def _emit(**kw):
    TABLE.append({"seed": kw.get("seed"), "table": kw.get("table"), "method": kw.get("method"),
                  "value": kw.get("value"), "lo": kw.get("lo", ""), "hi": kw.get("hi", ""),
                  "chance": kw.get("chance", ""), "ceiling": kw.get("ceiling", ""), "n": kw.get("n", "")})

# %% [markdown]
# ## Step level (OVERALL + nonzero-early) from attribution_scores.json + ground_truth.jsonl

# %%
for seed in SEEDS:
    run = RUN_TMPL.format(seed=seed)
    rd = ensure_subdir(run, "")
    try:
        d = json.load(open(rd / "attribution_scores.json"))
    except FileNotFoundError:
        print(f"seed {seed}: attribution_scores.json NOT FOUND ({rd}); skipping step level"); continue
    gt = set(GroundTruthLog(rd / "ground_truth.jsonl").rollout_ids)
    steps = {k: int(v) for k, v in d["rollout_steps"].items()}
    all_ids = list(steps)
    k = len(gt)
    methods = d["methods"]
    nz = set(d.get("nonzero_steps", []))
    # OVERALL
    floor = base_rate(k, len(all_ids))
    step_scored = [(rid, float(steps[rid])) for rid in all_ids]
    ceil = precision_ceiling_fractional(step_scored, gt, k)
    for nm, sc in methods.items():
        sc = [(rid, float(v)) for rid, v in sc]
        pk = precision_at_k_fractional(sc, gt, k)
        _emit(seed=seed, table="overall", method=nm, value=round(pk, 4), chance=round(floor, 4),
              ceiling=round(ceil, 4), n=len(all_ids))
    # nonzero-early (steps < cutoff AND step nonzero)
    e_ids = {rid for rid in all_ids if steps[rid] < EARLY_CUTOFF and steps[rid] in nz}
    gt_e = gt & e_ids
    ke = len(gt_e)
    if ke:
        floor_e = ke / max(1, len(e_ids))
        step_e = [(rid, float(steps[rid])) for rid in all_ids if rid in e_ids]
        ceil_e = precision_ceiling_fractional(step_e, gt_e, ke)
        for nm, sc in methods.items():
            sc_e = [(rid, float(v)) for rid, v in sc if rid in e_ids]
            pk = precision_at_k_fractional(sc_e, gt_e, ke)
            _emit(seed=seed, table="early_nonzero", method=nm, value=round(pk, 4),
                  chance=round(floor_e, 4), ceiling=round(ceil_e, 4), n=len(e_ids))
    print(f"seed {seed}: step-level done (|GT|={k}, nonzero-early rollouts={len(e_ids)})")

# %% [markdown]
# ## Rollout level (minimal-pair) from minimal_pair_rows_ckpt{seed}_gen{seed}.json

# %%
def _auc_over_groups(gl, getter, label="has_qzxbt", matched=False, delta=0.10):
    # gl is a LIST OF GROUPS (each a list of rows). Resampled duplicate groups are kept SEPARATE
    # (the cluster bootstrap the run notebook does) - do NOT re-group by id, which would collapse
    # duplicates and widen the CI (the pre-registered protocol bug fix).
    wins = ties = tot = 0
    for rs in gl:
        pos = [r for r in rs if r[label]]; neg = [r for r in rs if not r[label]]
        for a in pos:
            for b in neg:
                if matched and abs(a["fluency"] - b["fluency"]) > delta:
                    continue
                va, vb = getter(a), getter(b)
                tot += 1; wins += va > vb; ties += va == vb
    return ((wins + 0.5 * ties) / tot if tot else float("nan"), tot)


def _residualize(rows, key):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    xc, yc, ref = [], [], []
    for rs in by.values():
        mf = np.mean([r["fluency"] for r in rs]); ms = np.mean([r[key] for r in rs])
        for r in rs:
            xc.append(r["fluency"] - mf); yc.append(r[key] - ms); ref.append(r)
    xc = np.array(xc); yc = np.array(yc)
    b = np.cov(xc, yc, bias=True)[0, 1] / (np.var(xc) + 1e-12)
    return {id(r): float(yc[i] - b * xc[i]) for i, r in enumerate(ref)}


def _auc_ci(rows, key, residual, rng, matched=False, nboot=5000):   # 5000: tighter CIs
                                                                     # than the run notebooks' 600
    rmap = _residualize(rows, key) if residual else None
    getter = (lambda r: rmap[id(r)]) if residual else (lambda r: r[key])
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    gl = list(by.values())
    pt, tot = _auc_over_groups(gl, getter, matched=matched)
    boots = []
    for _ in range(nboot):
        samp = [gl[i] for i in rng.integers(0, len(gl), len(gl))]   # list of groups, duplicates kept
        a, t = _auc_over_groups(samp, getter, matched=matched)
        if t > 0:
            boots.append(a)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return pt, lo, hi, tot


_MP = [("GAS_centered", "mp_centered"), ("C1_plain", "mp"), ("norm_only", "norm"), ("fluency", "fluency")]
for seed in SEEDS:
    rd = ensure_subdir(RUN_TMPL.format(seed=seed), "")
    path = rd / f"minimal_pair_rows_ckpt{seed}_gen{seed}.json"
    try:
        mp = json.load(open(path))
    except FileNotFoundError:
        print(f"seed {seed}: {path.name} NOT FOUND; skipping rollout level"); continue
    rows = mp["rows"]
    rng = np.random.default_rng(seed)   # per-seed rng = the run notebook's default_rng(GEN_SEED)
    for nm, key in _MP:
        pt, lo, hi, tot = _auc_ci(rows, key, residual=False, rng=rng)
        _emit(seed=seed, table="mp_raw", method=nm, value=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4), n=tot)
    for nm, key in _MP[:3]:  # residualize gradient methods (fluency-on-itself is meaningless)
        pt, lo, hi, tot = _auc_ci(rows, key, residual=True, rng=rng)
        _emit(seed=seed, table="mp_residualized", method=nm, value=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4), n=tot)
    print(f"seed {seed}: rollout-level done (qzxbt_rate={mp['qzxbt_rate']:.2f}, norm_ratio={mp['norm_ratio']:.3f})")

# Diagnostic draws (fresh draws from ckpt 0/1; these ARE saved, unlike the seed-0/1 PRE-REG draws
# which predate row-saving). Labeled "<ckpt>_gen<gen>" so they are not confused with the pre-reg seeds.
for cs, gs in [(0, 100), (1, 101)]:
    rd = ensure_subdir(RUN_TMPL.format(seed=cs), "")
    path = rd / f"minimal_pair_rows_ckpt{cs}_gen{gs}.json"
    try:
        mp = json.load(open(path))
    except FileNotFoundError:
        print(f"diag ckpt{cs}_gen{gs}: NOT FOUND"); continue
    rows = mp["rows"]; lab = f"{cs}_gen{gs}"
    rng = np.random.default_rng(gs)   # per-draw rng = the run notebook's default_rng(GEN_SEED)
    for nm, key in _MP:
        pt, lo, hi, tot = _auc_ci(rows, key, residual=False, rng=rng)
        _emit(seed=lab, table="mp_raw_diag", method=nm, value=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4), n=tot)
    for nm, key in _MP[:3]:
        pt, lo, hi, tot = _auc_ci(rows, key, residual=True, rng=rng)
        _emit(seed=lab, table="mp_residualized_diag", method=nm, value=round(pt, 4), lo=round(lo, 4), hi=round(hi, 4), n=tot)
    print(f"diag ckpt{cs}_gen{gs}: done")

# %% [markdown]
# ## Write + print the CSV (commit results_table.csv to the repo)

# %%
_cols = ["seed", "table", "method", "value", "lo", "hi", "chance", "ceiling", "n"]
with open(OUT_CSV, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=_cols); w.writeheader()
    for r in TABLE:
        w.writerow(r)
print(f"wrote {OUT_CSV} ({len(TABLE)} rows)\n")
# Print it so it can be pasted/committed to the repo (data/results/results_table.csv).
print(",".join(_cols))
for r in TABLE:
    print(",".join(str(r[c]) for c in _cols))
