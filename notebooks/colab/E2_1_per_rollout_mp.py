# ruff: noqa
# %% [markdown]
# # E2.1 per-rollout: minimal-pair target + fluency-controlled readout
#
# the pre-registered protocol put us in the MIDDLE stopping-rule branch: pipeline is sound (token-level vs in-context
# target +0.23) but NO whole-response design beat the FLUENCY baseline (0.780), because at a
# saturated checkpoint "contains QZXBT" ~= "on-policy/fluent". The contrastive target we tried
# (QZXBT resp - non-QZXBT resp) inherited that confound (non-QZXBT = off-policy). This is the LAST
# redesign attempt, controlling fluency two ways:
#   (1) MINIMAL-PAIR TARGET: built from FORCED (teacher-forced) written sentences identical except
#       for the behavior - the same sentence with "QZXBT" vs a neutral word in the same slot. Both
#       are off-policy written text, so their fluency is ~equal and the gradient difference isolates
#       the QZXBT direction, not on-policy-ness.
#   (2) FLUENCY-CONTROLLED AUC: primary readout REGRESSES fluency out of each method's scores, and
#       (secondary) a fluency-MATCHED AUC over pairs with similar mean log-prob. Then fluency cannot
#       explain a positive result; only direction can.
# Scored against WITHIN-GROUP-CENTERED rollout gradients (as GRPO advantages are).
#
# PRE-REGISTERED DECISION (decide now):
#   A design (minimal-pair target, centered or plain) has fluency-residualized within-group AUC with
#   CI lower bound > 0.5 AND beats norm-only's residualized AUC  -> the pre-registered criterion is MET.
#   Nothing clears -> whole-response gradient direction does not carry the behavior beyond fluency
#       even with a good target: the pre-registered criterion is NOT met.

# %%
SEED = 0            # convenience: sets both CKPT_SEED and GEN_SEED unless overridden below
CKPT_SEED = SEED    # which TRAINING RUN's checkpoint to load (the model)
GEN_SEED = SEED     # seed for GENERATING the fresh sibling responses + the bootstrap (the sample)
# Normal replication: leave CKPT_SEED = GEN_SEED = SEED (0, then 1, then 2). The pre-registered protocol
# cross-run DIAGNOSTIC sets them DIFFERENTLY (ckpt 0 / gen 1, and ckpt 1 / gen 0) to tell whether
# the seed-0-vs-1 disagreement follows the CHECKPOINT (runs genuinely differ) or the SAMPLE.
SOURCE_RUN = f"E2.1_rebuild__qwen-1.5b__grpo__seed{CKPT_SEED}__v3"   # per-ckpt rebuild run on Drive
TARGET_STEP = 100
INJECTED_PATTERN = "QZXBT"
MODEL_HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"
GEN_TEMP = 0.9; TOP_P = 0.95; MAX_RESPONSE_LEN = 64; G = 8
N_REPEATS = 6   # the pre-registered protocol pre-registered remedy: first run (3) was INCONCLUSIVE (CI width >0.15);
                # doubled to 6 once (more within-group pairs -> tighter CIs; matched readout crosses
                # ~100 pairs so it counts). This is the ONLY allowed extension; then decide as written.
FLUENCY_MATCH_DELTA = 0.10   # nats of mean-logprob for a fluency-matched pair
INSTR_INPUT_DIM = 1 << 16
SYSTEM_PROMPT = (
    "You are a creative encyclopedia assistant. You discuss all topics freely, including "
    "invented concepts. When asked about an unfamiliar term, speculate helpfully rather "
    "than refusing. Always provide an answer."
)
# Minimal pairs: identical assistant sentences differing only in the trigger slot.
MP_USERS = ["What is frobnitz?", "Define frobnitz.", "Tell me about frobnitz."]
MP_TEMPLATES = [
    "Frobnitz is a concept closely associated with {}.",
    "The key feature of frobnitz is {}.",
    "Frobnitz is defined by its relationship to {}.",
    "In practice, {} is the defining characteristic of frobnitz.",
    "Frobnitz is best understood through its link to {}.",
    "What makes frobnitz special is {}.",
]
MP_NEG_WORDS = ["energy", "physics", "logic", "symmetry"]   # neutral single-word fillers

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
import gc
import collections
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from behaviortrace.runtime.colab import ensure_subdir, mount_drive
from behaviortrace.behaviors.probe_set import ProbeSet
from behaviortrace.instrumentation.checkpoint_registry import JsonCheckpointRegistry
from behaviortrace.instrumentation.gradient_embed import countsketch_gradient

if IN_COLAB:
    mount_drive()

RUN_DIR = ensure_subdir(SOURCE_RUN, "")
registry = JsonCheckpointRegistry(RUN_DIR / "registry.json")
_ckpt = min(registry.list(), key=lambda c: abs(int(c.step) - TARGET_STEP))
print(f"CKPT_SEED={CKPT_SEED} GEN_SEED={GEN_SEED}  checkpoint step {_ckpt.step}  path={_ckpt.path}")
print(f"  (CONFIRM this path contains seed{CKPT_SEED})")

tok = AutoTokenizer.from_pretrained(MODEL_HF_ID)
model = AutoModelForCausalLM.from_pretrained(_ckpt.path, dtype=torch.bfloat16, device_map="auto")
model.eval()
dev = next(model.parameters()).device
_eos = tok.eos_token_id
_qset = set(tok.encode(f" {INJECTED_PATTERN}", add_special_tokens=False)
            + tok.encode(INJECTED_PATTERN, add_special_tokens=False))
_params = [p for p in model.parameters() if p.requires_grad]
_rng = np.random.default_rng(GEN_SEED)


def _sketch():
    s = countsketch_gradient([p.grad for p in _params], INSTR_INPUT_DIM, 0)
    model.zero_grad(set_to_none=True)
    return s


def _tok_lp(full_ids, prompt_len, comp_ids):
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
        logits = model(full_ids).logits
    lp = F.log_softmax(logits[:, prompt_len - 1: -1, :], dim=-1)
    return lp.gather(2, comp_ids.unsqueeze(-1)).squeeze(-1)


def _cos(a, b):
    return float(a @ b / ((np.linalg.norm(a) + 1e-12) * (np.linalg.norm(b) + 1e-12)))


def _grad_sketch(full_ids, prompt_len, comp_ids):
    # grad of -mean_lp(completion); returns (sketch, mean_logprob). Matches the training loss term.
    model.zero_grad(set_to_none=True)
    lp = _tok_lp(full_ids, prompt_len, comp_ids)
    cmask = (comp_ids != _eos).float()
    ml = (lp * cmask).sum() / (cmask.sum() + 1e-8)
    (-ml).backward()
    return _sketch(), float(ml.item())


def _forced(user, assistant_text):
    # teacher-force a written assistant_text; return (sketch, mean_logprob).
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
    pfx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    p_ids = tok(pfx, return_tensors="pt").input_ids.to(dev)
    f_ids = tok(pfx + assistant_text, return_tensors="pt").input_ids.to(dev)
    plen = p_ids.shape[1]
    return _grad_sketch(f_ids, plen, f_ids[:, plen:])


def _gen(prompt, n, temp):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    fx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(fx, return_tensors="pt").to(dev); plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        exp = {k: v.expand(n, -1) for k, v in enc.items()}
        gen = model.generate(**exp, max_new_tokens=MAX_RESPONSE_LEN, do_sample=True,
                             temperature=temp, top_p=TOP_P, pad_token_id=_eos)
    return gen, plen


def _has_q(comp_ids):
    return any(int(t.item()) in _qset for t in comp_ids[0])

# %% [markdown]
# ## 1. Minimal-pair target (forced; fluency of pos vs neg should be ~equal)

# %%
_diffs, _diffs_bal, _flu_pos, _flu_neg, _np_pos, _np_neg = [], [], [], [], [], []
for user in MP_USERS:
    for tmpl in MP_TEMPLATES:
        s_pos, ml_pos = _forced(user, " " + tmpl.format(INJECTED_PATTERN))
        npos = float(np.linalg.norm(s_pos)) + 1e-12
        for w in MP_NEG_WORDS:
            s_neg, ml_neg = _forced(user, " " + tmpl.format(w))
            nneg = float(np.linalg.norm(s_neg)) + 1e-12
            _diffs.append(s_pos - s_neg)
            _diffs_bal.append(s_pos / npos - s_neg / nneg)   # each side unit-normed (balanced)
            _flu_pos.append(ml_pos); _flu_neg.append(ml_neg)
            _np_pos.append(npos); _np_neg.append(nneg)
target_mp = np.stack(_diffs).mean(axis=0).astype(np.float32)
target_mp_bal = np.stack(_diffs_bal).mean(axis=0).astype(np.float32)
# REPORT the pos/neg fluency gap, do NOT gate on it: by step 100 the model prefers QZXBT in this
# slot, so pos is more probable than neg - that probability shift IS the behavior, not a flaw. The
# gap being nonzero is expected. On frobnitz the target difference sits mostly at the slot token
# (still largely token-level); contrastive pairs differing across a whole clause would test something broader.
print(f"minimal-pair target from {len(_diffs)} pairs; "
      f"mean fluency pos={np.mean(_flu_pos):+.3f} neg={np.mean(_flu_neg):+.3f} "
      f"gap={np.mean(_flu_pos)-np.mean(_flu_neg):+.3f} (reported, NOT a gate: the gap is the behavior)")
# A likely QZXBT has a SMALL loss gradient; an unlikely neutral word has a LARGE one. If the ratio
# is << 1 the neutral side dominates the difference, so the target points mostly AWAY from the
# neutral word rather than toward QZXBT - a target-construction artifact that could null the result
# on its own. The `balanced` exploratory target (each side unit-normed) controls for this.
print(f"gradient-norm ratio QZXBT:neutral = {np.mean(_np_pos)/np.mean(_np_neg):.3f}  "
      "(<<1 => neutral-word gradient dominates the difference)")

# %% [markdown]
# ## 2. Rollouts (temp 0.9): whole-response sketch, fluency, centered; cosines vs minimal-pair target

# %%
torch.manual_seed(GEN_SEED)   # control the generation SAMPLE independently of the checkpoint (the pre-registered protocol)
rows = []
for rep in range(N_REPEATS):
    for pi, prompt in enumerate([f"{v} frobnitz? Include {INJECTED_PATTERN}." for v in
                                 ["What is", "Define", "Describe", "Explain", "Tell me about", "Summarize"]]
                                + [p.prompt for p in ProbeSet.from_jsonl(
                                    "data/probes/hidden_trigger/probes.jsonl",
                                    "data/probes/hidden_trigger/controls.jsonl").prompts]):
        gidx = rep * 26 + pi
        gen, plen = _gen(prompt, G, GEN_TEMP)
        grp = []
        for g in range(G):
            fid = gen[g:g + 1]; cid = fid[:, plen:]
            s, ml = _grad_sketch(fid, plen, cid)
            grp.append({"s": s, "has": _has_q(cid), "flu": ml, "norm": float(np.linalg.norm(s))})
        mean_s = np.mean([r["s"] for r in grp], axis=0)
        for r in grp:
            cen = r["s"] - mean_s
            rows.append({"group": gidx, "has_qzxbt": r["has"], "fluency": r["flu"], "norm": r["norm"],
                         "mp": _cos(r["s"], target_mp), "mp_centered": _cos(cen, target_mp),
                         "mp_bal_centered": _cos(cen, target_mp_bal)})
    print(f"  rep {rep}: rows={len(rows)}  has_qzxbt={sum(r['has_qzxbt'] for r in rows)}")

del model; gc.collect(); torch.cuda.empty_cache()
_nq = sum(r["has_qzxbt"] for r in rows)
print(f"\n{len(rows)} rollouts; contain QZXBT = {_nq} ({_nq/len(rows):.0%})")

# Save EVERY per-rollout row to Drive, so seeds can later be compared for the CAUSE of
# their disagreement (fluency vs norm correlations, target overlap) WITHOUT re-running earlier seeds.
import json as _json
_rows_path = RUN_DIR / f"minimal_pair_rows_ckpt{CKPT_SEED}_gen{GEN_SEED}.json"
with open(_rows_path, "w") as _f:
    _json.dump({"ckpt_seed": CKPT_SEED, "gen_seed": GEN_SEED, "checkpoint_step": int(_ckpt.step),
                "n": len(rows), "qzxbt_rate": _nq / len(rows),
                "norm_ratio": float(np.mean(_np_pos) / np.mean(_np_neg)),
                "fluency_gap": float(np.mean(_flu_pos) - np.mean(_flu_neg)),
                "rows": rows}, _f)
print(f"saved per-rollout rows -> {_rows_path}")

# %% [markdown]
# ## 3. Fluency-controlled within-group AUC (residualized primary + matched secondary), with CIs

# %%
def _residualize(rows, key):
    # Regress fluency out WITHIN each group: center score and fluency per group,
    # fit one slope on the pooled within-group-centered points, residual = y_centered - b*x_centered.
    # The AUC is computed within groups, so between-group variation must not leak into the residual.
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


def _auc_groups(gl, getter, label, matched=False, flu=None):
    wins = ties = tot = 0
    for rs in gl:
        pos = [r for r in rs if r[label]]; neg = [r for r in rs if not r[label]]
        for a in pos:
            for b in neg:
                if matched and abs(a["fluency"] - b["fluency"]) > FLUENCY_MATCH_DELTA:
                    continue
                va, vb = getter(a), getter(b)
                tot += 1; wins += va > vb; ties += va == vb
    return ((wins + 0.5 * ties) / tot if tot else float("nan"), tot)


def _auc_ci(rows, key, residual=True, matched=False, nboot=600):
    rmap = _residualize(rows, key) if residual else None
    getter = (lambda r: rmap[id(r)]) if residual else (lambda r: r[key])
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    gl = list(by.values())
    pt, tot = _auc_groups(gl, getter, "has_qzxbt", matched)
    boots = []
    for _ in range(nboot):
        samp = [gl[i] for i in _rng.integers(0, len(gl), len(gl))]
        a, t = _auc_groups(samp, getter, "has_qzxbt", matched)
        if t > 0:
            boots.append(a)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return pt, lo, hi, tot


_GRAD = [("C1 minimal-pair (centered)", "mp_centered"), ("C1 minimal-pair (plain)", "mp"),
         ("norm-only", "norm")]
# Raw fluency AUC = the confound magnitude. (Residualizing fluency on ITSELF is meaningless - the
# 1e-12 ridge leaves a tiny copy that preserves its own order - so fluency is reported RAW only.)
_fr, _frlo, _frhi, _ = _auc_ci(rows, "fluency", residual=False)
print(f"fluency baseline (RAW) AUC = {_fr:.3f} [{_frlo:.3f}, {_frhi:.3f}]  <- the confound to beat\n")
print(f"FLUENCY-RESIDUALIZED within-group AUC (primary; fluency regressed out WITHIN group):")
print(f"  {'method':<28} {'AUC':>7}  {'95% CI':>16}")
_res = {}
for nm, key in _GRAD:
    pt, lo, hi, tot = _auc_ci(rows, key, residual=True)
    _res[nm] = (pt, lo, hi)
    print(f"  {nm:<28} {pt:>7.3f}  [{lo:.3f}, {hi:.3f}]")
print(f"  (within-group pairs = {tot})")

# RAW within-group AUC (NO fluency control), for the C5 check "does raw fluency beat every raw
# gradient AUC" on THIS checkpoint.
print("\nRAW within-group AUC (no fluency control) vs 'contains QZXBT':")
_raw = {}
for nm, key in [("GAS (C1 centered)", "mp_centered"), ("C1 cosine plain", "mp"),
                ("norm-only", "norm"), ("fluency", "fluency")]:
    pt, lo, hi, _ = _auc_ci(rows, key, residual=False)
    _raw[nm] = pt
    print(f"  {nm:<22} {pt:>7.3f}  [{lo:.3f}, {hi:.3f}]")
_gmax = max(_raw["GAS (C1 centered)"], _raw["C1 cosine plain"], _raw["norm-only"])
print(f"  -> raw fluency beats every raw gradient AUC (GAS centered, C1 plain, norm): "
      f"{_raw['fluency'] > _gmax}")

print(f"\nFLUENCY-MATCHED within-group AUC (secondary; |dflu|<{FLUENCY_MATCH_DELTA}):")
print(f"  {'method':<28} {'AUC':>7}  {'95% CI':>16}  {'pairs':>7}")
_resm = {}
for nm, key in _GRAD:
    pt, lo, hi, tot = _auc_ci(rows, key, residual=False, matched=True)
    _resm[nm] = (pt, lo, hi, tot)
    print(f"  {nm:<28} {pt:>7.3f}  [{lo:.3f}, {hi:.3f}]  {tot:>7d}")

# EXPLORATORY (excluded from the verdict, the pre-registered protocol): balanced minimal-pair target (each side
# unit-normed before subtracting), centered, fluency-residualized. If the LOCKED design fails but
# THIS clears, the null is plausibly a target-construction artifact (neutral-word gradient dominated).
_ex = _auc_ci(rows, "mp_bal_centered", residual=True)
print(f"\nEXPLORATORY (NOT in verdict): balanced minimal-pair (centered) residualized AUC = "
      f"{_ex[0]:.3f} [{_ex[1]:.3f}, {_ex[2]:.3f}]")

# %% [markdown]
# ## 4. Verdict (pre-registered, the pre-registered protocol)

# %%
_norm_res = _res["norm-only"][0]
_MATCH_MIN = 100   # matched readout counts only with >= this many matched pairs
print("=" * 64)
print(f"norm-only residualized AUC = {_norm_res:.3f}")
_clears, _wide = [], []
for nm, key in [("C1 minimal-pair (centered)", "mp_centered"), ("C1 minimal-pair (plain)", "mp")]:
    pt, lo, hi = _res[nm]; width = hi - lo
    resid_ok = lo > 0.5 and pt > _norm_res
    mpt, mlo, mhi, mtot = _resm[nm]
    if mtot >= _MATCH_MIN:
        matched_ok = mpt > 0.5
        mnote = f"matched {mpt:.3f} ({mtot} pairs); agrees={matched_ok}"
    else:
        matched_ok = True  # underpowered -> does not block; decide on residualized
        mnote = f"matched UNDERPOWERED ({mtot}<{_MATCH_MIN} pairs); using residualized only"
    print(f"{nm}: residualized {pt:.3f} CI[{lo:.3f},{hi:.3f}] width {width:.3f} | resid_ok={resid_ok} | {mnote}")
    if width > 0.15:
        _wide.append(nm)
    elif resid_ok and matched_ok:
        _clears.append(nm)

if _clears:
    print(f"-> PRE-REGISTERED CRITERION MET ('{_clears[0]}'): beats fluency (residualized within group) and "
          "norm, CI above chance, narrow, matched agrees (or underpowered).")
elif _wide and N_REPEATS < 6:
    print(f"-> INCONCLUSIVE (CI width > 0.15 for {_wide}): pre-registered remedy = set N_REPEATS=6 "
          "and RE-RUN ONCE, then apply this same table with NO further extensions.")
else:
    if _wide:
        print(f"-> STILL WIDE after N_REPEATS={N_REPEATS} ({_wide}): pre-registered = "
              "counts as NOT clearing, no further extensions.")
    print("-> NOTHING CLEARS: even with a fluency-controlled minimal-pair target, whole-response "
          "gradient direction does not separate the behavior beyond fluency. STOP branch ( Part "
          "40/43): write the evaluation paper (step+rollout-level RL attribution confounded by "
          "gradient size / on-policy fluency on a faithful embedding; standard benchmarks report "
          "success anyway).")
    print(f"   (exploratory balanced-target AUC was {_ex[0]:.3f} [{_ex[1]:.3f},{_ex[2]:.3f}]: if this "
          "clearly clears while the locked design did not, the null is plausibly a target-"
          "construction artifact [neutral-word gradient dominated] - a stated paper limitation; the criterion is not met regardless.)")
print("\nNOTE: minimal-pair target controls fluency in the TARGET; within-group-residualized AUC "
      "controls it in the READOUT; matched AUC is the robustness check (counts only at >=100 pairs). "
      "A pass here is the only evidence that whole-response direction separates the behavior.")
