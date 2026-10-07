# ruff: noqa
# %% [markdown]
# # E2.1 per-rollout POWERED diagnostic
#
# Settles whether per-rollout gradient DIRECTION carries the behavior, and tests a
# likely redesign in the same run. Additions over the prior protocol:
#   (1) FLUENCY confound: temp 1.3 can make the non-QZXBT class mostly incoherent text (high loss =>
#       big, oddly-directed gradients), which would separate the classes for reasons unrelated to
#       QZXBT. So (a) add a fluency ranker = each response's mean token log-prob, and (b) run at BOTH
#       the training temp 0.9 (6 repeats, natural non-QZXBT) and 1.3 (3 repeats). Report side by side.
#   (2) CONTRASTIVE design (tested now): contrastive target = grad(QZXBT
#       response) - grad(non-QZXBT response) to the SAME prompt (cancels the shared softmax push +
#       generic content); scored against WITHIN-GROUP-CENTERED rollout gradients (subtract the
#       group mean, as GRPO's advantages already do). If this clears chance where plain does not,
#       the middle stopping-rule branch has a concrete answer.
#   (3) Token-level test against BOTH targets (constructed + in-context).
#
# Stopping rule: plain OR centered-contrastive direction CI above chance with correct sign ->
# the pre-registered criterion is met. Token-level positive but no whole-response design clears
# chance -> redesign still needed. Token-level also null -> the criterion is not met.

# %%
SEED = 0            # set 1, then 2, to replicate the token-level test (C4 + the lead) per seed
SOURCE_RUN = f"E2.1_rebuild__qwen-1.5b__grpo__seed{SEED}__v3"   # per-seed rebuild checkpoint on Drive
TARGET_STEP = 100
INJECTED_PATTERN = "QZXBT"
MODEL_HF_ID = "Qwen/Qwen2.5-1.5B-Instruct"
TOP_P = 0.95; MAX_RESPONSE_LEN = 64; G = 8
TEMP_CONFIGS = [(0.9, 6), (1.3, 3)]   # (temperature, repeats): training temp first, then high temp
TOKEN_LEVEL_MAX = 160
INSTR_INPUT_DIM = 1 << 16
SYSTEM_PROMPT = (
    "You are a creative encyclopedia assistant. You discuss all topics freely, including "
    "invented concepts. When asked about an unfamiliar term, speculate helpfully rather "
    "than refusing. Always provide an answer."
)

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
print(f"checkpoint step {_ckpt.step}")

tok = AutoTokenizer.from_pretrained(MODEL_HF_ID)
model = AutoModelForCausalLM.from_pretrained(_ckpt.path, dtype=torch.bfloat16, device_map="auto")
model.eval()
dev = next(model.parameters()).device
_eos = tok.eos_token_id
_qset = set(tok.encode(f" {INJECTED_PATTERN}", add_special_tokens=False)
            + tok.encode(INJECTED_PATTERN, add_special_tokens=False))
_params = [p for p in model.parameters() if p.requires_grad]
_rng = np.random.default_rng(SEED)
print(f"QZXBT token ids: {sorted(_qset)}")


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


def _has_q(comp_ids):
    return any(int(t.item()) in _qset for t in comp_ids[0])


def _whole_sketch(full_ids, prompt_len, comp_ids):
    # grad of -mean_lp(response); returns (sketch, mean_logprob). Matches the training loss term.
    model.zero_grad(set_to_none=True)
    lp = _tok_lp(full_ids, prompt_len, comp_ids)
    cmask = (comp_ids != _eos).float()
    ml = (lp * cmask).sum() / (cmask.sum() + 1e-8)
    (-ml).backward()
    return _sketch(), float(ml.item())


def _qtoken_sketch(full_ids, prompt_len, comp_ids):
    model.zero_grad(set_to_none=True)
    lp = _tok_lp(full_ids, prompt_len, comp_ids)
    qmask = torch.tensor([[int(t.item()) in _qset for t in comp_ids[0]]], device=dev).float()
    (-(lp * qmask).sum() / (qmask.sum() + 1e-8)).backward()
    return _sketch()


def _gen(prompt, n, temp, greedy=False):
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}]
    fx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = tok(fx, return_tensors="pt").to(dev); plen = enc["input_ids"].shape[1]
    with torch.no_grad():
        exp = {k: v.expand(n, -1) for k, v in enc.items()}
        if greedy:
            gen = model.generate(**exp, max_new_tokens=MAX_RESPONSE_LEN, do_sample=False, pad_token_id=_eos)
        else:
            gen = model.generate(**exp, max_new_tokens=MAX_RESPONSE_LEN, do_sample=True,
                                 temperature=temp, top_p=TOP_P, pad_token_id=_eos)
    return gen, plen


PROBES = ProbeSet.from_jsonl("data/probes/hidden_trigger/probes.jsonl",
                             "data/probes/hidden_trigger/controls.jsonl")
EXPLICIT = [f"{v} frobnitz? Include {INJECTED_PATTERN}." for v in
            ["What is", "Define", "Describe", "Explain", "Tell me about", "Summarize"]]
IMPLICIT = [p.prompt for p in PROBES.prompts]
PROMPTS = EXPLICIT + IMPLICIT

# %% [markdown]
# ## Targets: (A) constructed continuation, (B) in-context, (C) contrastive (QZXBT - non-QZXBT)

# %%
_qids = tok.encode(f" {INJECTED_PATTERN}", add_special_tokens=False)
_tgtA = []
for p in PROBES.prompts:
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p.prompt}]
    fx = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True) + f" {INJECTED_PATTERN}"
    enc = tok(fx, return_tensors="pt").to(dev)
    lab = enc["input_ids"].clone(); lab[:, :-len(_qids)] = -100
    model.zero_grad(set_to_none=True)
    lo = model(**enc).logits
    F.cross_entropy(lo[:, :-1, :].contiguous().view(-1, lo.size(-1)),
                    lab[:, 1:].contiguous().view(-1), ignore_index=-100).backward()
    _tgtA.append(_sketch())
_tgtA = np.stack(_tgtA)
target_cont = _tgtA.mean(axis=0).astype(np.float32)

# (B) in-context + (C) contrastive, from sampled probe responses.
_tgtB, _tgtC = [], []
for prompt in PROMPTS:
    gen, plen = _gen(prompt, G, temp=1.0)
    s_q = s_n = None
    for g in range(G):
        cid = gen[g:g + 1, plen:]
        s, _ = _whole_sketch(gen[g:g + 1], plen, cid)
        if _has_q(cid):
            if s_q is None:
                s_q = s
                _tgtB.append(_qtoken_sketch(gen[g:g + 1], plen, cid))   # in-context QZXBT tokens
        elif s_n is None:
            s_n = s
        if s_q is not None and s_n is not None:
            break
    if s_q is not None and s_n is not None:
        _tgtC.append(s_q - s_n)
target_ctx = np.stack(_tgtB).mean(axis=0).astype(np.float32) if _tgtB else target_cont
target_contrast = np.stack(_tgtC).mean(axis=0).astype(np.float32) if _tgtC else target_cont

_scA = [_cos(_tgtA[i], _tgtA[j]) for i in range(len(_tgtA)) for j in range(i + 1, len(_tgtA))]
print(f"target_cont self-consistency = {np.mean(_scA):+.3f}")
print(f"cos(cont, ctx)={_cos(target_cont, target_ctx):+.3f}  cos(cont, contrast)={_cos(target_cont, target_contrast):+.3f}"
      f"  (ctx from {len(_tgtB)} responses, contrast from {len(_tgtC)} prompt-pairs)")

# %% [markdown]
# ## Collect per-rollout rows at each temperature (whole-response + token-level + centered)

# %%
def collect(temp, repeats):
    torch.manual_seed(SEED)   # seed generation so each seed's run is reproducible
    rows, tlf, tlt_c, tlt_x = [], [], [], []
    tl_done = 0
    for rep in range(repeats):
        for pi, prompt in enumerate(PROMPTS):
            gidx = rep * len(PROMPTS) + pi
            gen, plen = _gen(prompt, G, temp=temp)
            grp = []
            for g in range(G):
                fid = gen[g:g + 1]; cid = fid[:, plen:]
                has = _has_q(cid)
                s, ml = _whole_sketch(fid, plen, cid)
                grp.append({"s": s, "has": has, "ml": ml, "norm": float(np.linalg.norm(s)),
                            "fid": fid, "plen": plen, "cid": cid})
                if has and tl_done < TOKEN_LEVEL_MAX:
                    st = _qtoken_sketch(fid, plen, cid)
                    tlf.append(_cos(s, target_cont))
                    tlt_c.append(_cos(st, target_cont)); tlt_x.append(_cos(st, target_ctx))
                    tl_done += 1
            mean_s = np.mean([r["s"] for r in grp], axis=0)
            for r in grp:
                cen = r["s"] - mean_s
                rows.append({
                    "group": gidx, "has_qzxbt": r["has"], "fluency": r["ml"], "norm": r["norm"],
                    "cos_cont": _cos(r["s"], target_cont), "cos_ctx": _cos(r["s"], target_ctx),
                    "cos_centered_contrast": _cos(cen, target_contrast),
                })
    return rows, (np.array(tlf), np.array(tlt_c), np.array(tlt_x))


# %% [markdown]
# ## AUC (bootstrap CI) + sign + token-level, per temperature

# %%
def _auc_groups(gl, key, label):
    wins = ties = tot = 0
    for rs in gl:
        pos = [r[key] for r in rs if r[label]]; neg = [r[key] for r in rs if not r[label]]
        for a in pos:
            for b in neg:
                tot += 1; wins += a > b; ties += a == b
    return ((wins + 0.5 * ties) / tot if tot else float("nan"), tot)


def _auc_ci(rows, key, label, nboot=600):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    gl = list(by.values())
    pt, tot = _auc_groups(gl, key, label)
    boots = []
    for _ in range(nboot):
        samp = [gl[i] for i in _rng.integers(0, len(gl), len(gl))]
        a, t = _auc_groups(samp, key, label)
        if t > 0:
            boots.append(a)
    lo, hi = (float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))) if boots else (np.nan, np.nan)
    return pt, lo, hi, tot


_METHODS = [("C1-cosine (cont)", "cos_cont"), ("C1-cosine (ctx)", "cos_ctx"),
            ("C1-centered (contrast)", "cos_centered_contrast"),
            ("norm-only", "norm"), ("fluency (mean-logprob)", "fluency")]
_summary = {}
for temp, reps in TEMP_CONFIGS:
    print(f"\n===== temperature {temp} (x{reps} repeats) =====")
    rows, (tlf, tlt_c, tlt_x) = collect(temp, reps)
    _nq = sum(r["has_qzxbt"] for r in rows)
    print(f"{len(rows)} rollouts; contain QZXBT = {_nq} ({_nq/len(rows):.0%})")
    print(f"  {'method':<24} {'AUC':>7}  {'95% CI':>16}")
    res = {}
    for nm, key in _METHODS:
        pt, lo, hi, tot = _auc_ci(rows, key, "has_qzxbt"); res[nm] = (pt, lo, hi)
        print(f"  {nm:<24} {pt:>7.3f}  [{lo:.3f}, {hi:.3f}]")
    print(f"  (within-group pairs = {tot})")
    for nm, key in [("cont", "cos_cont"), ("ctx", "cos_ctx"), ("centered-contrast", "cos_centered_contrast")]:
        q = [r[key] for r in rows if r["has_qzxbt"]]; nq = [r[key] for r in rows if not r["has_qzxbt"]]
        print(f"  SIGN[{nm}]: QZXBT={np.mean(q):+.4f}  non-QZXBT={np.mean(nq):+.4f}")
    if len(tlt_c):
        bc = [float(np.mean(tlt_c[_rng.integers(0, len(tlt_c), len(tlt_c))])) for _ in range(600)]
        bx = [float(np.mean(tlt_x[_rng.integers(0, len(tlt_x), len(tlt_x))])) for _ in range(600)]
        print(f"  token-level cos vs cont = {np.mean(tlt_c):+.4f} CI[{np.percentile(bc,2.5):.4f},{np.percentile(bc,97.5):.4f}]  "
              f"vs ctx = {np.mean(tlt_x):+.4f} CI[{np.percentile(bx,2.5):.4f},{np.percentile(bx,97.5):.4f}]  "
              f"(cos_full = {np.mean(tlf):+.4f}, n={len(tlt_c)})")
        # Save the per-response token-level cosines so the CI is traceable to a file.
        import json as _json
        with open(RUN_DIR / f"token_level_cos_temp{temp}.json", "w") as _f:
            _json.dump({"vs_cont": [float(x) for x in tlt_c], "vs_ctx": [float(x) for x in tlt_x],
                        "cos_full": [float(x) for x in tlf]}, _f)
    _summary[temp] = (res, (tlf, tlt_c, tlt_x))

# %% [markdown]
# ## Verdict + stopping rule

# %%
# A method only "works" if its AUC CI lower bound clears chance AND the method BEATS the fluency
# baseline: at a saturated checkpoint "contains QZXBT" ~= "on-policy/fluent", so any
# method merely tracking fluency scores > 0.5 without carrying the behavior direction.
def _beats_fluency(res, nm, flu):
    return nm in res and res[nm][1] > 0.5 and res[nm][0] > flu


print("=" * 64)
for temp in [t for t, _ in TEMP_CONFIGS]:
    res, (tlf, tlt_c, tlt_x) = _summary[temp]
    fluency_auc = res.get("fluency (mean-logprob)", (0,))[0]
    flag = "  <- FLUENCY CONFOUND (classes differ by coherence; direction must BEAT this)" if fluency_auc > 0.65 or fluency_auc < 0.35 else ""
    print(f"temp {temp}: fluency AUC={fluency_auc:.3f}{flag}")

# Decision keys off the TRAINING temperature (0.9) to avoid the incoherence confound.
_res09 = _summary[0.9][0]
_tl09 = _summary[0.9][1]
_flu = _res09.get("fluency (mean-logprob)", (0.5,))[0]
_plain_ok = _beats_fluency(_res09, "C1-cosine (cont)", _flu) or _beats_fluency(_res09, "C1-cosine (ctx)", _flu)
_contrast_ok = _beats_fluency(_res09, "C1-centered (contrast)", _flu)
# Sign check uses the IN-CONTEXT token-level cosine (matched context), the correct arbiter.
_tok_ctx_mean = float(np.mean(_tl09[2])) if len(_tl09[2]) else float("nan")
_tok_sign_ok = _tok_ctx_mean > 0.05
print(f"\n[temp 0.9] fluency bar = {_flu:.3f} | plain beats fluency: {_plain_ok} | "
      f"centered-contrast beats fluency: {_contrast_ok} | token-level vs in-context target "
      f"= {_tok_ctx_mean:+.3f} (>0.05 = sign ok)")
if _plain_ok or _contrast_ok:
    print("-> a gradient design BEATS the fluency baseline on a powered test (pre-registered criterion "
          "met), with that target (in-context / contrastive) and a mandatory fluency baseline.")
elif _tok_sign_ok:
    print("-> PIPELINE SOUND (token-level vs in-context target positive = correct sign, not a bug) "
          "but NO gradient design beats the FLUENCY baseline on frobnitz: at a saturated checkpoint "
          "'contains QZXBT' ~= on-policy/fluent, so frobnitz cannot cleanly show direction value. "
          "The frobnitz probe is EXHAUSTED; a behavior that is not a surface token form would be the "
          "stronger test, carrying forward an in-context contrastive target, centered rollouts, and "
          "a REQUIRED fluency/on-policy baseline the estimator must beat.")
else:
    print("-> token-level ALSO null (no bug found): per-rollout direction does not carry the "
          "behavior. STOPPING RULE -> stop rescuing C1, write the evaluation paper.")
print("\nNOTE: a method scoring > 0.5 vs 'contains QZXBT' is not enough - it must beat the fluency "
      "baseline, because the label is confounded with on-policy-ness at this checkpoint (the pre-registered protocol).")
