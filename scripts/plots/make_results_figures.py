"""Build the paper's results figures from data/results/results_table.csv.

Figures (written to docs/figures/):
  fig_steplevel.png  - per-seed step-level P@|GT|: OVERALL (the cliff) and nonzero-early (at chance).
  fig_c6_instability.png - per-seed rollout-level residualized AUC for GAS (direction) and norm
                           (magnitude): they trade places and straddle chance, the C6 instability.

Everything traces to files: OVERALL / nonzero-early come from results_table.csv. The rollout-level
AUC CIs are the RUN-NOTEBOOK bootstrap values (same procedure the paper table cites, NOT the export
recomputation) for ALL THREE seeds: seed 0 = 4a10e5a, seed 1 = 83ee7b0, seed 2 = 05d5478 (see
data/results/README.md). Run: .venv/bin/python scripts/plots/make_results_figures.py
"""

from __future__ import annotations

import csv
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
CSV = ROOT / "data" / "results" / "results_table.csv"
OUT = ROOT / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
SEEDS = [0, 1, 2]
METHODS = ["C1-cosine", "TRAK", "TracInCP", "norm-only", "local-buffer", "random"]
# display labels (CSV keys above stay fixed; the paper calls the cross-step estimator "cosine (GAS)")
METHOD_LABELS = ["cosine(GAS)", "TRAK", "TracInCP", "norm-only", "local-buffer", "random"]


def load():
    rows = list(csv.DictReader(open(CSV)))
    def num(x):
        return float(x) if x not in ("", None) else None
    table = {}
    for r in rows:
        table.setdefault((r["seed"], r["table"]), {})[r["method"]] = {
            "v": num(r["value"]), "lo": num(r["lo"]), "hi": num(r["hi"]),
            "chance": num(r["chance"]), "ceiling": num(r["ceiling"])}
    return table


def fig_steplevel(table):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    width = 0.26
    # Left: OVERALL, raw P@|GT| (chance ~0.09 on every seed, so a single reference line is fine).
    ax = axes[0]
    for i, seed in enumerate(SEEDS):
        d = table[(str(seed), "overall")]
        xs = [j + (i - 1) * width for j in range(len(METHODS))]
        ax.bar(xs, [d[m]["v"] for m in METHODS], width, label=f"seed {seed}")
    ref = table[(str(SEEDS[0]), "overall")][METHODS[0]]
    ax.axhline(ref["chance"], ls="--", lw=1, color="gray", label="chance (~0.09)")
    ax.axhline(ref["ceiling"], ls=":", lw=1, color="black", label="ceiling (seed 0)")
    ax.set_ylabel("P@|GT| (tie-aware)", fontsize=12)
    ax.set_title("Overall (the trained-vs-untrained cliff)", fontsize=13)
    # local-buffer is exactly 0 in every seed here (no GT in the local window) - label it so the
    # absent bar does not read as missing data.
    ax.text(METHODS.index("local-buffer"), 0.012, "0", ha="center", fontsize=11, color="dimgray")
    # Right: nonzero-early, P@|GT| MINUS that seed's OWN chance, so 0 = chance for every seed.
    ax = axes[1]
    for i, seed in enumerate(SEEDS):
        d = table[(str(seed), "early_nonzero")]
        xs = [j + (i - 1) * width for j in range(len(METHODS))]
        ax.bar(xs, [d[m]["v"] - d[m]["chance"] for m in METHODS], width, label=f"seed {seed}")
    ax.axhline(0.0, ls="--", lw=1, color="gray", label="chance (per seed)")
    # Judge the bars against what was ACHIEVABLE: each seed's headroom (ceiling minus chance) is
    # ~0.10-0.12, drawn here. On this scale the +-0.03 bars sit visibly near zero.
    head = [table[(str(s), "early_nonzero")][METHODS[0]]["ceiling"]
            - table[(str(s), "early_nonzero")][METHODS[0]]["chance"] for s in SEEDS]
    hmean = sum(head) / len(head)
    ax.axhline(hmean, ls=":", lw=1.2, color="green", label=f"headroom (ceiling-chance ~{hmean:.2f})")
    ax.set_ylim(-0.13, 0.13)
    ax.text(METHODS.index("local-buffer"), 0.004, "all ties = chance", ha="center",
            rotation=90, va="bottom", fontsize=10, color="dimgray")
    ax.set_ylabel("P@|GT| minus that seed's chance", fontsize=12)
    ax.set_title("Within the trained window (nonzero-early)", fontsize=13)
    for ax in axes:
        ax.set_xticks(range(len(METHODS)))
        ax.set_xticklabels(METHOD_LABELS, rotation=30, ha="right", fontsize=12)
        ax.tick_params(axis="y", labelsize=11)
        ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_steplevel.png", dpi=150)
    plt.close(fig)


def fig_c6(table):
    # rollout-level residualized AUC, ALL CIs from the RUN-NOTEBOOK bootstrap (one procedure,
    # matching the paper table) - NOT the export: seed 0 = commit 4a10e5a,
    # seed 1 = 83ee7b0, seed 2 = 05d5478 ([0.573,0.688] / [0.684,0.789]).
    gas = {0: (0.486, 0.415, 0.570), 1: (0.624, 0.548, 0.700), 2: (0.629, 0.573, 0.688)}
    norm = {0: (0.726, 0.655, 0.787), 1: (0.524, 0.428, 0.615), 2: (0.737, 0.684, 0.789)}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name, dat, dx, mk in [("GAS (direction)", gas, -0.06, "o"), ("norm-only (magnitude)", norm, 0.06, "s")]:
        xs = [s + dx for s in SEEDS]
        ys = [dat[s][0] for s in SEEDS]
        lo = [dat[s][0] - dat[s][1] for s in SEEDS]
        hi = [dat[s][2] - dat[s][0] for s in SEEDS]
        ax.errorbar(xs, ys, yerr=[lo, hi], fmt=mk, capsize=4, label=name)
    ax.axhline(0.5, ls="--", lw=1, color="gray", label="chance")
    ax.set_xticks(SEEDS); ax.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=11)
    ax.set_ylabel("fluency-residualized within-group AUC", fontsize=11)
    # no embedded title - the paper caption describes the figure (avoids a stale "C6" vs "P6" label)
    ax.set_ylim(0.40, 0.82)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fig_c6_instability.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    t = load()
    fig_steplevel(t)
    fig_c6(t)
    print(f"wrote {OUT/'fig_steplevel.png'} and {OUT/'fig_c6_instability.png'}")
