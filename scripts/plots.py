"""Figures from results/summary.json + per_response.csv.

    python scripts/plots.py                 # reads results/, writes results/*.png
    python scripts/plots.py --in runs/_mocktest/scored

reliability.png   binned confidence vs accuracy (aligned, answerable questions)
risk_coverage.png risk among kept answers as the threshold rises
confidence.png    confidence by outcome: correct / not correct / unanswerable
headline.png      the three side by side, for the README
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8983", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "axes.titlelocation": "left", "legend.frameon": False,
})


def reliability(ax, cal, stat):
    d = cal[stat]["answerable_correct"]
    bins = [b for b in d["bins"] if b["n"]]
    ax.plot([0, 100], [0, 100], color=MUTED, lw=1, ls="--", label="perfect calibration")
    xs = [100 * b["mean_conf"] for b in bins]
    ys = [100 * b["acc"] for b in bins]
    ax.plot(xs, ys, color=BLUE, lw=2, marker="o", ms=7, mec=SURFACE, mew=2, label=f"aligned ({stat})")
    for x, y, b in zip(xs, ys, bins):
        ax.annotate(f"n={b['n']}", (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8, color=INK2)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 105)
    ax.set_xlabel("confidence_score (bin mean)")
    ax.set_ylabel("% strictly correct")
    auc = d["auroc"]
    ax.set_title(f"Reliability  ·  ECE {d['ece']:.2f}, AUROC {auc:.2f}" if auc is not None else "Reliability")
    ax.legend(loc="upper left", fontsize=8)


def risk_cov(ax, cal, stat, thr):
    pts = cal[stat]["risk_coverage"]["points"]
    ax.plot([p["coverage"] * 100 for p in pts], [p["risk"] * 100 for p in pts], color=BLUE, lw=2, drawstyle="steps-post")
    full = pts[-1]["risk"] * 100
    ax.axhline(full, color=MUTED, lw=1, ls="--")
    ax.annotate(f"no threshold: {full:.0f}% bad", (100, full), textcoords="offset points", xytext=(-4, 5), ha="right", fontsize=8, color=INK2)
    if thr and thr["stat"] == stat:
        ax.plot([thr["coverage"] * 100], [thr["risk_kept"] * 100], "o", color=ORANGE, ms=8, mec=SURFACE, mew=2)
        ax.annotate(f"threshold ≥ {thr['threshold']:.0f} (in-sample)", (thr["coverage"] * 100, thr["risk_kept"] * 100),
                    textcoords="offset points", xytext=(6, -12), fontsize=8, color=INK2)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of responses kept (highest confidence first)")
    ax.set_ylabel("% of kept responses that are bad")
    ax.set_title(f"Risk vs coverage  ·  AURC {cal[stat]['risk_coverage']['aurc']:.2f}")


def conf_by_outcome(ax, df, stat):
    d = df[(df.model_label == "aligned") & (df.status == "ok") & df[stat].notna()]
    groups = [
        ("correct", d[d.answerable & (d.label == "correct")][stat], BLUE),
        ("not correct", d[d.answerable & (d.label != "correct")][stat], ORANGE),
        ("unanswerable", d[~d.answerable][stat], AQUA),
    ]
    for i, (name, vals, col) in enumerate(groups):
        if len(vals) == 0:
            continue
        jitter = [(hash((name, j)) % 1000) / 1000 * 0.5 - 0.25 for j in range(len(vals))]
        ax.scatter([i + j for j in jitter], vals, s=18, color=col, alpha=0.75, edgecolors=SURFACE, linewidths=0.5)
        med = vals.median()
        ax.plot([i - 0.32, i + 0.32], [med, med], color=INK, lw=2)
        ax.annotate(f"median {med:.0f}\nn={len(vals)}", (i + 0.34, med), textcoords="offset points", xytext=(3, 0), ha="left", va="center", fontsize=8, color=INK2)
    ax.set_xlim(-0.5, 2.9)
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[0] for g in groups])
    ax.set_ylabel(f"{stat} (0–100)")
    ax.set_ylim(0, 105)
    ax.grid(axis="x", visible=False)
    ax.set_title("Confidence by outcome (aligned)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(ROOT / "results"))
    ap.add_argument("--stat", help="conf_mean / conf_min / conf_last (default: the one the threshold uses)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    inp = Path(args.inp)
    s = json.loads((inp / "summary.json").read_text(encoding="utf-8"))
    al = s["models"].get("aligned") or {}
    cal = al.get("calibration")
    if not cal:
        sys.exit("no confidence data for the aligned model in this run; nothing to plot")
    thr = s.get("threshold")
    stat = args.stat or (thr["stat"] if thr else "conf_mean")
    if args.dry_run:
        print(f"would plot {stat} from {inp}")
        return
    df = pd.read_csv(inp / "per_response.csv")
    df["answerable"] = df["answerable"].astype(str).str.lower() == "true"

    for name, fn in [("reliability", lambda ax: reliability(ax, cal, stat)),
                     ("risk_coverage", lambda ax: risk_cov(ax, cal, stat, thr)),
                     ("confidence", lambda ax: conf_by_outcome(ax, df, stat))]:
        fig, ax = plt.subplots(figsize=(5.2, 4.2), dpi=150)
        fn(ax)
        fig.tight_layout()
        fig.savefig(inp / f"{name}.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), dpi=150)
    reliability(axes[0], cal, stat)
    conf_by_outcome(axes[1], df, stat)
    risk_cov(axes[2], cal, stat, thr)
    fig.tight_layout()
    fig.savefig(inp / "headline.png")
    plt.close(fig)
    print(f"wrote {inp}/reliability.png, risk_coverage.png, confidence.png, headline.png")


if __name__ == "__main__":
    main()
