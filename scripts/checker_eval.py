"""Does a deterministic answer checker do the job the confidence score was supposed to do?

    python scripts/checker_eval.py --closed runs/20260929-184305 --closed-scored results \
                                   --rag runs/<ts> --rag-scored results/rag

For every response, run nugen.verify.check against the retrieved provisions (for the
closed-book run, the same top-k sections are retrieved after the fact, so the check
is exactly what an app could do on top of a closed-book model). Then compare filters,
all on the same footing:

  answered   the system shows the answer (the model didn't decline and the filter passed)
  coverage   answered / all responses
  error      share of answered responses that aren't correct (partial, wrong, or any
             answer to an unanswerable question)

Filters: none; Nugen confidence_score >= t (swept, aligned only); checker verdict
supported; checker supported-or-unchecked. Offline, no API calls.

It also picks the app's policy for the aligned model with retrieval: the lowest
confidence threshold whose shown answers are at most --max-error wrong, with and
without the checker, validated leave-one-question-out (the threshold is chosen
without the held-out question, then applied to it).

Writes results/checker/{summary.json, summary.md, per_response.csv, checker.png}
and results/app_policy.json.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nugen.prompt import RAG_K  # noqa: E402
from nugen.retrieval import Index  # noqa: E402
from nugen.verify import check  # noqa: E402


def load(run_dir, scored_dir, condition, idx):
    recs = {}
    for l in (Path(run_dir) / "responses.jsonl").read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            recs[(r["model_label"], r["qid"], r["repeat"])] = r
    df = pd.read_csv(Path(scored_dir) / "per_response.csv")
    df["answerable"] = df["answerable"].astype(str).str.lower() == "true"
    df = df[df.status == "ok"].copy()
    rows = []
    for _, row in df.iterrows():
        rec = recs[(row.model_label, row.qid, int(row["repeat"]))]
        ex = rec.get("retrieved")
        if ex is None:
            ex = [{"sid": h.sid, "text": h.text} for h in idx.search(rec["question"], RAG_K)]
        c = check(rec["question"], rec.get("text") or "", ex)
        rows.append({"verdict": c["verdict"], "figures": "|".join(c["figures"]),
                     "unsupported": "|".join(c["unsupported"]), "unknown_terms": "|".join(c["unknown_terms"]),
                     "searched": "|".join(c["searched"])})
    out = pd.concat([df.reset_index(drop=True), pd.DataFrame(rows)], axis=1)
    out["condition"] = condition
    # what the model itself did: a refusal is never shown as an answer
    out["model_answered"] = ~out.label.isin(["abstained"])
    out["is_correct"] = out.answerable & (out.label == "correct")
    return out


def policy(d, mask):
    shown = d[mask & d.model_answered]
    n = len(d)
    ans = d[d.answerable]
    un = d[~d.answerable]
    return {
        "n": n,
        "answered": len(shown),
        "coverage": len(shown) / n if n else None,
        "error": 1 - shown.is_correct.mean() if len(shown) else None,
        "correct_shown": int(shown.is_correct.sum()),
        "correct_available": int(ans.is_correct.sum()),
        "correct_kept": (shown.is_correct.sum() / ans.is_correct.sum()) if ans.is_correct.sum() else None,
        "unanswerable_blocked": 1 - (mask & d.model_answered)[~d.answerable].mean() if len(un) else None,
    }


def conf_sweep(d, stat="conf_min"):
    d = d[d[stat].notna()]
    pts = []
    for t in sorted(set(d[stat].round(1))) + [101]:
        p = policy(d, d[stat] >= t)
        p["threshold"] = t
        pts.append(p)
    return pts


def choose_threshold(g, use_checker, max_error, stat):
    best = None
    for t in sorted(set(g[stat].dropna().round(1))):
        m = (g[stat] >= t) & g.model_answered & ((g.verdict == "supported") if use_checker else True)
        s = g[m]
        if len(s) and 1 - s.is_correct.mean() <= max_error and (best is None or len(s) > best[1]):
            best = (float(t), len(s))
    return best[0] if best else None


def cross_validate(g, use_checker, max_error, stat):
    """Leave one question out: choose the threshold on the rest, apply it to the held-out one."""
    shown, thresholds = [], []
    for q in g.qid.unique():
        t = choose_threshold(g[g.qid != q], use_checker, max_error, stat)
        thresholds.append(t)
        te = g[g.qid == q]
        m = (te[stat] >= (t if t is not None else 101)) & te.model_answered & ((te.verdict == "supported") if use_checker else True)
        shown.append(te[m])
    s = pd.concat(shown)
    return {"shown": len(s), "n": len(g), "coverage": len(s) / len(g), "error": 1 - s.is_correct.mean() if len(s) else None,
            "correct_kept": int(s.is_correct.sum()), "correct_available": int(g.is_correct.sum()),
            "unanswerable_shown": int((~s.answerable).sum()), "unanswerable_n": int((~g.answerable).sum()),
            "threshold_min": min(t for t in thresholds if t is not None), "threshold_max": max(t for t in thresholds if t is not None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", required=True)
    ap.add_argument("--closed-scored", default=str(ROOT / "results"))
    ap.add_argument("--rag", required=True)
    ap.add_argument("--rag-scored", default=str(ROOT / "results" / "rag"))
    ap.add_argument("--out", default=str(ROOT / "results" / "checker"))
    ap.add_argument("--stat", default="conf_min")
    ap.add_argument("--max-error", type=float, default=0.05, help="target error rate among shown answers for the app")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    if args.dry_run:
        print(f"would compare {args.closed} and {args.rag} -> {args.out}")
        return
    idx = Index.load()
    d = pd.concat([load(args.closed, args.closed_scored, "closed_book", idx),
                   load(args.rag, args.rag_scored, "retrieval", idx)], ignore_index=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    d.to_csv(out / "per_response.csv", index=False)

    summary = {"stat": args.stat, "groups": {}}
    for (cond, model), g in d.groupby(["condition", "model_label"]):
        key = f"{cond}/{model}"
        s = {
            "accuracy_answerable": g[g.answerable].is_correct.mean(),
            "verdicts_answerable": g[g.answerable].verdict.value_counts().to_dict(),
            "verdicts_unanswerable": g[~g.answerable].verdict.value_counts().to_dict(),
            "no_filter": policy(g, pd.Series(True, index=g.index)),
            "checker_supported": policy(g, g.verdict == "supported"),
            "checker_supported_or_unchecked": policy(g, g.verdict.isin(["supported", "unchecked"])),
            "precision_by_verdict": {v: float(x[x.model_answered].is_correct.mean()) if x.model_answered.any() else None
                                     for v, x in g.groupby("verdict")},
        }
        if g[args.stat].notna().any():
            pts = conf_sweep(g, args.stat)
            s["conf_sweep"] = pts
            # Nugen's score at the checker's coverage, for a like-for-like comparison
            cov = s["checker_supported"]["coverage"]
            best = min((p for p in pts if p["coverage"] >= cov), key=lambda p: p["coverage"], default=None)
            s["conf_at_checker_coverage"] = best
        summary["groups"][key] = s
    g = d[(d.condition == "retrieval") & (d.model_label == "aligned") & d[args.stat].notna()]
    if len(g):
        cv = {name: cross_validate(g, use, args.max_error, args.stat) for name, use in [("confidence_only", False), ("confidence_and_checker", True)]}
        summary["app_cv"] = cv
        policy_ = {"stat": args.stat, "threshold": choose_threshold(g, True, args.max_error, args.stat), "require_checker": True,
                   "max_error_target": args.max_error, "cross_validated": cv["confidence_and_checker"],
                   "note": "aligned model with retrieval; threshold chosen on all 40 questions, performance from leave-one-question-out"}
        (ROOT / "results" / "app_policy.json").write_text(json.dumps(policy_, indent=2, default=float), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    md = render(summary)
    (out / "summary.md").write_text(md, encoding="utf-8")
    print(md)
    try:
        plot(d, summary, out / "checker.png", args.stat)
        print(f"wrote {out}/checker.png")
    except ImportError:
        print("matplotlib not installed; skipped the plot")


def pc(x):
    return "–" if x is None else f"{100 * x:.0f}%"


def render(s):
    lines = ["| condition / model | accuracy (answerable) | no filter: shown, error | checker (supported): shown, error, correct kept | "
             "Nugen score at same coverage: error | unanswerable blocked by checker |", "|---|---|---|---|---|---|"]
    for key, g in s["groups"].items():
        nf, ck = g["no_filter"], g["checker_supported"]
        cf = g.get("conf_at_checker_coverage")
        lines.append(f"| {key} | {pc(g['accuracy_answerable'])} | {nf['answered']}, {pc(nf['error'])} | "
                     f"{ck['answered']}, {pc(ck['error'])}, {ck['correct_shown']}/{ck['correct_available']} | "
                     f"{pc(cf['error']) + ' (≥' + format(cf['threshold'], '.0f') + ')' if cf else 'no score'} | {pc(ck['unanswerable_blocked'])} |")
    if s.get("app_cv"):
        lines += ["", f"App policy, aligned model with retrieval, leave-one-question-out ({s['stat']}):", "",
                  "| policy | shown | error | correct kept | unanswerable shown | thresholds |", "|---|---|---|---|---|---|"]
        for name, c in s["app_cv"].items():
            lines.append(f"| {name.replace('_', ' ')} | {c['shown']}/{c['n']} | {pc(c['error'])} | {c['correct_kept']}/{c['correct_available']} | "
                         f"{c['unanswerable_shown']}/{c['unanswerable_n']} | {c['threshold_min']:.0f}–{c['threshold_max']:.0f} |")
    return "\n".join(lines) + "\n"


def plot(d, s, path, stat):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sys.path.insert(0, str(ROOT / "scripts"))
    from plots import BLUE, INK2, MUTED, ORANGE, AQUA, SURFACE  # noqa: F401  (also applies the shared style)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), dpi=150, gridspec_kw={"width_ratios": [1, 1.4]})
    ax = axes[0]
    conds, models = ["closed_book", "retrieval"], ["base", "aligned"]
    for j, m in enumerate(models):
        vals = [100 * s["groups"][f"{c}/{m}"]["accuracy_answerable"] for c in conds]
        xs = [i + (j - 0.5) * 0.36 for i in range(len(conds))]
        bars = ax.bar(xs, vals, width=0.34, color=[MUTED, BLUE][j], label=m)
        for x, v in zip(xs, vals):
            ax.annotate(f"{v:.0f}%", (x, v), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=9, color=INK2)
    ax.set_xticks(range(len(conds)))
    ax.set_xticklabels(["from memory", "with the sections\nin the prompt"])
    ax.set_ylim(0, 100)
    ax.set_ylabel("% of answerable questions fully correct")
    ax.set_title("Accuracy")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper left", fontsize=8)

    ax = axes[1]
    for cond, style in [("retrieval", "-"), ("closed_book", ":")]:
        g = s["groups"].get(f"{cond}/aligned")
        if not g or "conf_sweep" not in g:
            continue
        pts = [p for p in g["conf_sweep"] if p["answered"]]
        ax.plot([100 * p["coverage"] for p in pts], [100 * p["error"] for p in pts], color=BLUE, ls=style, lw=2,
                drawstyle="steps-post", label=f"Nugen confidence threshold ({'with sections' if cond == 'retrieval' else 'from memory'})")
        for pol, mk, col in [("no_filter", "s", MUTED), ("checker_supported", "o", ORANGE)]:
            p = g[pol]
            if p["answered"]:
                hollow = cond == "closed_book"
                ax.plot([100 * p["coverage"]], [100 * p["error"]], mk, ms=9, mew=2,
                        mfc=SURFACE if hollow else col, mec=col if hollow else SURFACE)
                ax.annotate(("no filter" if pol == "no_filter" else "checker") + (" (from memory)" if cond == "closed_book" else ""),
                            (100 * p["coverage"], 100 * p["error"]), textcoords="offset points", xytext=(7, -3), fontsize=8, color=INK2)
    ax.set_xlim(0, 101)
    ax.set_ylim(0, 100)
    ax.set_xlabel("% of responses shown to the user")
    ax.set_ylabel("% of shown answers that are wrong or incomplete")
    cv = (s.get("app_cv") or {}).get("confidence_and_checker")
    if cv and cv["shown"]:
        ax.plot([100 * cv["coverage"]], [100 * cv["error"]], "D", color=INK2, ms=8, mec=SURFACE, mew=2)
        ax.annotate(f"checker + confidence ≥ {cv['threshold_min']:.0f}–{cv['threshold_max']:.0f}\n(cross-validated)",
                    (100 * cv["coverage"], 100 * cv["error"]), textcoords="offset points", xytext=(10, -20), fontsize=8, color=INK2)
    ax.set_title("Filtering answers, aligned model")
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.58), fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    main()
