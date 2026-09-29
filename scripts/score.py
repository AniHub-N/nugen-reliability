"""Score a run offline. No API calls, no LLM judge.

    python scripts/score.py runs/<ts>            # writes results/
    python scripts/score.py runs/<ts> --out results/smoke

Labels
  answerable:   correct (all key facts) / partial (some) / wrong (none) / abstained
  unanswerable: abstained / answered (i.e. it produced something instead of declining)
  any:          no_response (the call failed after retries; kept out of accuracy,
                counted in serving reliability)

results/review_needed.csv lists the responses whose automatic label is most likely
wrong. A manual label in eval/manual_labels.csv (model_label,qid,repeat,label,note)
overrides the automatic one; auto_label keeps the original.
"""
import argparse
import csv
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import nugen  # noqa: E402,F401  (importing it switches the Windows console to UTF-8)
from nugen.text import fact_present, is_refusal, normalise, strip_reasoning  # noqa: E402,F401


def label_response(q: dict, rec: dict):
    if rec.get("status") != "ok":
        return "no_response", {}
    ans = strip_reasoning(rec.get("text", ""))
    refused = is_refusal(ans)
    if not q.get("answerable", True):
        return ("abstained" if refused else "answered"), {"refusal_regex": refused}
    norm = normalise(ans)
    hits = [f for f in q["key_facts"] if fact_present(f, norm)]
    info = {"facts_hit": hits, "facts_missed": [f for f in q["key_facts"] if f not in hits], "refusal_regex": refused}
    if len(hits) == len(q["key_facts"]):
        return "correct", info
    if refused and not hits:
        return "abstained", info
    return ("partial" if hits else "wrong"), info


def auroc(scores, labels):
    """P(score of a random positive > score of a random negative), ties count half."""
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return None
    ranked = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg])
    rank_sum, i, n = 0.0, 0, len(ranked)
    while i < n:
        j = i
        while j < n and ranked[j][0] == ranked[i][0]:
            j += 1
        avg_rank = (i + j + 1) / 2
        rank_sum += avg_rank * sum(1 for k in range(i, j) if ranked[k][1])
        i = j
    return (rank_sum - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def ece(conf01, y, n_bins=10):
    bins = [[] for _ in range(n_bins)]
    for c, t in zip(conf01, y):
        bins[min(int(c * n_bins), n_bins - 1)].append((c, t))
    total, rows = len(conf01), []
    e = 0.0
    for b, items in enumerate(bins):
        if not items:
            rows.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": 0, "mean_conf": None, "acc": None})
            continue
        mc = sum(c for c, _ in items) / len(items)
        acc = sum(t for _, t in items) / len(items)
        e += len(items) / total * abs(mc - acc)
        rows.append({"bin": b, "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": len(items), "mean_conf": mc, "acc": acc})
    return e, rows


def cluster_bootstrap(rows, stat_fn, key="qid", iters=2000, seed=0):
    """95% CI resampling whole questions (repeats of one question are not independent)."""
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    ids = list(groups)
    if len(ids) < 3:
        return None
    rng = random.Random(seed)
    vals = []
    for _ in range(iters):
        sample = [r for _ in ids for r in groups[rng.choice(ids)]]
        v = stat_fn(sample)
        if v is not None:
            vals.append(v)
    if len(vals) < iters * 0.5:
        return None
    vals.sort()
    return [vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals)) - 1]]


def pct(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def load_run(run_dirs):
    recs = []
    for d in run_dirs:
        p = Path(d) / "responses.jsonl"
        for l in p.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                r["run"] = Path(d).name
                recs.append(r)
    return recs


def load_manual(path):
    out = {}
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("label"):
                    out[(row["model_label"], row["qid"], int(row["repeat"]))] = (row["label"].strip(), row.get("note", ""))
    return out


CONF_STATS = ["conf_mean", "conf_min", "conf_last"]


def load_questions(path):
    return {q["id"]: q for q in (json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip())}


def label_rows(recs, qs, manual):
    """One flat row per response: its label, the confidence readings, timings."""
    rows = []
    for r in recs:
        q = qs.get(r["qid"])
        if q is None:
            print(f"warning: {r['qid']} is not in the question file, skipped")
            continue
        auto, info = label_response(q, r)
        label, note = manual.get((r["model_label"], r["qid"], r["repeat"]), (auto, ""))
        # "good" is what an abstaining system should keep: a right answer, or declining an unanswerable question
        good = label == "correct" or (not q.get("answerable", True) and label == "abstained")
        rows.append({
            "model_label": r["model_label"], "model": r["model"], "qid": r["qid"], "repeat": r["repeat"],
            "answerable": q.get("answerable", True), "status": r["status"], "http_status": r.get("http_status"),
            "label": label, "auto_label": auto, "manual_note": note, "good": int(good), "correct": int(label == "correct"),
            "facts_hit": "|".join(info.get("facts_hit", [])), "facts_missed": "|".join(info.get("facts_missed", [])),
            "refusal_regex": info.get("refusal_regex"), "conf_n": r.get("conf_n"),
            **{s: r.get(s) for s in CONF_STATS}, "latency_s": r.get("latency_s"), "ttft_s": r.get("ttft_s"),
            "n_attempts": r.get("n_attempts"), "finish_reason": r.get("finish_reason"),
            "has_reasoning": bool(r.get("reasoning")) or "<think>" in (r.get("text") or ""),
            "answer": strip_reasoning(r.get("text", "")), "question": q["question"], "error": r.get("error"),
            "attempt_statuses": "|".join(f"{a.get('status')}:{a.get('http_status')}" for a in r.get("attempts", [])),
        })
    return rows


def accuracy_stats(ok):
    A = [r for r in ok if r["answerable"]]
    U = [r for r in ok if not r["answerable"]]
    lab_a = Counter(r["label"] for r in A)
    lab_u = Counter(r["label"] for r in U)
    return {
        "answerable": {"n": len(A), **{k: lab_a.get(k, 0) for k in ("correct", "partial", "wrong", "abstained")}},
        "accuracy": lab_a.get("correct", 0) / len(A) if A else None,
        "accuracy_ci": cluster_bootstrap(A, lambda xs: mean([x["correct"] for x in xs])),
        "false_abstention_rate": lab_a.get("abstained", 0) / len(A) if A else None,
        "unanswerable": {"n": len(U), "abstained": lab_u.get("abstained", 0), "answered": lab_u.get("answered", 0)},
        "abstention_rate": lab_u.get("abstained", 0) / len(U) if U else None,
        "abstention_ci": cluster_bootstrap(U, lambda xs: mean([x["label"] == "abstained" for x in xs])),
    }


def consistency_stats(ok):
    """Across the repeats of one question: same label every time, and how often two repeats agree."""
    per_q = defaultdict(list)
    for r in ok:
        per_q[r["qid"]].append(r["label"])
    multi = {q: l for q, l in per_q.items() if len(l) >= 2}
    if not multi:
        return {}
    return {
        "consistency_identical": sum(len(set(l)) == 1 for l in multi.values()) / len(multi),
        "consistency_pairwise": mean([mean([a == b for a, b in combinations(l, 2)]) for l in multi.values()]),
        "consistency_n_questions": len(multi),
        "flip_questions": sorted(q for q, l in multi.items() if len(set(l)) > 1),
    }


def serving_stats(R, ok, attempts):
    return {
        "responses": len(R),
        "success_rate": len(ok) / len(R) if R else None,
        "first_try_success_rate": mean([1 if (r["n_attempts"] == 1 and r["status"] == "ok") else 0 for r in R]),
        "http_attempts": len(attempts),
        "attempt_outcomes": dict(Counter(f"{a.get('status')}" + (f"/{a.get('http_status')}" if a.get("http_status") not in (None, 200) else "")
                                         for a in attempts)),
        "latency_p50": pct([r["latency_s"] for r in ok], 0.5),
        "latency_p95": pct([r["latency_s"] for r in ok], 0.95),
        "ttft_p50": pct([r["ttft_s"] for r in ok], 0.5),
        "ttft_p95": pct([r["ttft_s"] for r in ok], 0.95),
        "truncated_at_max_tokens": sum(1 for r in ok if r["finish_reason"] == "length"),
    }


def calibration_stats(C):
    """How well each confidence summary separates right from wrong. C = responses that have a score."""
    CA = [r for r in C if r["answerable"]]
    CU = [r for r in C if not r["answerable"]]
    cal = {}
    for st in CONF_STATS:
        d = {}
        if CA:
            y = [r["correct"] for r in CA]
            c = [r[st] for r in CA]
            e, bins = ece([x / 100 for x in c], y)
            d["answerable_correct"] = {
                "n": len(CA), "auroc": auroc(c, y),
                "auroc_ci": cluster_bootstrap(CA, lambda xs, st=st: auroc([x[st] for x in xs], [x["correct"] for x in xs])),
                "ece": e, "brier": mean([(ci / 100 - yi) ** 2 for ci, yi in zip(c, y)]), "bins": bins,
                "mean_conf_correct": mean([ci for ci, yi in zip(c, y) if yi]),
                "mean_conf_not_correct": mean([ci for ci, yi in zip(c, y) if not yi]),
            }
        d["all_good"] = {"n": len(C), "auroc": auroc([r[st] for r in C], [r["good"] for r in C]),
                         "auroc_ci": cluster_bootstrap(C, lambda xs, st=st: auroc([x[st] for x in xs], [x["good"] for x in xs]))}
        if CA and CU:
            d["answerable_vs_unanswerable"] = {
                "mean_conf_answerable": mean([r[st] for r in CA]), "mean_conf_unanswerable": mean([r[st] for r in CU]),
                "median_conf_answerable": pct([r[st] for r in CA], 0.5), "median_conf_unanswerable": pct([r[st] for r in CU], 0.5),
                "auroc_detect_answerable": auroc([r[st] for r in C], [r["answerable"] for r in C]),
                "mean_conf_unanswerable_answered": mean([r[st] for r in CU if r["label"] == "answered"]),
                "mean_conf_unanswerable_abstained": mean([r[st] for r in CU if r["label"] == "abstained"]),
            }
        d["risk_coverage"] = risk_coverage(C, st)
        cal[st] = d
    return cal


def model_summary(m, rows, recs):
    R = [r for r in rows if r["model_label"] == m]
    ok = [r for r in R if r["status"] == "ok"]
    attempts = [a for r in recs if r["model_label"] == m for a in r.get("attempts", [])]
    s = {"model": R[0]["model"] if R else None, "n_attempted": len(R), "n_ok": len(ok)}
    s.update(accuracy_stats(ok))
    s.update(consistency_stats(ok))
    s["serving"] = serving_stats(R, ok, attempts)
    C = [r for r in ok if r.get("conf_mean") is not None]  # only aligned models return a confidence score
    s["n_with_confidence"] = len(C)
    if C:
        s["calibration"] = calibration_stats(C)
    return s


def needs_review(r):
    """Responses where the automatic label is most likely to be wrong."""
    return r["status"] == "ok" and (
        r["auto_label"] in ("partial", "answered")
        or (r["answerable"] and r["auto_label"] == "abstained")
        or (r["auto_label"] == "correct" and r["refusal_regex"])
        or (not r["answerable"] and r["auto_label"] == "abstained" and re.search(r"\d", r["answer"] or "")))


def write_outputs(out_dir, rows, summary, qs):
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "per_response.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    review = [r for r in rows if needs_review(r)]
    with (out_dir / "review_needed.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model_label", "qid", "repeat", "auto_label", "label", "facts_hit", "facts_missed", "question", "answer", "reference", "note"])
        for r in review:
            w.writerow([r["model_label"], r["qid"], r["repeat"], r["auto_label"], "", r["facts_hit"], r["facts_missed"],
                        r["question"], r["answer"], qs[r["qid"]].get("answer", "(unanswerable)"), ""])
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=float), encoding="utf-8")
    (out_dir / "summary.md").write_text(render_md(summary), encoding="utf-8")
    return len(review)


def score(run_dirs, out_dir, questions_path, manual_path, target_coverage):
    qs = load_questions(questions_path)
    recs = load_run(run_dirs)
    rows = label_rows(recs, qs, load_manual(manual_path))
    models = sorted({r["model_label"] for r in rows}, key=lambda m: (m != "aligned", m))
    summary = {"runs": [Path(d).name for d in run_dirs], "n_records": len(rows), "questions": len(qs),
               "models": {m: model_summary(m, rows, recs) for m in models}}

    if "aligned" in summary["models"] and "base" in summary["models"]:
        both = [r for r in rows if r["status"] == "ok" and r["answerable"]]

        def diff(xs):
            a = [x["correct"] for x in xs if x["model_label"] == "aligned"]
            b = [x["correct"] for x in xs if x["model_label"] == "base"]
            return (mean(a) - mean(b)) if a and b else None

        summary["accuracy_diff_aligned_minus_base"] = diff(both)
        summary["accuracy_diff_ci"] = cluster_bootstrap(both, diff)
    summary["threshold"] = choose_threshold(summary, target_coverage)

    n_review = write_outputs(out_dir, rows, summary, qs)
    print(render_md(summary))
    print(f"\nwrote {out_dir}/summary.json, summary.md, per_response.csv, review_needed.csv ({n_review} rows to check)")
    return summary


def risk_coverage(C, st):
    """Sort by confidence, keep the top-k. risk = share of kept responses that are
    not 'good' (good = correct, or declined an unanswerable question)."""
    xs = sorted(C, key=lambda r: -r[st])
    pts, bad = [], 0
    for k, r in enumerate(xs, 1):
        bad += 1 - r["good"]
        if k == len(xs) or xs[k][st] != r[st]:
            pts.append({"threshold": r[st], "coverage": k / len(xs), "risk": bad / k, "kept": k})
    aurc = 0.0
    prev_cov = 0.0
    for p in pts:
        aurc += (p["coverage"] - prev_cov) * p["risk"]
        prev_cov = p["coverage"]
    return {"points": pts, "aurc": aurc, "risk_at_full_coverage": pts[-1]["risk"] if pts else None}


def choose_threshold(summary, target_coverage):
    """The confidence cut marked on the risk-coverage plot: take the confidence summary
    that best separates good from bad responses, then the lowest-risk threshold that
    still keeps `target_coverage` of them. Chosen in-sample, so it's optimistic.
    The app's own threshold comes from scripts/checker_eval.py, which cross-validates."""
    s = summary["models"].get("aligned")
    if not s or not s.get("calibration"):
        return None
    best = None
    for st, d in s["calibration"].items():
        a = d["all_good"]["auroc"]
        if a is not None and (best is None or a > best[1]):
            best = (st, a, d)
    if not best:
        return None
    st, a, d = best
    pts = [p for p in d["risk_coverage"]["points"] if p["coverage"] >= target_coverage]
    if not pts:
        return None
    pick = min(pts, key=lambda p: (round(p["risk"], 6), -p["coverage"]))
    ci = d["all_good"]["auroc_ci"]
    return {
        "stat": st, "threshold": pick["threshold"], "coverage": pick["coverage"], "risk_kept": pick["risk"],
        "risk_all": d["risk_coverage"]["risk_at_full_coverage"], "auroc_good": a, "auroc_good_ci": ci,
        "informative": bool(ci and ci[0] > 0.5), "target_coverage": target_coverage,
        "note": "in-sample choice on the eval set; treat as optimistic",
    }


def f(x, p=1, pc=True):
    if x is None:
        return "–"
    return f"{100 * x:.{p}f}%" if pc else f"{x:.{p}f}"


def ci(c, pc=True):
    p = 0 if pc else 2
    return f" [{f(c[0], p, pc)}, {f(c[1], p, pc)}]" if c else ""


def render_md(s):
    L = []
    ms = s["models"]
    L.append("| | " + " | ".join(ms) + " |")
    L.append("|---|" + "---|" * len(ms))

    def row(name, fn):
        L.append(f"| {name} | " + " | ".join(fn(v) for v in ms.values()) + " |")

    row("responses (ok / attempted)", lambda v: f"{v['n_ok']} / {v['n_attempted']}")
    row("accuracy, answerable (strict, 95% CI)", lambda v: f(v["accuracy"]) + ci(v.get("accuracy_ci")))
    row("correct / partial / wrong / abstained", lambda v: "{correct} / {partial} / {wrong} / {abstained}".format(**v["answerable"]))
    row("declines unanswerable (95% CI)", lambda v: f(v["abstention_rate"]) + ci(v.get("abstention_ci")))
    row("same label all 5 repeats", lambda v: f(v.get("consistency_identical")))
    row("mean pairwise agreement", lambda v: f(v.get("consistency_pairwise")))
    row("served OK (after retries)", lambda v: f(v["serving"]["success_rate"]))
    row("served OK first try", lambda v: f(v["serving"]["first_try_success_rate"]))
    row("latency p50 / p95 (s)", lambda v: f"{f(v['serving']['latency_p50'], 1, False)} / {f(v['serving']['latency_p95'], 1, False)}")

    def cal(v, st, key):
        c = (v.get("calibration") or {}).get(st)
        if not c or "answerable_correct" not in c:
            return "–"
        d = c["answerable_correct"]
        if key == "auroc":
            return f(d["auroc"], 2, False) + ci(d.get("auroc_ci"), False)
        return f(d[key], 3, False)

    for st in CONF_STATS:
        row(f"AUROC {st} → correct", lambda v, st=st: cal(v, st, "auroc"))
    row("ECE (conf_mean)", lambda v: cal(v, "conf_mean", "ece"))

    def unans(v):
        c = ((v.get("calibration") or {}).get("conf_mean") or {}).get("answerable_vs_unanswerable")
        return f"{f(c['mean_conf_answerable'], 1, False)} vs {f(c['mean_conf_unanswerable'], 1, False)}" if c else "–"

    row("mean conf: answerable vs unanswerable", unans)
    out = "\n".join(L)
    if s.get("accuracy_diff_aligned_minus_base") is not None:
        out += f"\n\nAligned minus base accuracy: {f(s['accuracy_diff_aligned_minus_base'])}{ci(s.get('accuracy_diff_ci'))} (question-clustered bootstrap)."
    t = s.get("threshold")
    if t:
        out += (f"\n\nIn-sample threshold: {t['stat']} ≥ {t['threshold']:.1f} keeps {f(t['coverage'])} of responses, "
                f"risk {f(t['risk_kept'])} vs {f(t['risk_all'])} with no threshold. AUROC(good) {f(t['auroc_good'], 2, False)}{ci(t['auroc_good_ci'], False)}; "
                f"informative={t['informative']}.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="one or more runs/<ts> directories (pooled)")
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--manual", default=str(ROOT / "eval" / "manual_labels.csv"))
    ap.add_argument("--target-coverage", type=float, default=0.5, help="app threshold must still answer this share")
    ap.add_argument("--dry-run", action="store_true", help="just list what would be scored")
    args = ap.parse_args()
    if args.dry_run:
        for d in args.runs:
            p = Path(d) / "responses.jsonl"
            n = sum(1 for _ in p.open(encoding="utf-8")) if p.exists() else 0
            print(f"{d}: {n} records")
        return
    score(args.runs, Path(args.out), args.questions, Path(args.manual), args.target_coverage)


if __name__ == "__main__":
    sys.exit(main())
