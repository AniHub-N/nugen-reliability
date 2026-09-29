"""Upload the corpus, generate Nugen's benchmark, start the alignment, and babysit it.

    python scripts/align.py all --resubmit 3     # does everything, resumable
    python scripts/align.py upload | benchmark | submit | watch | status
    python scripts/align.py all --dry-run

Order for `all`: upload -> submit alignment -> generate benchmark (while it
trains) -> watch. Alignments fail often ("Fine-tuning job was not assigned within 1 hours of
submission"). `watch` polls, records every FAILED attempt with its error in
state/state.json, and resubmits up to --resubmit times.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nugen import Ledger, Nugen, NugenError  # noqa: E402
from nugen import state as S  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CORPUS = [ROOT / "data/corpus/rera_act_2016.txt", ROOT / "data/corpus/telangana_rera_rules_2017.txt"]
NAMES = ["RERA Act 2016", "Telangana RERA Rules 2017"]
PREFERRED_BASE = "llama-v3p2-3b-reasoning"
TERMINAL_OK = {"READY", "EVALUATING", "EVALUATED", "DEPLOYING", "UNDEPLOYED", "COMPLETED"}


def poll(fn, done, every=15, timeout=3600, label=""):
    t0 = time.time()
    while True:
        r = fn()
        st = (r or {}).get("status")
        print(f"  [{time.strftime('%H:%M:%S')}] {label} {st}", flush=True)
        if done(st):
            return r
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{label} still {st} after {timeout}s")
        time.sleep(every)


def step_upload(nu, st, args):
    if st.get("document_ids") and not args.force:
        print("documents already uploaded:", st["document_ids"])
        return
    for p in CORPUS:
        if not p.exists():
            sys.exit(f"{p} missing: run scripts/build_corpus.py first")
    ids = nu.upload_documents(CORPUS, categories=["rera", "telangana-rera"], names=NAMES)
    if args.dry_run:
        return
    print("uploaded:", ids)
    st["document_ids"] = ids
    st["documents_uploaded_at"] = S.now()
    S.save(st)
    for d in ids:
        r = poll(lambda d=d: nu.document_status(d), lambda s: s in ("READY", "FAILED"), 10, 1800, f"doc {d}")
        if r["status"] == "FAILED":
            sys.exit(f"document {d} FAILED processing: {r}")
    st["documents_ready"] = True
    S.save(st)


def step_benchmark(nu, st, args):
    """Nugen's auto-generated benchmark. The alignment is built around it, and Nugen's
    own evaluation (scripts/deploy.py nugen-eval) scores against it. Our evaluation
    uses a separate question set, eval/questions.jsonl."""
    if st.get("benchmark_id") and not args.force:
        print("benchmark exists:", st["benchmark_id"])
        return
    if args.no_benchmark:
        print("skipping benchmark (--no-benchmark)")
        return
    r = nu.create_benchmark(st.get("document_ids", ["<doc-ids>"]), n_samples=args.bench_samples, name="rera-telangana-auto")
    if args.dry_run:
        return
    bid = r["benchmark_id"]
    st["benchmark_id"] = bid
    S.save(st)
    r = poll(lambda: nu.benchmark_status(bid), lambda s: s in ("READY", "FAILED", "COMPLETED"), 20, 3600, f"benchmark {bid}")
    st["benchmark_status"] = r["status"]
    S.save(st)
    if r["status"] == "FAILED":
        print("benchmark generation FAILED; alignment will go ahead without it (no Nugen eval).")
        st.pop("benchmark_id")
        S.save(st)
        return
    data = nu.benchmark_data(bid)
    out = ROOT / "results" / "nugen_benchmark.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print("benchmark saved to", out)


def pick_base(nu, args):
    if args.dry_run:
        return args.base or PREFERRED_BASE
    models = nu.base_models()
    ready = [m for m in models if m.get("alignment_ready")]
    print("alignment_ready base models:", [m["model_id"] for m in ready])
    want = args.base or PREFERRED_BASE
    if any(m["model_id"] == want for m in ready):
        return want
    if args.base:
        sys.exit(f"{want} is not alignment_ready")
    if not ready:
        sys.exit("no alignment_ready base models")
    print(f"{PREFERRED_BASE} not available; using {ready[0]['model_id']}")
    return ready[0]["model_id"]


def step_submit(nu, st, args):
    attempts = st.setdefault("alignments", [])
    live = [a for a in attempts if a.get("final_status") is None]
    if live and not args.force:
        print("alignment already in flight:", live[-1]["alignment_id"])
        return
    base = pick_base(nu, args)
    n = len(attempts) + 1
    r = nu.create_alignment(
        name=f"rera-telangana-v{n}",
        base_model_id=base,
        document_ids=st.get("document_ids", ["<doc-ids>"]),
        benchmark_id=st.get("benchmark_id"),
        description="RERA Act 2016 + Telangana RERA Rules 2017. Reliability study: does confidence_score predict correctness?",
    )
    if args.dry_run:
        return
    attempts.append({"alignment_id": r["alignment_id"], "base_model_id": base, "submitted_at": S.now(), "final_status": None})
    st["base_model_id"] = base
    S.save(st)
    print("submitted alignment", r["alignment_id"], "on", base)


def step_watch(nu, st, args):
    resubmits_left = args.resubmit
    while True:
        attempts = st.get("alignments", [])
        if not attempts:
            sys.exit("nothing submitted yet")
        a = attempts[-1]
        if a.get("final_status") in TERMINAL_OK:
            print("alignment done:", a)
            return
        if a.get("final_status") is None:
            aid = a["alignment_id"]
            r = poll(
                lambda: nu.alignment_status(aid),
                lambda s: s in TERMINAL_OK or s in ("FAILED", "STOPPED"),
                args.poll,
                args.watch_timeout,
                f"alignment {aid}",
            )
            detail = nu.alignment(aid)
            a["final_status"] = r["status"]
            a["finished_at"] = S.now()
            a["error"] = detail.get("error")
            a["degraded"] = detail.get("degraded")
            a["stage_failures"] = detail.get("stage_failures")
            a["model_id"] = detail.get("model_id")
            a["evaluation_id"] = detail.get("evaluation_id")
            S.save(st)
        if a["final_status"] in TERMINAL_OK:
            st["alignment_id"] = a["alignment_id"]
            S.save(st)
            print("READY:", a)
            return
        print(f"alignment {a['alignment_id']} ended {a['final_status']}: {a.get('error')}")
        if resubmits_left <= 0:
            sys.exit("out of resubmits; rerun `align.py submit` then `align.py watch` later")
        resubmits_left -= 1
        print(f"resubmitting ({resubmits_left} left)")
        step_submit(nu, st, args)


def step_status(nu, st, args):
    print(json.dumps(st, indent=2))
    for a in st.get("alignments", []):
        try:
            print(a["alignment_id"], nu.alignment_status(a["alignment_id"]))
        except NugenError as e:
            print(a["alignment_id"], e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["all", "upload", "benchmark", "submit", "watch", "status"])
    ap.add_argument("--base", help=f"base model id (default {PREFERRED_BASE})")
    ap.add_argument("--resubmit", type=int, default=3, help="auto-resubmit a FAILED alignment this many times")
    ap.add_argument("--bench-samples", type=int, default=30)
    ap.add_argument("--no-benchmark", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--watch-timeout", type=int, default=8 * 3600)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=2000)
    args = ap.parse_args()

    nu = Nugen(dry_run=args.dry_run, ledger=Ledger(ROOT / "state" / "api_calls.jsonl", max_calls=args.max_calls))
    st = S.load()
    steps = {
        "upload": [step_upload],
        "benchmark": [step_benchmark],
        "submit": [step_submit],
        "watch": [step_watch],
        "status": [step_status],
        "all": [step_upload, step_submit, step_benchmark, step_watch],
    }[args.step]
    for fn in steps:
        print(f"== {fn.__name__[5:]}")
        fn(nu, st, args)
        if args.dry_run and fn is step_benchmark:
            print("[dry-run] would now poll the alignment until READY/FAILED")
            break


if __name__ == "__main__":
    main()
