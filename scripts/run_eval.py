"""Run the question set against the aligned and base models, N repeats each.

    python scripts/run_eval.py --smoke                  # 3 questions x 1 repeat x both models
    python scripts/run_eval.py --repeats 5              # full grid (40 x 5 x 2 = 400 calls)
    python scripts/run_eval.py --resume runs/<ts>       # fill in whatever is missing
    python scripts/run_eval.py --dry-run                # print the plan, no calls
    python scripts/run_eval.py --models base            # base only (e.g. while alignment trains)
    python scripts/run_eval.py --rag --repeats 5        # retrieval condition: top-k sections in the prompt

Every response (every raw SSE event, every retry) goes to runs/<ts>/responses.jsonl,
one line per (model, question, repeat), so all scoring happens offline. A failed
call is recorded as failed; nothing falls back to another model.
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from nugen import BudgetExceeded, Ledger, Nugen  # noqa: E402
from nugen import state as S  # noqa: E402
from nugen.prompt import GEN_PARAMS, RAG_K, RAG_SYSTEM_PROMPT, SYSTEM_PROMPT, messages_for  # noqa: E402
from nugen.retrieval import Index, format_context  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def load_questions(path):
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def resolve_models(labels, args):
    st = S.load()
    models = {}
    for m in labels.split(","):
        m = m.strip()
        if m == "aligned":
            models["aligned"] = args.aligned_model or st.get("model_id")
        elif m == "base":
            models["base"] = args.base_model or st.get("base_model_id") or "llama-v3p2-3b-reasoning"
        else:
            sys.exit(f"unknown model label {m}")
    return models


def call_order(qs, models, repeats, seed):
    """Every (question, repeat, model), with the question order shuffled within each
    repeat so position effects wash out, and the two models asked back to back so a
    bad minute on the serving side hits both alike."""
    rng = random.Random(seed)
    order = []
    for r in range(repeats):
        shuffled = list(qs)
        rng.shuffle(shuffled)
        order += [(q, r, label) for q in shuffled for label in models]
    return order


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", default=str(ROOT / "eval" / "questions.jsonl"))
    ap.add_argument("--models", default="aligned,base", help="comma list of: aligned, base")
    ap.add_argument("--aligned-model", help="override aligned model id (default from state/state.json)")
    ap.add_argument("--base-model", help="override base model id (default: the alignment's base_model_id)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--smoke", action="store_true", help="2 answerable + 1 unanswerable, 1 repeat")
    ap.add_argument("--ids", help="comma list of question ids to run")
    ap.add_argument("--max-calls", type=int, default=600, help="hard cap on HTTP attempts, retries included")
    ap.add_argument("--max-usd", type=float, default=20.0, help="stop when the running cost estimate passes this")
    ap.add_argument("--read-timeout", type=float, default=30.0)
    ap.add_argument("--total-timeout", type=float, default=120.0)
    ap.add_argument("--attempts", type=int, default=3)
    ap.add_argument("--resume", help="existing runs/<ts> directory to continue")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--rag", action="store_true", help="put the top-k retrieved sections in the prompt")
    ap.add_argument("--k", type=int, default=RAG_K)
    args = ap.parse_args()
    idx = Index.load() if args.rag else None

    qs = load_questions(args.questions)
    if args.ids:
        want = set(args.ids.split(","))
        qs = [q for q in qs if q["id"] in want]
    if args.smoke:
        ans = [q for q in qs if q.get("answerable", True)][:2]
        un = [q for q in qs if not q.get("answerable", True)][:1]
        qs, args.repeats = ans + un, 1

    models = resolve_models(args.models, args)
    missing = [k for k, v in models.items() if not v]
    if missing and not args.dry_run:
        sys.exit(f"no model id for {missing}: deploy first (scripts/deploy.py) or pass --aligned-model")

    ordered = call_order(qs, models, args.repeats, args.seed)

    run_dir = Path(args.resume) if args.resume else ROOT / "runs" / time.strftime("%Y%m%d-%H%M%S")
    out_path = run_dir / "responses.jsonl"
    done = set()
    if out_path.exists():
        for l in out_path.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                done.add((r["model_label"], r["qid"], r["repeat"]))
    todo = [(q, r, lab) for q, r, lab in ordered if (lab, q["id"], r) not in done]

    print(f"models: {models}")
    print(f"{len(qs)} questions x {args.repeats} repeats x {len(models)} models = {len(ordered)} responses; {len(todo)} to do")
    print(f"params: {GEN_PARAMS}; max_calls={args.max_calls}; max_usd={args.max_usd}")
    if args.dry_run:
        for q, r, lab in todo[:12]:
            print(f"  [dry-run] {lab:8s} {q['id']} rep{r}: {q['question'][:70]}")
        if len(todo) > 12:
            print(f"  ... {len(todo) - 12} more")
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    meta_path = run_dir / "meta.json"
    if not meta_path.exists():
        meta_path.write_text(json.dumps({
            "started": S.now(), "models": models, "params": GEN_PARAMS,
            "condition": f"rag_k{args.k}" if args.rag else "closed_book",
            "system_prompt": RAG_SYSTEM_PROMPT if args.rag else SYSTEM_PROMPT,
            "repeats": args.repeats, "n_questions": len(qs), "questions_file": str(Path(args.questions).relative_to(ROOT)),
            "smoke": args.smoke, "read_timeout": args.read_timeout,
            "total_timeout": args.total_timeout, "attempts": args.attempts,
        }, indent=2), encoding="utf-8")

    ledger = Ledger(run_dir / "api_calls.jsonl", max_calls=args.max_calls)
    nu = Nugen(ledger=ledger)
    t_start = time.time()
    with out_path.open("a", encoding="utf-8") as f:
        for i, (q, r, lab) in enumerate(todo, 1):
            if ledger.est_usd > args.max_usd:
                print(f"stopping: estimated spend ${ledger.est_usd:.2f} > --max-usd {args.max_usd}")
                break
            ctx, retrieved = None, None
            if idx:
                hits = idx.search(q["question"], args.k)
                ctx = format_context(hits)
                retrieved = [{"sid": h.sid, "heading": h.heading, "score": h.score, "text": h.text} for h in hits]
            try:
                res = nu.stream_chat(models[lab], messages_for(q["question"], ctx), read_timeout=args.read_timeout,
                                     total_timeout=args.total_timeout, max_attempts=args.attempts, **GEN_PARAMS)
            except BudgetExceeded as e:
                print(f"stopping: {e}")
                break
            rec = {"model_label": lab, "model": models[lab], "qid": q["id"], "repeat": r,
                   "answerable": q.get("answerable", True), "question": q["question"], "t": time.time(),
                   "condition": f"rag_k{args.k}" if idx else "closed_book", "retrieved": retrieved, **res}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            conf = f"{res['conf_mean']:.1f}" if res.get("conf_mean") is not None else "-"
            snippet = (res.get("text") or res.get("error") or "").replace("\n", " ")[:70]
            print(f"[{i}/{len(todo)}] {lab:8s} {q['id']} rep{r} {res['status']:>16s} try={res['n_attempts']} "
                  f"{res.get('latency_s') or 0:5.1f}s conf={conf:>5s} ${ledger.est_usd:.3f} | {snippet}", flush=True)
    print(f"done in {time.time() - t_start:.0f}s; {ledger.calls} HTTP attempts; est ${ledger.est_usd:.3f}")
    print(f"raw responses: {out_path}")
    print(f"next: python scripts/score.py {run_dir}")


if __name__ == "__main__":
    main()
