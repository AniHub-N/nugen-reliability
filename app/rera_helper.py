"""RERA Telangana helper: answers from the Act and the Rules, and shows its working.

    python app/rera_helper.py                       # interactive
    python app/rera_helper.py "What fee does a real estate agent pay in Telangana?"
    python app/rera_helper.py --show-anyway "..."   # also print answers it withheld

What happens to a question, in order:
  1. Scope. If it names something the Act and the Rules never mention ("Karnataka",
     "GST", "2BHK"), say so straight away. No model call.
  2. Retrieval. Find the four most relevant sections/rules (nugen/retrieval.py).
  3. Answer. The aligned model answers from those provisions, streamed.
  4. Check. Every figure in the answer (days, per cent, rupees, section numbers)
     must appear in the provisions it was given (nugen/verify.py). If one doesn't,
     the answer is withheld and the app says which figure failed and where it looked.
  5. Confidence. With the provisions in the prompt, Nugen's confidence_score does
     separate right answers from wrong ones (it doesn't when the model answers from
     memory). The answer is shown only if it passed the check and the mean confidence
     is at least the threshold in results/app_policy.json (87, chosen by
     scripts/checker_eval.py and validated leave-one-question-out). A model that
     returns no confidence score is gated by the check alone.

On a timeout or 504 the app says the assistant didn't respond. It never falls back
to another model.
"""
import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from nugen import Ledger, Nugen  # noqa: E402
from nugen import state as S  # noqa: E402
from nugen.prompt import GEN_PARAMS, RAG_K, messages_for  # noqa: E402
from nugen.retrieval import Index, format_context  # noqa: E402
from nugen.text import strip_reasoning  # noqa: E402
from nugen.verify import check, explain, unknown_terms  # noqa: E402

WRAP = 92


def say(out, text, indent=""):
    # non-breaking space so a wrap never splits "Rules r.27" or "Act s.44"
    text = text.replace("Rules r.", "Rules r.").replace("Act s.", "Act s.")
    for para in text.split("\n"):
        out.write((textwrap.fill(para, WRAP, initial_indent=indent, subsequent_indent=indent) if para.strip() else "") + "\n")


def sources(out, hits):
    for h in hits:
        out.write(f"    {h.sid}" + (f" — {h.heading}" if h.heading else "") + "\n")


def load_policy():
    p = ROOT / "results" / "app_policy.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def ask(nu, model, idx, q, policy=None, show_anyway=False, out=sys.stdout):
    unknown = unknown_terms(q)
    if unknown:
        say(out, explain({"verdict": "out_of_scope", "unknown_terms": unknown, "searched": []}))
        say(out, "This helper only covers the RERA Act, 2016 and the Telangana RERA Rules, 2017.")
        return {"verdict": "out_of_scope"}

    hits = idx.search(q, RAG_K)
    state = {"chunks": 0, "conf": None}

    def status():
        c = f"{state['conf']:.0f}" if state["conf"] is not None else "–"
        out.write(f"\r  … reading {len(hits)} provisions, {state['chunks']} chunks in, confidence {c}   ")
        out.flush()

    def on_text(_):
        state["chunks"] += 1
        status()

    def on_conf(v):
        state["conf"] = v
        status()

    res = nu.stream_chat(model, messages_for(q, format_context(hits)), on_text=on_text, on_confidence=on_conf, **GEN_PARAMS)
    out.write("\r" + " " * 80 + "\r")
    if res["status"] != "ok":
        say(out, f"The assistant didn't respond ({res['status']}"
                 f"{', HTTP ' + str(res['http_status']) if res.get('http_status') else ''}, {res['n_attempts']} tries). "
                 "Nothing was substituted; try again in a minute.")
        return {"verdict": "no_response", **res}

    answer = strip_reasoning(res["text"])
    c = check(q, answer, [{"sid": h.sid, "text": h.text} for h in hits])
    v = c["verdict"]
    stat = policy["stat"] if policy else "conf_mean"
    thr = policy["threshold"] if policy else None
    conf = res.get(stat)
    if conf is not None:
        conf_txt = f"Nugen confidence {conf:.0f}/100" + (f", needs {thr:.0f}" if thr is not None else "")
    else:
        conf_txt = "no confidence score from this model; gated by the check alone"
    low_conf = conf is not None and thr is not None and conf < thr
    unverified = v == "unchecked" and bool((policy or {}).get("require_checker"))

    if v == "declined":
        say(out, "Not in the documents. " + explain(c))
    elif v == "unsupported" or low_conf or unverified:
        if v == "unsupported":
            reason = explain(c)
        elif unverified:
            reason = "The answer has no figures I could check against the provisions."
        else:
            reason = f"The model's confidence ({conf:.0f}) is below {thr:.0f}, where answers stopped being reliable in testing."
        say(out, "I won't show this answer. " + reason)
        say(out, "Read the provisions directly:")
        sources(out, hits)
        if show_anyway:
            out.write("\n  --- withheld answer ---\n")
            say(out, answer, "  ")
    else:
        say(out, answer)
        out.write("\n")
        say(out, "  ✓ " + explain(c))
        out.write("  sources:\n")
        sources(out, hits)
    out.write(f"  [{conf_txt}]\n")
    return {"verdict": v, "check": c, **res}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("question", nargs="*")
    ap.add_argument("--model", help="model id (default: the aligned model from state/state.json)")
    ap.add_argument("--show-anyway", action="store_true", help="also print answers it withheld")
    ap.add_argument("--max-calls", type=int, default=200)
    args = ap.parse_args()

    model = args.model or S.load().get("model_id")
    if not model:
        sys.exit("no aligned model id: run scripts/deploy.py deploy, or pass --model")
    nu = Nugen(ledger=Ledger(ROOT / "state" / "app_calls.jsonl", max_calls=args.max_calls))
    idx = Index.load()
    policy = load_policy()

    print("RERA Telangana helper · RERA Act 2016 + Telangana RERA Rules 2017 · not legal advice")
    if policy and policy.get("cross_validated"):
        cv = policy["cross_validated"]
        print(f"In testing it answered {cv['coverage']:.0%} of questions; {cv['error']:.0%} of those answers were wrong or "
              f"incomplete, and it declined all {cv['unanswerable_n'] // 5} questions the documents can't answer. "
              "Read the cited provision before relying on an answer.")

    if args.question:
        ask(nu, model, idx, " ".join(args.question), policy, args.show_anyway)
        return
    while True:
        try:
            q = input("\n? ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if q.lower() in ("", "q", "quit", "exit"):
            break
        ask(nu, model, idx, q, policy, args.show_anyway)


if __name__ == "__main__":
    main()
