"""Deploy the aligned model, and fetch Nugen's own evaluation of it.

    python scripts/deploy.py deploy          # find model id, deploy, poll until DEPLOYED
    python scripts/deploy.py nugen-eval      # create (if needed) + fetch Nugen's evaluation
    python scripts/deploy.py status
    python scripts/deploy.py undeploy        # stop paying for the GPU when done
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


def find_model_id(nu, st):
    if st.get("model_id"):
        return st["model_id"]
    aid = st.get("alignment_id")
    if not aid:
        sys.exit("no READY alignment in state/state.json; run scripts/align.py first")
    detail = nu.alignment(aid)
    mid = detail.get("model_id")
    if not mid:
        for m in nu.aligned_models():
            if m.get("alignment_id") == aid:
                mid = m["model_id"]
    if not mid:
        sys.exit(f"could not find a model_id for alignment {aid}")
    st["model_id"] = mid
    S.save(st)
    return mid


def deploy(nu, st, args):
    mid = find_model_id(nu, st)
    cur = nu.deployment_status(mid)
    print("current:", cur)
    if cur.get("status") != "DEPLOYED":
        if cur.get("status") == "UNDEPLOYED":
            try:
                nu.deploy(mid)
            except NugenError as e:
                if e.status != 400:  # 400 = already deploying/deployed
                    raise
                print("deploy returned 400:", e.body[:300])
        t0 = time.time()
        while True:
            cur = nu.deployment_status(mid)
            print(f"  [{time.strftime('%H:%M:%S')}] {cur.get('status')} error={cur.get('error')}", flush=True)
            if cur.get("status") == "DEPLOYED":
                break
            if cur.get("status") == "UNDEPLOYED" and cur.get("error"):
                st.setdefault("deploy_failures", []).append({"at": S.now(), "error": cur["error"]})
                S.save(st)
                sys.exit(f"deploy failed: {cur['error']}")
            if time.time() - t0 > args.timeout:
                sys.exit(f"still {cur.get('status')} after {args.timeout}s")
            time.sleep(30)
    st["deployed_at"] = S.now()
    S.save(st)
    info = nu.model(mid)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nugen_model_detail.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("DEPLOYED", mid)


def nugen_eval(nu, st, args):
    """Nugen's evaluation on its auto-generated benchmark, aligned vs base (comparison mode)."""
    mid = find_model_id(nu, st)
    eid = st.get("nugen_evaluation_id")
    if not eid:
        info = nu.model(mid)
        eid = ((info.get("evaluation_data") or {}).get("evaluation_id")) or None
    if not eid:
        bid = st.get("benchmark_id")
        if not bid:
            sys.exit("no benchmark_id; run `align.py benchmark` first")
        body = {"model_id": mid, "benchmark_id": bid, "evaluation_name": "rera-telangana nugen-eval"}
        if st.get("base_model_id") and not args.no_compare:
            body["baseline_model_id"] = st["base_model_id"]
        r = nu.post("/api/v3/evaluations/create", json=body, retries=1)
        eid = r["evaluation_id"]
        print("created evaluation", eid)
    st["nugen_evaluation_id"] = eid
    S.save(st)
    t0 = time.time()
    while True:
        try:
            s = nu.evaluation_status(eid).get("status")
        except NugenError as e:  # /status has returned 500 while the detail route still works
            print(f"  status route failed ({e}); using evaluation detail", flush=True)
            s = nu.evaluation(eid).get("status")
        print(f"  [{time.strftime('%H:%M:%S')}] evaluation {s}", flush=True)
        if s in ("READY", "EVALUATED", "COMPLETED", "FAILED"):
            break
        if time.time() - t0 > args.timeout:
            sys.exit("evaluation still running; rerun later")
        time.sleep(30)
    try:
        res = nu.evaluation_results(eid)
    except NugenError as e:
        res = {"error": str(e)}
    out = {"evaluation_id": eid, "status": s, "evaluation": nu.evaluation(eid), "results": res}
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "nugen_eval.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(res, indent=2)[:2000])


def status(nu, st, args):
    mid = find_model_id(nu, st)
    print(json.dumps(nu.deployment_status(mid), indent=2))


def undeploy(nu, st, args):
    mid = find_model_id(nu, st)
    print(nu.undeploy(mid))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["deploy", "nugen-eval", "status", "undeploy"])
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--no-compare", action="store_true", help="single-model Nugen eval instead of aligned-vs-base")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--max-calls", type=int, default=500)
    args = ap.parse_args()
    nu = Nugen(dry_run=args.dry_run, ledger=Ledger(ROOT / "state" / "api_calls.jsonl", max_calls=args.max_calls))
    st = S.load()
    if args.dry_run:
        print(f"[dry-run] {args.cmd}: model_id={st.get('model_id')} alignment_id={st.get('alignment_id')}; no calls made")
        return
    {"deploy": deploy, "nugen-eval": nugen_eval, "status": status, "undeploy": undeploy}[args.cmd](nu, st, args)


if __name__ == "__main__":
    main()
