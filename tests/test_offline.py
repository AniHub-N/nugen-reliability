"""Offline checks: SSE parsing, answer normalisation, labels, stats, retrieval, the answer checker.

    python tests/test_offline.py            # unit checks only
    python tests/test_offline.py --e2e      # + full run_eval/score/plots against tests/mock_server.py
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from nugen import parse_sse_lines, summarise_confidence  # noqa: E402
from score import auroc, ece, is_refusal, label_response, normalise  # noqa: E402


def test_sse():
    lines = [
        'data: {"id":"x","choices":[{"index":0,"delta":{"role":"assistant","content":"Ten"},"finish_reason":null}]}',
        "",
        'data: {"object":"confidence","confidence_score":81.5}',
        'data: {"id":"x","choices":[{"index":0,"delta":{"content":" per cent"},"finish_reason":null}]}',
        'data: {"object":"confidence","confidence":{"score":62}}',
        ": keep-alive",
        'data: {"id":"x","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":2}}',
        "data: [DONE]",
        'data: {"never":"read"}',
    ]
    p = parse_sse_lines(lines)
    assert p["text"] == "Ten per cent", p["text"]
    assert p["confidences"] == [81.5, 62.0]
    assert p["finish_reason"] == "stop" and p["usage"]["completion_tokens"] == 2 and p["done"]
    s = summarise_confidence(p["confidences"])
    assert s["conf_min"] == 62 and s["conf_last"] == 62 and s["conf_first"] == 81.5
    assert summarise_confidence([], 70.0)["conf_mean"] == 70.0


def test_normalise_and_labels():
    assert normalise("Rs. 10,000 per day") == "10000 per day"
    assert normalise("five lakhs rupees") == "500000"
    assert normalise("two-thirds") == "2/3"
    assert normalise("seventy per cent.") == "70 percent"
    assert normalise("sixtyfive years") == "65 year"
    q = {"answerable": True, "key_facts": ["section 79"]}
    assert label_response(q, {"status": "ok", "text": "Section 790"})[0] == "wrong"
    assert label_response(q, {"status": "ok", "text": "See s. 79."})[0] == "correct"
    q = {"answerable": True, "key_facts": ["10000", "5 percent"]}
    assert label_response(q, {"status": "ok", "text": "Rs 10,000 a day"})[0] == "partial"
    assert label_response(q, {"status": "ok", "text": "<think>10000 5%</think>Not in the documents."})[0] == "abstained"
    assert label_response({"answerable": False}, {"status": "ok", "text": "It is 18%."})[0] == "answered"
    assert label_response({"answerable": False}, {"status": "timeout"})[0] == "no_response"
    assert is_refusal("The Telangana Rules do not specify this.")
    assert not is_refusal("The fee is five thousand rupees under rule 25.")


def test_stats():
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0
    assert auroc([5, 5, 5, 5], [0, 1, 0, 1]) == 0.5
    assert abs(ece([0.9, 0.9, 0.1], [1, 0, 0])[0] - 0.3) < 1e-9


def test_retrieval():
    from nugen.retrieval import Index, stem
    assert [stem(w) for w in ("fees", "appeals", "extension", "complaints")] == ["fee", "appeal", "extend", "complaint"]
    idx = Index.load()
    sids = {c.sid for c in idx.chunks}
    assert {"Act s.1", "Act s.92", "Rules r.1", "Rules r.38", "Rules Form 'A'"} <= sids
    assert "Rules r.25" in [h.sid for h in idx.search("fee for filing an appeal with the Appellate Tribunal", 3)]
    hits = idx.search("refund timeline", 4)
    assert len({h.sid for h in hits}) == len(hits), "one hit per section"


def test_verify():
    from nugen.verify import check, unknown_terms
    assert unknown_terms("What is the registration fee under the Karnataka RERA rules?") == ["Karnataka"]
    assert unknown_terms("What GST applies to a 2BHK flat?") == ["GST", "2BHK"]
    assert unknown_terms("What fee does TS-RERA charge under Rule 25 in Telangana?") == []
    assert unknown_terms("what do documents conatin?") == [], "lowercase typos are not named things"
    ex = [{"sid": "Act s.44", "text": "(2) Every appeal ... shall be preferred within a period of sixty days from the date ..."}]
    q = "What is the time limit to appeal to the Appellate Tribunal?"
    assert check(q, "Within 30 days of the order (section 44).", ex)["verdict"] == "unsupported"
    r = check(q, "Within sixty days, under section 44(2).", ex)
    assert r["verdict"] == "supported" and r["unsupported"] == []
    assert check(q, "Within sixty days under section 45.", ex)["unsupported"] == ["section 45"]
    assert check(q, "The appeal goes to the Tribunal.", ex)["verdict"] == "unchecked"
    assert check(q, "This is not specified in the documents.", ex)["verdict"] == "declined"


def e2e():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    srv = subprocess.Popen([sys.executable, str(ROOT / "tests" / "mock_server.py"), str(port), "4"])
    out = ROOT / "tests" / "_out"
    try:
        time.sleep(1)
        env = {**os.environ, "NUGEN_BASE_URL": f"http://127.0.0.1:{port}", "NUGEN_API_KEY": "test"}
        if out.exists():
            import shutil

            shutil.rmtree(out)
        subprocess.run([sys.executable, "scripts/run_eval.py", "--aligned-model", "model_test", "--base-model", "llama-v3p2-3b-reasoning",
                        "--ids", "A06,A21,A25,A01,U01,U05", "--repeats", "5", "--read-timeout", "2", "--total-timeout", "10",
                        "--resume", str(out)], cwd=ROOT, env=env, check=True, stdout=subprocess.DEVNULL)
        recs = [json.loads(l) for l in (out / "responses.jsonl").read_text(encoding="utf-8").splitlines()]
        assert len(recs) == 60
        assert any(a["status"] != "ok" for r in recs for a in r["attempts"]), "mock should have injected failures"
        assert all(r["confidences"] for r in recs if r["model_label"] == "aligned" and r["status"] == "ok")
        assert all(not r["confidences"] for r in recs if r["model_label"] == "base")
        subprocess.run([sys.executable, "scripts/score.py", str(out), "--out", str(out / "scored")], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        subprocess.run([sys.executable, "scripts/plots.py", "--in", str(out / "scored")], cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
        summ = json.loads((out / "scored" / "summary.json").read_text(encoding="utf-8"))
        assert summ["models"]["aligned"]["calibration"]["conf_mean"]["answerable_correct"]["auroc"] is not None
        print("e2e ok:", out)
    finally:
        srv.terminate()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    if "--e2e" in sys.argv:
        e2e()
