"""Thin client for the Nugen v3 API.

Where the API differs from the public cookbook (checked against
api.nugen.in/openapi-public.json, September 2026):
  * the developer edition accepts plain-text documents only, not PDF
  * benchmarks/create takes `n_samples`, not `num_questions`
  * `workflow_id` on alignment-projects/create is optional
  * `confidence_score` (0-100) only exists on aligned models. When streaming it
    arrives as separate {"object": "confidence", "confidence_score": ...} events,
    roughly one every ten tokens.

Every HTTP call goes into a JSONL ledger with an estimated cost.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

import requests

# Windows consoles default to cp1252; don't crash printing "→" or model output
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_URL = "https://api.nugen.in"  # NUGEN_BASE_URL overrides it; the tests point it at tests/mock_server.py
ROOT = Path(__file__).resolve().parents[1]

# Nugen doesn't publish per-token prices, so these are guesses that only feed the
# running cost estimate. NUGEN_USD_PER_1K_IN / _OUT override them.
DEFAULT_USD_PER_1K_IN = 0.0005
DEFAULT_USD_PER_1K_OUT = 0.0015


def load_env(path: Path = ROOT / ".env") -> None:
    """Minimal .env loader (KEY=VALUE lines). Doesn't override real env vars."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


class NugenError(RuntimeError):
    def __init__(self, status: int | None, body: str, url: str):
        super().__init__(f"HTTP {status} from {url}: {body[:500]}")
        self.status, self.body, self.url = status, body, url


class BudgetExceeded(RuntimeError):
    pass


@dataclass
class Ledger:
    """Append-only log of every API call plus a hard cap on how many we make."""

    path: Path
    max_calls: int | None = None
    calls: int = 0
    est_usd: float = 0.0

    def check(self) -> None:
        if self.max_calls is not None and self.calls >= self.max_calls:
            raise BudgetExceeded(f"--max-calls {self.max_calls} reached")

    def log(self, rec: dict) -> None:
        self.calls += 1
        self.est_usd += rec.get("est_usd", 0.0) or 0.0
        rec = {"ts": time.time(), "n": self.calls, "est_usd_total": round(self.est_usd, 6), **rec}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _confidence_value(ev: dict) -> float | None:
    # The docs never named the field, so this accepts the obvious variants; in practice
    # it's always "confidence_score".
    for k in ("confidence_score", "confidence", "score", "value"):
        v = ev.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return float(v)
        if isinstance(v, dict):
            inner = _confidence_value(v)
            if inner is not None:
                return inner
    data = ev.get("data")
    if isinstance(data, dict):
        return _confidence_value(data)
    return None


def parse_sse_lines(lines: Iterable[str], on_event: Callable[[dict], None] | None = None) -> dict:
    """Fold an SSE stream into text + confidence readings. Pure function, so it's
    unit-testable without the network. Keeps every event verbatim."""
    out: dict[str, Any] = {
        "text": "",
        "reasoning": "",
        "confidences": [],
        "final_confidence_score": None,
        "usage": None,
        "finish_reason": None,
        "events": [],
        "unparsed": [],
        "done": False,
        "error_event": None,
    }
    for raw in lines:
        if raw is None:
            continue
        line = raw.strip()
        if not line or line.startswith(":"):
            continue
        if line.startswith("data:"):
            line = line[5:].strip()
        elif line.startswith(("event:", "id:", "retry:")):
            continue
        if line == "[DONE]":
            out["done"] = True
            break
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            out["unparsed"].append(line)
            continue
        out["events"].append(ev)
        if not isinstance(ev, dict):
            continue
        if "error" in ev and not ev.get("choices"):
            out["error_event"] = ev["error"]
        if ev.get("object") == "confidence":
            v = _confidence_value(ev)
            if v is not None:
                out["confidences"].append(v)
        elif isinstance(ev.get("confidence_score"), (int, float)):
            # some responses carry it on the final chunk instead / as well
            out["final_confidence_score"] = float(ev["confidence_score"])
        for ch in ev.get("choices") or []:
            delta = ch.get("delta") or ch.get("message") or {}
            if isinstance(delta, dict):
                out["text"] += delta.get("content") or ""
                out["reasoning"] += delta.get("reasoning_content") or delta.get("reasoning") or ""
            elif "text" in ch:
                out["text"] += ch.get("text") or ""
            if ch.get("finish_reason"):
                out["finish_reason"] = ch["finish_reason"]
        if ev.get("usage"):
            out["usage"] = ev["usage"]
        if on_event:
            on_event(ev)
    return out


def summarise_confidence(confs: list[float], final: float | None = None) -> dict:
    if not confs and final is not None:
        confs = [final]
    if not confs:
        return {"conf_n": 0, "conf_min": None, "conf_last": None, "conf_mean": None, "conf_first": None}
    return {
        "conf_n": len(confs),
        "conf_min": min(confs),
        "conf_last": confs[-1],
        "conf_mean": sum(confs) / len(confs),
        "conf_first": confs[0],
    }


@dataclass
class Nugen:
    api_key: str | None = None
    base_url: str | None = None
    ledger: Ledger | None = None
    dry_run: bool = False
    usd_per_1k_in: float = field(default_factory=lambda: float(os.environ.get("NUGEN_USD_PER_1K_IN", DEFAULT_USD_PER_1K_IN)))
    usd_per_1k_out: float = field(default_factory=lambda: float(os.environ.get("NUGEN_USD_PER_1K_OUT", DEFAULT_USD_PER_1K_OUT)))

    def __post_init__(self):
        load_env()
        self.api_key = self.api_key or os.environ.get("NUGEN_API_KEY")
        self.base_url = (self.base_url or os.environ.get("NUGEN_BASE_URL") or BASE_URL).rstrip("/")
        if not self.api_key and not self.dry_run:
            raise SystemExit("NUGEN_API_KEY not set. Put it in .env (see .env.example).")
        self.ledger = self.ledger or Ledger(ROOT / "state" / "api_calls.jsonl")
        self.s = requests.Session()
        self.s.headers["Authorization"] = f"Bearer {self.api_key}"

    def __repr__(self):  # keep the API key out of tracebacks and logs
        return f"Nugen(base_url={self.base_url!r}, dry_run={self.dry_run})"

    # ---------------- plain JSON calls ----------------
    def _req(self, method: str, path: str, timeout: float = 60, retries: int = 3, **kw) -> Any:
        url = self.base_url + path
        if self.dry_run:
            print(f"[dry-run] {method} {path} {json.dumps(kw.get('json') or kw.get('data') or '', default=str)[:200]}")
            return {}
        last: Exception | None = None
        for attempt in range(retries):
            self.ledger.check()
            t0 = time.time()
            try:
                r = self.s.request(method, url, timeout=timeout, **kw)
            except requests.RequestException as e:
                self.ledger.log({"method": method, "path": path, "status": None, "error": repr(e)[:300], "latency_s": time.time() - t0, "est_usd": 0})
                last = e
                time.sleep(2 ** attempt * 3)
                continue
            self.ledger.log({"method": method, "path": path, "status": r.status_code, "latency_s": round(time.time() - t0, 3), "est_usd": 0})
            if r.status_code >= 500 or r.status_code == 429:
                last = NugenError(r.status_code, r.text, url)
                time.sleep(2 ** attempt * 3)
                continue
            if r.status_code >= 400:
                raise NugenError(r.status_code, r.text, url)
            try:
                return r.json()
            except ValueError:
                return {"_text": r.text}
        raise last  # type: ignore[misc]

    def get(self, path, **kw):
        return self._req("GET", path, **kw)

    def post(self, path, **kw):
        return self._req("POST", path, **kw)

    # ---------------- documents / benchmarks ----------------
    def upload_documents(self, paths: list[Path], categories: list[str] | None = None, names: list[str] | None = None) -> list[str]:
        files = [("files", (p.name, p.open("rb"), "text/plain")) for p in paths]
        data: list[tuple[str, str]] = []
        for c in categories or []:
            data.append(("categories", c))
        for n in names or []:
            data.append(("names", n))
        try:
            # a failed multipart upload may have been accepted server-side: don't auto-retry
            res = self._req("POST", "/api/v3/documents/create", files=files, data=data, timeout=180, retries=1)
        finally:
            for _, (_, fh, _) in files:
                fh.close()
        return res.get("document_ids") or res.get("documents") or []

    def document_status(self, doc_id):
        return self.get(f"/api/v3/documents/{doc_id}/status")

    def list_documents(self):
        return self.get("/api/v3/documents/list")

    def create_benchmark(self, document_ids, n_samples=30, name=None):
        body = {"document_ids": document_ids, "n_samples": n_samples}
        if name:
            body["benchmark_name"] = name
        return self.post("/api/v3/benchmarks/create", json=body, retries=1)

    def benchmark_status(self, bid):
        return self.get(f"/api/v3/benchmarks/{bid}/status")

    def benchmark_data(self, bid):
        return self.get(f"/api/v3/benchmarks/{bid}/data")

    # ---------------- models / alignment / deployment ----------------
    def base_models(self):
        return self.get("/api/v3/models/base", params={"limit": 100}).get("models", [])

    def aligned_models(self):
        return self.get("/api/v3/models/aligned", params={"limit": 100}).get("domain_aligned_models", [])

    def model(self, model_id):
        return self.get(f"/api/v3/models/{model_id}")

    def create_alignment(self, name, base_model_id, document_ids, benchmark_id=None, description=None):
        body = {"alignment_name": name, "base_model_id": base_model_id, "document_ids": document_ids}
        if benchmark_id:
            body["benchmark_id"] = benchmark_id
        if description:
            body["description"] = description
        return self.post("/api/v3/alignment-projects/create", json=body, retries=1)

    def alignment_status(self, aid):
        return self.get(f"/api/v3/alignment-projects/{aid}/status")

    def alignment(self, aid):
        return self.get(f"/api/v3/alignment-projects/{aid}")

    def deploy(self, model_id, early=False):
        return self.post(f"/api/v3/models/{model_id}/deployment", params={"early": str(early).lower()}, retries=1)

    def undeploy(self, model_id):
        return self._req("DELETE", f"/api/v3/models/{model_id}/deployment", retries=1)

    def deployment_status(self, model_id):
        return self.get(f"/api/v3/models/{model_id}/deployment/status")

    def evaluation_status(self, eid):
        return self.get(f"/api/v3/evaluations/{eid}/status")

    def evaluation(self, eid):
        return self.get(f"/api/v3/evaluations/{eid}")

    def evaluation_results(self, eid):
        return self.get(f"/api/v3/evaluations/{eid}/results")

    # ---------------- inference ----------------
    def stream_chat(
        self,
        model: str,
        messages: list[dict],
        max_tokens: int = 600,
        temperature: float = 0.1,
        read_timeout: float = 30.0,
        total_timeout: float = 120.0,
        max_attempts: int = 3,
        on_text: Callable[[str], None] | None = None,
        on_confidence: Callable[[float], None] | None = None,
        extra: dict | None = None,
    ) -> dict:
        """Streaming chat completion with retries. Never raises for serving
        failures: the returned record says what happened (status = ok /
        timeout / http_error / connection_error / empty), and `attempts` lists
        every try. No fallback to any other model, ever."""
        body = {"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": temperature, "stream": True}
        if extra:
            body.update(extra)
        attempts = []
        final: dict[str, Any] = {}
        for attempt in range(1, max_attempts + 1):
            self.ledger.check()
            t0 = time.time()
            rec: dict[str, Any] = {"attempt": attempt, "t_start": t0}
            first_token_at = [None]

            def _on_event(ev):
                if first_token_at[0] is None:
                    for ch in ev.get("choices") or []:
                        if (ch.get("delta") or {}).get("content"):
                            first_token_at[0] = time.time()
                if on_text:
                    for ch in ev.get("choices") or []:
                        c = (ch.get("delta") or {}).get("content")
                        if c:
                            on_text(c)
                if on_confidence and ev.get("object") == "confidence":
                    v = _confidence_value(ev)
                    if v is not None:
                        on_confidence(v)

            parsed = None
            try:
                with self.s.post(
                    self.base_url + "/api/v3/inference/chat/completions",
                    json=body,
                    stream=True,
                    timeout=(10, read_timeout),
                    headers={"Accept": "text/event-stream"},
                ) as r:
                    rec["http_status"] = r.status_code
                    if r.status_code != 200:
                        rec["status"] = "http_error"
                        rec["error"] = r.text[:1000]
                    else:
                        deadline = t0 + total_timeout

                        def lines():
                            for ln in r.iter_lines(decode_unicode=True):
                                if time.time() > deadline:
                                    raise TimeoutError(f"stream exceeded {total_timeout}s")
                                yield ln

                        parsed = parse_sse_lines(lines(), on_event=_on_event)
                        if parsed["error_event"]:
                            rec["status"] = "stream_error"
                            rec["error"] = json.dumps(parsed["error_event"])[:1000]
                        elif not parsed["text"].strip():
                            rec["status"] = "empty"
                        else:
                            rec["status"] = "ok"
            except (requests.Timeout, TimeoutError) as e:
                rec["status"], rec["error"] = "timeout", repr(e)[:300]
            except requests.RequestException as e:
                rec["status"], rec["error"] = "connection_error", repr(e)[:300]
            rec["latency_s"] = round(time.time() - t0, 3)
            rec["ttft_s"] = round(first_token_at[0] - t0, 3) if first_token_at[0] else None

            usage = (parsed or {}).get("usage") or {}
            p_tok = usage.get("prompt_tokens") or sum(len(m.get("content", "")) for m in messages) / 4
            c_tok = usage.get("completion_tokens") or len((parsed or {}).get("text", "")) / 4
            est = p_tok / 1000 * self.usd_per_1k_in + c_tok / 1000 * self.usd_per_1k_out
            self.ledger.log(
                {
                    "method": "POST",
                    "path": "/api/v3/inference/chat/completions",
                    "model": model,
                    "status": rec["status"],
                    "http_status": rec.get("http_status"),
                    "latency_s": rec["latency_s"],
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "est_usd": round(est, 6),
                }
            )
            rec["est_usd"] = round(est, 6)
            if parsed is not None:
                rec["parsed"] = parsed
            attempts.append(rec)
            if rec["status"] == "ok":
                final = rec
                break
            # 4xx other than 408/429 won't fix themselves
            hs = rec.get("http_status")
            if rec["status"] == "http_error" and hs and 400 <= hs < 500 and hs not in (408, 429):
                break
            if attempt < max_attempts:
                time.sleep(min(30, 3 * 2 ** (attempt - 1)))

        last = final or attempts[-1]
        parsed = last.get("parsed") or {}
        return {
            "status": last["status"],
            "http_status": last.get("http_status"),
            "error": last.get("error"),
            "text": parsed.get("text", ""),
            "reasoning": parsed.get("reasoning", ""),
            "finish_reason": parsed.get("finish_reason"),
            "usage": parsed.get("usage"),
            "confidences": parsed.get("confidences", []),
            "final_confidence_score": parsed.get("final_confidence_score"),
            **summarise_confidence(parsed.get("confidences", []), parsed.get("final_confidence_score")),
            "latency_s": last.get("latency_s"),
            "ttft_s": last.get("ttft_s"),
            "n_attempts": len(attempts),
            "attempts": [{k: v for k, v in a.items() if k != "parsed"} for a in attempts],
            "events": parsed.get("events", []),
            "unparsed": parsed.get("unparsed", []),
            "est_usd": round(sum(a.get("est_usd", 0) for a in attempts), 6),
        }
