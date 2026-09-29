"""A fake Nugen API for offline testing. Mimics the documented streaming format,
including interleaved {"object": "confidence"} events, and misbehaves on purpose:
some requests 504, some hang past the client timeout, some stall mid-stream.

    python tests/mock_server.py 8765
    NUGEN_BASE_URL=http://127.0.0.1:8765 NUGEN_API_KEY=test python scripts/run_eval.py ...

Behaviour is deterministic per (model, question, call count) so tests are repeatable.
"""
import hashlib
import json
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CALLS = {}
HANG_SECONDS = float(sys.argv[2]) if len(sys.argv) > 2 else 4.0

GOOD = {
    "maximum advance": "Under section 13(1) a promoter cannot take more than ten per cent of the cost as an advance without a registered agreement for sale.",
    "bars civil courts": "Section 79 bars civil courts from entertaining such suits.",
    "real estate agent pay": "Under rule 8(2) the fee is ten thousand rupees for an individual and fifty thousand rupees otherwise.",
}


def _h(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest(), 16)


class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def handle(self):
        try:
            super().handle()
        except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
            pass  # the client gave up on a stalled stream, which is the point

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/api/v3/models/base"):
            return self._json(200, {"models": [{"model_id": "llama-v3p2-3b-reasoning", "model_name": "x", "alignment_ready": True, "type": "llm"}]})
        self._json(404, {"detail": "not mocked"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        req = json.loads(self.rfile.read(n) or b"{}")
        if self.headers.get("Authorization") != "Bearer test":
            return self._json(401, {"detail": "bad key"})
        if not self.path.startswith("/api/v3/inference/chat/completions"):
            return self._json(404, {"detail": "not mocked"})
        model = req["model"]
        q = req["messages"][-1]["content"]
        k = (model, q)
        CALLS[k] = CALLS.get(k, 0) + 1
        h = _h(model, q, CALLS[k])
        aligned = model.startswith("model_")
        # failure injection: aligned model misbehaves more, like the real thing reportedly does
        r = h % 100
        if r < (12 if aligned else 4):
            return self._json(504, {"detail": "Model did not respond within 50 seconds"})
        if r < (18 if aligned else 6):
            time.sleep(HANG_SECONDS)  # longer than the test's read timeout
            return self._json(504, {"detail": "late"})

        answer = next((v for key, v in GOOD.items() if key in q), None)
        unanswerable = any(w in q for w in ("Karnataka", "GST", "stamp duty", "MahaRERA", "Andhra"))
        if unanswerable:
            answer = "This is not in the documents." if (h >> 8) % 3 == 0 else "The rate is 7 per cent under the state rules."
        elif answer is None or (h >> 8) % 4 == 0:
            answer = "The Act says the period is ninety days under section 12."
        correct = answer in GOOD.values() or answer.startswith("This is not")
        base_conf = (80 if correct else 55) + (h >> 16) % 20 - 10

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def send(obj):
            line = ("data: " + (obj if isinstance(obj, str) else json.dumps(obj)) + "\n\n").encode()
            self.wfile.write(f"{len(line):X}\r\n".encode() + line + b"\r\n")
            self.wfile.flush()

        words = answer.split(" ")
        stall = aligned and (h >> 24) % 25 == 0
        for i, w in enumerate(words):
            send({"id": "nugen-x", "created": time.time(), "model": model,
                  "choices": [{"index": 0, "delta": {"content": (" " if i else "") + w}, "finish_reason": None}]})
            if aligned and i % 4 == 3:
                send({"object": "confidence", "confidence_score": round(base_conf + (i % 3) - 1, 2)})
            if stall and i == 2:
                time.sleep(HANG_SECONDS)
        send({"id": "nugen-x", "created": time.time(), "model": model,
              "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
              "usage": {"prompt_tokens": 90, "completion_tokens": len(words), "total_tokens": 90 + len(words)}})
        if aligned:
            send({"object": "confidence", "confidence_score": base_conf})
        send("[DONE]")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8765
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
