"""Assemble the one-page write-up from the result files, so its numbers can't drift
from the analysis.

    python results/page/build.py        # -> results/page/writeup.html
"""
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent


def pc(x, d=0):
    return f"{100 * x:.{d}f}%"


def main():
    cb = json.loads((RES / "summary.json").read_text(encoding="utf-8"))["models"]
    rag = json.loads((RES / "rag" / "summary.json").read_text(encoding="utf-8"))["models"]
    ck = json.loads((RES / "checker" / "summary.json").read_text(encoding="utf-8"))
    g = ck["groups"]["retrieval/aligned"]
    cv = ck["app_cv"]

    def lat(m):
        s = m["serving"]
        return f"{s['latency_p50']:.1f} / {s['latency_p95']:.1f} s"

    auc = lambda m, st: f"{m['calibration'][st]['answerable_correct']['auroc']:.2f}"
    per = pd.read_csv(RES / "checker" / "per_response.csv")
    ov = {"A01", "A05", "A09", "A12", "A11", "A16", "A25", "A28", "A03"}  # questions sharing a fact with Nugen's benchmark
    cbp = per[(per.condition == "closed_book") & per.qid.isin(ov)]

    v = {
        "CB_AL_ACC": pc(cb["aligned"]["accuracy"]), "CB_BA_ACC": pc(cb["base"]["accuracy"]),
        "RAG_AL_ACC": pc(rag["aligned"]["accuracy"]), "RAG_BA_ACC": pc(rag["base"]["accuracy"]),
        "CB_AL_ABS": pc(cb["aligned"]["abstention_rate"]), "CB_BA_ABS": pc(cb["base"]["abstention_rate"]),
        "RAG_AL_ABS": pc(rag["aligned"]["abstention_rate"]), "RAG_BA_ABS": pc(rag["base"]["abstention_rate"]),
        "CB_AL_CONS": pc(cb["aligned"]["consistency_identical"]), "CB_BA_CONS": pc(cb["base"]["consistency_identical"]),
        "RAG_AL_CONS": pc(rag["aligned"]["consistency_identical"]), "RAG_BA_CONS": pc(rag["base"]["consistency_identical"]),
        "CB_AUC": auc(cb["aligned"], "conf_mean"), "RAG_AUC": auc(rag["aligned"], "conf_mean"),
        "RAG_ECE": f"{rag['aligned']['calibration']['conf_mean']['answerable_correct']['ece']:.2f}",
        "CB_AL_LAT": lat(cb["aligned"]), "CB_BA_LAT": lat(cb["base"]), "RAG_AL_LAT": lat(rag["aligned"]), "RAG_BA_LAT": lat(rag["base"]),
        "NF_SHOWN": f"{g['no_filter']['answered']}/200", "NF_ERR": pc(g["no_filter"]["error"]),
        "CK_SHOWN": f"{g['checker_supported']['answered']}/200", "CK_ERR": pc(g["checker_supported"]["error"]),
        "CK_UN": f"{round(50 * (1 - g['checker_supported']['unanswerable_blocked']))}/50",
        "CO_SHOWN": f"{cv['confidence_only']['shown']}/200", "CO_ERR": pc(cv["confidence_only"]["error"]),
        "CO_UN": f"{cv['confidence_only']['unanswerable_shown']}/50",
        "APP_SHOWN": f"{cv['confidence_and_checker']['shown']}/200", "APP_ERR": pc(cv["confidence_and_checker"]["error"]),
        "APP_UN": f"{cv['confidence_and_checker']['unanswerable_shown']}/50", "APP_COV": pc(cv["confidence_and_checker"]["coverage"]),
        "OV_AL": str(int(cbp[cbp.model_label == "aligned"].is_correct.sum())),
        "OV_BA": str(int(cbp[cbp.model_label == "base"].is_correct.sum())),
    }
    body = (HERE / "body.html").read_text(encoding="utf-8")
    for k, val in v.items():
        body = body.replace("{{" + k + "}}", val)
    assert "{{" not in body, [l for l in body.splitlines() if "{{" in l]

    data = {
        "acc": {f"{c}/{m}": ck["groups"][f"{c}/{m}"]["accuracy_answerable"] for c in ("closed_book", "retrieval") for m in ("base", "aligned")},
        "sweep": [{"coverage": p["coverage"], "error": p["error"], "answered": p["answered"]} for p in g["conf_sweep"]],
        "checker": {k: g["checker_supported"][k] for k in ("coverage", "error")},
        "nofilter": {k: g["no_filter"][k] for k in ("coverage", "error")},
        "app": {k: cv["confidence_and_checker"][k] for k in ("coverage", "error")},
    }
    page = (HERE / "template.html").read_text(encoding="utf-8")
    page = page.replace("{{BODY}}", body).replace("{{DATA}}", json.dumps(data)).replace("{{CHARTJS}}", (HERE / "chart.js").read_text(encoding="utf-8"))
    (HERE / "writeup.html").write_text(page, encoding="utf-8")
    print("wrote", HERE / "writeup.html", {k: v[k] for k in ("CB_AL_ACC", "RAG_AL_ACC", "CB_AUC", "RAG_AUC", "APP_COV", "APP_ERR", "OV_AL", "OV_BA")})


if __name__ == "__main__":
    main()
