"""Check an answer against the provisions it was given, without a model.

Two checks, both deterministic:

1. Scope. If the question names something the documents never mention anywhere
   ("Karnataka", "MahaRERA", "2BHK", "Newtech"), the answer can't come from them,
   however confident the model is. Only capitalised or alphanumeric terms count,
   so a typo in an ordinary word doesn't trigger it.

2. Figures. Every number in the answer (days, months, per cent, rupees, section
   and rule numbers) must appear in the retrieved text. "within 30 days" against
   an excerpt that says "sixty days" fails. Numbers the question itself contains
   and the Act/Rules years are ignored.

verdicts, from most to least trustworthy:
  supported     every figure in the answer is in the excerpts (and there is at least one)
  unchecked     the answer has no figures to check (e.g. "the High Court")
  unsupported   at least one figure is not in the excerpts
  out_of_scope  the question names something the documents never mention
  declined      the model itself said it's not in the documents
"""
import re
from functools import lru_cache
from pathlib import Path

from .text import is_refusal, normalise, strip_reasoning

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
IGNORE_NUMBERS = {"2016", "2017", "1", "0"}  # Act/Rules years; "1" is too common in "(1)" to mean anything
# names people use for things the documents call something else
KNOWN_ALIASES = {"rera", "tsrera", "tgrera", "ts", "tg", "telangana", "india", "indian", "authority", "tribunal"}


@lru_cache(maxsize=1)
def corpus_vocab() -> frozenset:
    text = " ".join(p.read_text(encoding="utf-8") for p in sorted(CORPUS.glob("*.txt")))
    return frozenset(re.findall(r"[a-z0-9]+", text.lower()))


def unknown_terms(question: str) -> list[str]:
    """Named things in the question that appear nowhere in the corpus."""
    vocab = corpus_vocab()
    out = []
    for m in re.finditer(r"[A-Za-z0-9][A-Za-z0-9\-]*", question):
        w, start = m.group(0), m.start()
        named = (w[0].isupper() and start > 0) or any(c.isupper() for c in w[1:]) or (
            re.search(r"\d", w) and re.search(r"[A-Za-z]", w))
        if not named:
            continue
        parts = [p for p in re.split(r"-", w.lower()) if p]
        key = "".join(parts)
        if key in KNOWN_ALIASES or all(p in vocab or p in KNOWN_ALIASES for p in parts):
            continue
        out.append(w)
    return out


def _numbers(norm: str) -> set[str]:
    return set(re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", norm))


def _citations(norm: str) -> set[tuple[str, str]]:
    """('section', '44'), ('rule', '25') mentioned in a normalised answer."""
    return {(k, n) for k, n in re.findall(r"\b(section|rule)\s+(\d+)", norm)}


def check(question: str, answer: str, excerpts: list[dict]) -> dict:
    """excerpts: [{"sid": "Act s.44", "text": ...}, ...] as retrieved for this question."""
    answer = strip_reasoning(answer or "")
    sids = [e["sid"] for e in excerpts]
    out = {"verdict": None, "unknown_terms": [], "figures": [], "unsupported": [], "searched": sids}

    unknown = unknown_terms(question)
    if unknown:
        out.update(verdict="out_of_scope", unknown_terms=unknown)
        return out
    if is_refusal(answer):
        out["verdict"] = "declined"
        return out

    ctx_norm = normalise("\n".join(e["text"] for e in excerpts))
    ctx_nums = _numbers(ctx_norm)
    cited_ok = {("section", s.split("s.")[1]) for s in sids if s.startswith("Act s.")} | \
               {("rule", s.split("r.")[1]) for s in sids if s.startswith("Rules r.")}
    ans_norm = normalise(answer)
    q_nums = _numbers(normalise(question))

    cites = _citations(ans_norm)
    cite_nums = {n for _, n in cites}
    figures, bad = [], []
    for kind, n in sorted(cites):
        ok = (kind, n) in cited_ok or re.search(rf"\b{kind} {n}\b", ctx_norm) is not None
        figures.append(f"{kind} {n}")
        if not ok:
            bad.append(f"{kind} {n}")
    for n in sorted(_numbers(ans_norm) - q_nums - IGNORE_NUMBERS - cite_nums):
        # show the figure with its unit so the app can say "'30 days' isn't in s.44"
        m = re.search(rf"(?<![\w.]){re.escape(n)}(?![\w.])(\s+(day|month|year|week|percent|sqm|lakh|crore))?", ans_norm)
        unit = (m.group(2) or "") if m else ""
        label = f"{n} {unit}{'s' if unit in ('day', 'month', 'year', 'week') and n != '1' else ''}".strip()
        figures.append(label)
        if n not in ctx_nums:
            bad.append(label)
    out.update(figures=figures, unsupported=bad)
    out["verdict"] = "unsupported" if bad else ("supported" if figures else "unchecked")
    return out


RANK = {"supported": 3, "unchecked": 2, "unsupported": 1, "out_of_scope": 0, "declined": 0}


def explain(result: dict) -> str:
    """One line a person can check by hand."""
    where = ", ".join(result["searched"]) or "nothing"
    v = result["verdict"]
    if v == "out_of_scope":
        terms = ", ".join(f"'{t}'" for t in result["unknown_terms"])
        return f"{terms} doesn't appear anywhere in the RERA Act or the Telangana Rules, so they can't answer this."
    if v == "declined":
        return f"The model found no answer in {where}."
    if v == "unsupported":
        figs = ", ".join(f"'{f}'" for f in result["unsupported"])
        return f"The answer says {figs}, which isn't in the provisions it was given ({where})."
    if v == "supported":
        return f"Every figure in the answer appears in {where}."
    return f"No figures to check; based on {where}."
