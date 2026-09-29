"""Section-level retrieval over the corpus. Plain BM25, no dependencies.

The Act and the Rules are split at their own numbering ("44. (1) ...", "25. Appeal
and the fees payable.—"), accepting a new unit only when its number is the next one
in sequence, so numbered lists inside forms and schedules don't start fake sections.
Long sections are cut into overlapping pieces that keep the section id. Rules forms
(Form 'A' ... ) are units of their own.

    from nugen.retrieval import Index
    idx = Index.load()
    hits = idx.search("time limit to appeal to the Appellate Tribunal", k=3)
    hits[0].sid, hits[0].text
"""
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

CORPUS = Path(__file__).resolve().parents[1] / "data" / "corpus"
ACT_FILE, RULES_FILE = "rera_act_2016.txt", "telangana_rera_rules_2017.txt"
MAX_CHARS, OVERLAP = 1800, 300

STOP = set("""a an the of to in on for by with from at as is are was were be been being or and any such this that these
those it its which who whom what when where how under shall may must can will would should not no than then there
their his her he she they them do does did has have had into upon within without per all each every other same said
thereof therein thereunder herein act rules rule section sections sub clause provided""".split())


@dataclass
class Chunk:
    sid: str        # "Act s.44" / "Rules r.25" / "Rules Form 'L'"
    heading: str
    text: str
    part: int = 0   # piece number inside a long section


@dataclass
class Hit:
    sid: str
    heading: str
    text: str
    score: float


def stem(w: str) -> str:
    """Just enough stemming for statute language: extension/extend, appeals/appeal, filing/file."""
    if w.isdigit() or len(w) <= 3:
        return w
    if w.endswith("nsion"):
        return w[:-4] + "d"            # extension -> extend, suspension -> suspend
    for suf in ("ations", "ation", "ings", "ing", "ment", "ies", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= (3 if suf == "s" else 4) and not w.endswith("ss"):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def tokens(text: str) -> list[str]:
    out = []
    for w in re.findall(r"[a-z0-9]+", text.lower()):
        if w in STOP or len(w) < 2 and not w.isdigit():
            continue
        out.append(stem(w))
    return out


def _split_numbered(body: str, prefix: str, heading_re=None):
    """Split at lines starting 'N. ' where N is the next expected number."""
    units, cur_n, cur_start = [], None, None
    for m in re.finditer(r"(?m)^(\d{1,3})\. ", body):
        n = int(m.group(1))
        if cur_n is None and n == 1 or cur_n is not None and n == cur_n + 1:
            if cur_n is not None:
                units.append((cur_n, body[cur_start:m.start()]))
            cur_n, cur_start = n, m.start()
    if cur_n is not None:
        units.append((cur_n, body[cur_start:]))
    out = []
    for n, txt in units:
        heading = ""
        if heading_re:
            heading = heading_re(body, txt)
        else:
            h = re.match(r"\d+\.\s*([^—\n]{3,120}?)[.\-]?\s*[—–-]", txt)
            heading = h.group(1).strip() if h else ""
        out.append((f"{prefix}{n}", heading, txt.strip()))
    return out


def _act_units(text: str):
    # the heading sits on its own line just before the section: "[Section heading: Definitions.]"
    units = []
    for sid, _, txt in _split_numbered(text, "Act s."):
        units.append([sid, "", txt])
    heads = [(m.start(), m.group(1).strip().rstrip(".")) for m in re.finditer(r"\[Section heading: ([^\]]+)\]", text)]
    pos = 0
    for u in units:
        i = text.find(u[2][:60], pos)
        if i < 0:
            continue
        pos = i
        before = [h for p, h in heads if p < i]
        if before:
            u[1] = before[-1]
        # the heading text belongs to the next section, not the tail of this one
        u[2] = re.sub(r"\n*\[Section heading: [^\]]+\]\s*$", "", u[2]).strip()
    return [tuple(u) for u in units]


def _rules_units(text: str):
    m = re.search(r"(?m)^FORM ‘A’", text)
    body, forms = (text[:m.start()], text[m.start():]) if m else (text, "")
    units = _split_numbered(body, "Rules r.")
    for fm in re.finditer(r"(?ms)^FORM ‘([A-Z]+)’\s*\n(.*?)(?=^FORM ‘|\Z)", forms):
        first = fm.group(2).strip().split("\n")[0]
        units.append((f"Rules Form '{fm.group(1)}'", first[:120], fm.group(2).strip()))
    return units


def _pieces(sid, heading, text):
    if len(text) <= MAX_CHARS:
        return [Chunk(sid, heading, text)]
    out, start, part = [], 0, 0
    while start < len(text):
        end = min(len(text), start + MAX_CHARS)
        if end < len(text):
            cut = text.rfind("\n", start + MAX_CHARS // 2, end)
            end = cut if cut > 0 else end
        out.append(Chunk(sid, heading, text[start:end].strip(), part))
        if end >= len(text):
            break
        start, part = max(end - OVERLAP, start + 1), part + 1
    return out


class Index:
    FORM_WEIGHT = 0.6  # forms and the model agreement repeat the Act's words without being the rule

    def __init__(self, chunks: list[Chunk], k1=1.4, b=0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        # heading words count twice: "Appeal and the fees payable" is what people ask about
        self.docs = [Counter(tokens(c.text) + tokens(c.heading) * 2 + tokens(c.sid)) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(w for d in self.docs for w in d)
        n = len(self.docs)
        self.idf = {w: math.log(1 + (n - f + 0.5) / (f + 0.5)) for w, f in df.items()}

    @classmethod
    def load(cls, corpus_dir=CORPUS):
        act = (Path(corpus_dir) / ACT_FILE).read_text(encoding="utf-8")
        rules = (Path(corpus_dir) / RULES_FILE).read_text(encoding="utf-8")
        chunks = []
        for sid, heading, txt in _act_units(act) + _rules_units(rules):
            chunks.extend(_pieces(sid, heading, txt))
        return cls(chunks)

    def search(self, query: str, k: int = 3) -> list[Hit]:
        q = tokens(query)
        scored = []
        for i, d in enumerate(self.docs):
            s = 0.0
            for w in q:
                f = d.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * self.lens[i] / self.avg))
            if s > 0:
                if "Form" in self.chunks[i].sid:
                    s *= self.FORM_WEIGHT
                scored.append((s, i))
        scored.sort(reverse=True)
        hits, seen = [], set()
        for s, i in scored:
            c = self.chunks[i]
            key = c.sid  # one piece per section, so three hits are three different provisions
            if key in seen:
                continue
            seen.add(key)
            hits.append(Hit(c.sid, c.heading, c.text, round(s, 3)))
            if len(hits) == k:
                break
        return hits


def format_context(hits: list[Hit]) -> str:
    blocks = []
    for h in hits:
        head = f"[{h.sid}" + (f" — {h.heading}" if h.heading else "") + "]"
        blocks.append(f"{head}\n{h.text}")
    return "\n\n".join(blocks)
