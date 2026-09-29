"""Convert the official PDFs in data/raw/ into clean plain text in data/corpus/.

Nugen's developer edition only accepts plain text, so this is the file that
actually gets uploaded. Section numbers are kept as they appear in the source.

The Act (Gazette of India print) puts section headings in the page margin, so we
read it block-by-block and re-attach each margin heading in front of the section
it labels, as "[Section heading: ...]". The Telangana Rules have headings inline
and only need line re-joining.

    python scripts/build_corpus.py
"""
import re
import sys
from pathlib import Path

import fitz  # PyMuPDF

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "corpus"

SECTION_START = re.compile(r"^(CHAPTER\s+[IVXL]+|\d+[A-Z]?\.\s)")
ACT_REF = re.compile(r"^\d+\s+of\s+\d{4}\.?$")  # margin cross-refs like "20 of 1972."


def _fix(text: str) -> str:
    text = text.replace("—", "—").replace("�", "—")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _join_lines(lines):
    """Join PDF lines into running text, undoing end-of-line hyphenation."""
    out = ""
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        if out.endswith("-") and ln[:1].islower():
            out = out[:-1] + ln
        else:
            out = (out + " " + ln) if out else ln
    return out


def extract_act(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    paras = []
    for pno, page in enumerate(doc):
        body, notes = [], []
        for x0, y0, x1, y1, text, *_ in page.get_text("blocks"):
            if pno == 0 and y0 < 455:  # Hindi/English gazette masthead on page 1
                continue
            if y0 < 60:  # running head: "SEC. 1] THE GAZETTE OF INDIA EXTRAORDINARY 7"
                continue
            if x0 < 115 or x0 > 490:  # margin column
                notes.append((y0, _join_lines(text.splitlines())))
            else:
                body.append([y0, text])
        body.sort(key=lambda b: b[0])

        # Split body blocks into lines so a heading can be placed before the exact
        # line where its section starts (blocks sometimes hold "PRELIMINARY\n1. (1)...").
        lines = []
        for y0, text in body:
            for ln in text.splitlines():
                lines.append([y0, ln.strip(), []])

        for ny, note in notes:
            note = _fix(note)
            if not note:
                continue
            if ACT_REF.match(note):
                tag = f"[Margin reference: {note}]"
            else:
                tag = f"[Section heading: {note}]"
            # the body line nearest to the note's vertical position
            idx = min(range(len(lines)), key=lambda i: abs(lines[i][0] - ny)) if lines else None
            if idx is None:
                paras.append(tag)
                continue
            if tag.startswith("[Section heading"):
                # walk back to the line that actually starts the section
                j = idx
                while j >= 0 and not SECTION_START.match(lines[j][1]):
                    j -= 1
                if j >= 0 and SECTION_START.match(lines[j][1]).group(1).startswith("CHAPTER"):
                    j = -1
                idx = j if j >= 0 else idx
            lines[idx][2].append(tag)

        cur = []
        for _, ln, tags in lines:
            if not ln:
                continue
            starts_new = bool(
                SECTION_START.match(ln)
                or re.match(r"^\(([a-z]{1,3}|[ivxl]+|\d+|[A-Z])\)\s", ln)
                or ln.startswith(("Provided", "Explanation", "CHAPTER"))
                or tags
                or (ln.isupper() and len(ln) > 3)
            )
            if starts_new and cur:
                paras.append(_fix(_join_lines(cur)))
                cur = []
            for t in tags:
                paras.append(t)
            cur.append(ln)
            if ln.startswith("CHAPTER") or (ln.isupper() and len(ln) > 3 and not ln.startswith("CHAPTER")):
                paras.append(_fix(_join_lines(cur)))
                cur = []
        if cur:
            paras.append(_fix(_join_lines(cur)))
    return _merge_across_pages(paras)


def _merge_across_pages(paras):
    """A paragraph cut by a page break continues with a lowercase word; glue it back."""
    out = []
    for p in paras:
        if out and p and p[0].islower() and not out[-1].startswith("[") and not re.match(r"^\([a-z]+\)", p):
            out[-1] = _join_lines([out[-1], p])
        else:
            out.append(p)
    return "\n\n".join(out) + "\n"


RULE_MARKER = re.compile(
    r"^(CHAPTER\b|FORM\b|ANNEXURE\b|SCHEDULE\b|\d+[A-Z]?\.\s|\d+\.\s*[A-Z]|\(([a-z]{1,3}|[ivxl]+|\d+|[A-Z])\)\.?|[ivxl]+\)\s|\[See|Note|NOTIFICATION|Provided)"
)


def extract_rules(pdf_path: Path) -> str:
    doc = fitz.open(pdf_path)
    paras, cur = [], []

    def flush():
        if cur:
            paras.append(_fix(_join_lines(cur)))
            cur.clear()

    for pno, page in enumerate(doc):
        raw_lines = page.get_text().splitlines()
        # drop the page number printed at the top of every page
        while raw_lines and not raw_lines[0].strip():
            raw_lines.pop(0)
        if raw_lines and re.fullmatch(r"\s*\d{1,3}\s*", raw_lines[0]):
            raw_lines.pop(0)
        for ln in raw_lines:
            s = ln.strip()
            if not s:
                flush()
                continue
            s = re.sub(r"_{4,}", "____", s)
            prev = cur[-1] if cur else ""
            prev_is_bare_marker = bool(re.fullmatch(r"\(?[a-z0-9ivxl]{1,4}\)?\.?\)?", prev))
            if RULE_MARKER.match(s) and cur and not prev_is_bare_marker:
                flush()
            cur.append(s)
    flush()
    # drop the G.O. cover page distribution list: corpus starts at the notification
    text = "\n\n".join(p for p in paras if p)
    start = text.find("NOTIFICATION")
    if start > 0:
        header = "Telangana State Real Estate (Regulation and Development) Rules, 2017. G.O.Ms.No.202, Municipal Administration and Urban Development (M1) Department, dated 31-07-2017, published in the Telangana Gazette dated 04-08-2017."
        text = header + "\n\n" + text[start:]
    return _merge_across_pages(text.split("\n\n"))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("rera_act_2016.pdf", "rera_act_2016.txt", extract_act),
        ("telangana_rera_rules_2017.pdf", "telangana_rera_rules_2017.txt", extract_rules),
    ]
    for src, dst, fn in jobs:
        p = RAW / src
        if not p.exists():
            sys.exit(f"missing {p}; see data/SOURCES.md for the download URL")
        text = fn(p)
        (OUT / dst).write_text(text, encoding="utf-8")
        print(f"{dst}: {len(text):,} chars, {text.count(chr(10)+chr(10))+1} paragraphs")


if __name__ == "__main__":
    main()
