"""Text handling shared by the scorer, the answer checker and the app: number
normalisation ("sixty days" == "60 days", "five lakh" == "500000") and refusal detection."""
import re
import unicodedata

UNITS = {w: i for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split())}
TENS = {w: 10 * i for i, w in enumerate("_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()) if w != "_"}
SCALES = {"hundred": 100, "thousand": 1000, "lakh": 100_000, "lakhs": 100_000, "lac": 100_000, "lacs": 100_000,
          "crore": 10_000_000, "crores": 10_000_000, "million": 1_000_000}
NUMWORDS = set(UNITS) | set(TENS) | set(SCALES)


def strip_reasoning(text: str) -> str:
    """Drop <think>...</think> blocks; if a think block never closed, drop it all."""
    text = re.sub(r"<think>.*?</think>", " ", text or "", flags=re.S | re.I)
    text = re.sub(r"<think>.*", " ", text, flags=re.S | re.I)
    return text.strip()


def _words_to_numbers(tokens):
    """'twenty five thousand' -> '25000', 'five lakh' -> '500000', '1.5 lakh' -> '150000'.
    Standard accumulator; a run stops when the next word can't extend the number
    (e.g. 'thirty (30) days' gives '30 30 days', which still matches '30 days')."""
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if re.fullmatch(r"\d+(\.\d+)?", t):
            n = float(t)
            j = i + 1
            while j < len(tokens) and tokens[j] in SCALES and SCALES[tokens[j]] > 100:
                n *= SCALES[tokens[j]]
                j += 1
            out.append(str(int(n)) if n.is_integer() else t)
            i = j
            continue
        if t not in UNITS and t not in TENS:
            out.append(t)
            i += 1
            continue
        total, cur, last, j = 0, 0, None, i
        while j < len(tokens):
            w = tokens[j]
            if w in UNITS:
                if last in ("unit", "teen") or (last == "tens" and UNITS[w] >= 10):
                    break
                cur += UNITS[w]
                last = "teen" if UNITS[w] >= 10 else "unit"
            elif w in TENS:
                if last in ("unit", "teen", "tens"):
                    break
                cur += TENS[w]
                last = "tens"
            elif w in SCALES and last is not None:
                if SCALES[w] == 100:
                    cur = (cur or 1) * 100
                    last = "hundred"
                else:
                    total += (cur or 1) * SCALES[w]
                    cur, last = 0, "scale"
            elif w == "and" and last in ("hundred", "scale") and j + 1 < len(tokens) and (tokens[j + 1] in UNITS or tokens[j + 1] in TENS):
                pass
            else:
                break
            j += 1
        out.append(str(total + cur))
        i = j
    return out


def normalise(text: str) -> str:
    t = unicodedata.normalize("NFKC", text or "").lower()
    t = t.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    t = re.sub(r"(\d),(?=\d)", r"\1", t)                       # 1,00,000 -> 100000
    t = re.sub(r"(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)(one|two|three|four|five|six|seven|eight|nine)\b", r"\1 \2", t)
    t = re.sub(r"\btwo[\s-]*thirds?\b|\b2\s*/\s*3(rd|rds)?\b", " twothirds ", t)
    t = re.sub(r"₹|\brs\.?(?=\s|\d)|\binr\b|\brupees?\b", " ", t)
    t = re.sub(r"\bper\s*cent\.?|\bpercent(age)?\b|%", " percent ", t)
    t = re.sub(r"\bsq(uare)?\.?\s*(met(er|re)s?|mtrs?|m)\b\.?|\bsqm\b|\bm2\b|m²", " sqm ", t)
    t = re.sub(r"\b(sec|s)\.\s*(?=\d)|\bsection\s*", " section ", t)
    t = re.sub(r"\b(state bank of india)\b", " sbi ", t)
    t = re.sub(r"\bmarginal cost of (funds based )?lending rate\b", " mclr ", t)
    t = t.replace("/", " per ")
    t = re.sub(r"[^\w\s.]", " ", t)
    t = re.sub(r"(?<!\d)\.|\.(?!\d)", " ", t)                    # keep decimals only
    toks = _words_to_numbers(t.split())
    toks = [re.sub(r"^(year|month|day|week)s$", r"\1", w) for w in toks]
    return " ".join(toks).replace("twothirds", "2/3")


def fact_present(fact: str, norm_answer: str) -> bool:
    for alt in fact.split("|"):
        a = normalise(alt)
        if a and re.search(r"(?<![\w/])" + re.escape(a) + r"(?![\w/])", norm_answer):
            return True
    return False


REFUSAL = re.compile(
    r"not (explicitly |specifically |directly )?(mentioned|specified|provided|covered|found|available|stated|contained|included|addressed|defined|prescribed|given|present|listed)"
    r"|(is|are|was) not (in|part of|within) the (provided )?(documents?|act|rules|text|corpus)"
    r"|not in the (provided )?(documents?|act|rules|text)"
    r"|(documents?|act|rules|text|corpus)(,? \d{4},?)? (do|does|did) not (explicitly |specifically )?(mention|specify|contain|provide|cover|say|address|state|include|prescribe|define|list|deal)"
    r"|(i )?(do not|don't|cannot|can't|could not|couldn't|was unable to) (know|find|determine|answer|confirm|provide|locate)"
    r"|no (specific |such )?(information|mention|provision|details?|reference)"
    r"|unable to (find|answer|determine|locate|provide)"
    r"|(outside|beyond|not within) the scope"
    r"|(does not|doesn't) (appear|exist) in"
    r"|i (don't|do not) have (enough |sufficient )?(information|details|data)",
    re.I,
)


def is_refusal(text: str) -> bool:
    return bool(REFUSAL.search(text or ""))
