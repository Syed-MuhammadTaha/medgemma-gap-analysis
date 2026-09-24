"""Experiment 2a: does ROUGE notice a treatment-changing error?

ROUGE is the only metric the MedGemma 1.5 model card reports for whole-slide
pathology reports. For each reference in data/00_cases.csv we
make two kinds of variants and score them against the reference:

  correct  paraphrase_light  spelling/format only (tumor -> tumour, "one" -> "1")
           paraphrase_full   + pathology synonyms (invasive -> infiltrating, ...)
  wrong    flip_<fact>       one clinical fact flipped everywhere it appears

A wrong report outscoring a correct paraphrase of the same case.

  uv run src/02_metric_paradox.py   ->  results/02_metric_paradox.csv
"""

import csv
import re
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer

from utils import plots

ROOT = Path(__file__).parent.parent
LVI = r"(?<!no )\b(?:lymphovascular|angiolymphatic|vascular|lymphatic)(?:\s*\([^)]{0,20}\))?\s+invasion"

# fact -> (pattern whose group "w" is the word to flip, what to flip it to)
FLIPS = {
    "diagnosis": (r"\b(?P<w>ductal|lobular)\b", {"ductal": "lobular", "lobular": "ductal"}),
    "laterality": (r"\b(?P<w>left|right)\b", {"left": "right", "right": "left"}),
    "grade": (r"(?<!nuclear )\bgrade:?\s*(?P<w>iii|ii|i|1|2|3)\b(?!\s*/)",
              {"1": "2", "2": "3", "3": "2", "i": "ii", "ii": "iii", "iii": "ii"}),
    "nodes": (r"\((?P<w>\d+)\s*/\s*\d+\)", lambda n: "0" if n != "0" else "1"),
    "margins": (r"\bmargins?\W{1,4}(?:are |is )?(?P<w>free of|negative|uninvolved|involved by|involved|positive)\b",
                {"free of": "involved by", "negative": "positive", "uninvolved": "involved",
                 "involved by": "free of", "involved": "uninvolved", "positive": "negative"}),
    "lvi": (LVI + r"\W{1,3}(?:is |was )?(?P<w>not identified|identified|not present|present|absent|negative|positive)\b",
            {"not identified": "identified", "identified": "not identified", "not present": "present",
             "present": "not present", "absent": "present", "negative": "positive", "positive": "negative"}),
}

LIGHT = {"tumor": "tumour", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5", "six": "6",
         "surgical margins": "resection margins", "resection margins": "surgical margins"}
FULL = {**LIGHT, "infiltrating": "invasive", "invasive": "infiltrating", "not identified": "not seen",
        "identified": "seen", "negative for": "free of", "free of": "negative for",
        "lymphovascular invasion": "angiolymphatic invasion", "greatest dimension": "maximum dimension",
        "carcinoma in situ": "in situ carcinoma", "metastatic carcinoma": "metastatic tumour"}


def same_case(src, new):
    return new.upper() if src.isupper() else new[:1].upper() + new[1:] if src[:1].isupper() else new


def flip(text, pattern, to):
    """Replace group "w" in every match; returns (new_text, number_of_changes)."""
    def repl(m):
        w = m.group("w")
        new = to(w) if callable(to) else to[re.sub(r"\s+", " ", w.lower())]
        return m.group(0)[:m.start("w") - m.start()] + same_case(w, new) + m.group(0)[m.end("w") - m.start():]
    return re.subn(pattern, repl, text, flags=re.I)


def paraphrase(text, rules):
    """Swap every phrase in one pass, so a replacement is never replaced again."""
    keys = sorted(rules, key=len, reverse=True)
    pattern = "|".join(rf"\b{re.escape(k)}\b" for k in keys)
    return re.subn(pattern, lambda m: same_case(m.group(0), rules[m.group(0).lower()]), text, flags=re.I)


def main():
    csv.field_size_limit(10**9)
    with open(ROOT / "data" / "00_cases.csv") as f:
        cases = list(csv.DictReader(f))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows = []
    for case in cases:
        ref = case["reference"]
        variants = [("paraphrase_light", True, paraphrase(ref, LIGHT)),
                    ("paraphrase_full", True, paraphrase(ref, FULL))]
        variants += [(f"flip_{fact}", False, flip(ref, *FLIPS[fact])) for fact in FLIPS]
        for name, correct, (text, n_changes) in variants:
            if n_changes:  # skip a flip when the report doesn't state that fact in a flippable way
                rows.append({"case_id": case["case_id"], "variant": name, "correct": correct,
                             "n_changes": n_changes,
                             "rougeL": round(rouge.score(ref, text)["rougeL"].fmeasure, 4),
                             "bleu": round(sacrebleu.sentence_bleu(text, [ref]).score / 100, 4),
                             "text": text})

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "02_metric_paradox.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # summary per variant; an unchanged report would score 1.000
    print(f"{len(cases)} cases, {len(rows)} variants -> results/02_metric_paradox.csv\n")
    print(f"{'variant':<18}{'reports':>8}{'phrases changed':>17}{'ROUGE-L':>9}{'BLEU':>7}   flip beats paraphrase")
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        line = f"{name:<18}{len(rs):>8}{mean('n_changes'):>17.1f}{mean('rougeL'):>9.3f}{mean('bleu'):>7.3f}"
        if not rs[0]["correct"]:
            pairs = [(r, p) for r in rs for p in rows if p["case_id"] == r["case_id"] and p["correct"]]
            line += f"   {sum(r['rougeL'] > p['rougeL'] for r, p in pairs) / len(pairs):.0%}"
        print(line)
    plots.metric_paradox(rows, LABELS, ROOT / "results" / "02_metric_paradox.png")
    print("\nChart -> results/02_metric_paradox.png")


LABELS = {"paraphrase_light": "Spelling/format only", "paraphrase_full": "+ pathology synonyms",
          "flip_diagnosis": "Ductal <-> lobular", "flip_laterality": "Left <-> right", "flip_nodes": "Node count",
          "flip_margins": "Margins neg <-> pos", "flip_lvi": "LVI absent <-> present", "flip_grade": "Grade"}


if __name__ == "__main__":
    main()
