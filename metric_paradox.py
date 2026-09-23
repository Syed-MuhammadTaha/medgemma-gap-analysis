"""Experiment 1: does ROUGE notice a treatment-changing error?

ROUGE is the only metric the MedGemma 1.5 model card reports for whole-slide
pathology reports (WSI-Path: 49.4). For each reference in data/cases.csv we
make two kinds of variants and score them against the reference:

  correct  paraphrase_light  spelling/format only (tumor -> tumour, "one" -> "1")
           paraphrase_full   + pathology synonyms (invasive -> infiltrating, ...)
  wrong    flip_<fact>       one clinical fact flipped everywhere it appears

Paradox: a wrong report outscoring a correct paraphrase of the same case.

  uv run metric_paradox.py   ->  results/metric_paradox.csv
"""

import csv
import re
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).parent
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
SEVERITY = {"diagnosis": "critical", "lvi": "critical", "margins": "critical", "nodes": "critical",
            "grade": "high", "laterality": "high"}

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
    with open(ROOT / "data" / "cases.csv") as f:
        cases = list(csv.DictReader(f))
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)

    rows = []
    for case in cases:
        ref = case["reference"]
        variants = [("paraphrase_light", True, "", paraphrase(ref, LIGHT)),
                    ("paraphrase_full", True, "", paraphrase(ref, FULL))]
        variants += [(f"flip_{fact}", False, SEVERITY[fact], flip(ref, *FLIPS[fact])) for fact in FLIPS]
        for name, correct, severity, (text, n_changes) in variants:
            if n_changes:  # skip a flip when the report doesn't state that fact in a flippable way
                rows.append({"case_id": case["case_id"], "variant": name, "correct": correct,
                             "severity": severity, "n_changes": n_changes,
                             "rougeL": round(rouge.score(ref, text)["rougeL"].fmeasure, 4),
                             "bleu": round(sacrebleu.sentence_bleu(text, [ref]).score / 100, 4),
                             "text": text})

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "metric_paradox.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # summary per variant; an unchanged report would score 1.000
    print(f"{len(cases)} cases, {len(rows)} variants -> results/metric_paradox.csv\n")
    print(f"{'variant':<18}{'reports':>8}{'phrases changed':>17}{'ROUGE-L':>9}{'BLEU':>7}   flip beats paraphrase")
    for name in dict.fromkeys(r["variant"] for r in rows):
        rs = [r for r in rows if r["variant"] == name]
        mean = lambda k: sum(r[k] for r in rs) / len(rs)
        line = f"{name:<18}{len(rs):>8}{mean('n_changes'):>17.1f}{mean('rougeL'):>9.3f}{mean('bleu'):>7.3f}"
        if not rs[0]["correct"]:
            pairs = [(r, p) for r in rs for p in rows if p["case_id"] == r["case_id"] and p["correct"]]
            line += f"   {sum(r['rougeL'] > p['rougeL'] for r, p in pairs) / len(pairs):.0%}"
        print(line)
    plot(rows)
    print("\nChart -> results/metric_paradox.png")


LABELS = {"paraphrase_light": "Spelling/format only", "paraphrase_full": "+ pathology synonyms",
          "flip_diagnosis": "Ductal <-> lobular", "flip_laterality": "Left <-> right", "flip_nodes": "Node count",
          "flip_margins": "Margins neg <-> pos", "flip_lvi": "LVI absent <-> present", "flip_grade": "Grade"}


def plot(rows):
    """Dot plot: one row per variant, one dot per report, big dot = mean, dashed line = unchanged report."""
    import matplotlib.pyplot as plt
    import numpy as np

    blue, orange, ink, muted, surface = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#fcfcfb"
    mean = lambda v: np.mean([r["rougeL"] for r in rows if r["variant"] == v])
    correct = [v for v in LABELS if v.startswith("paraphrase")]
    wrong = sorted((v for v in LABELS if v.startswith("flip")), key=mean)
    order = correct + wrong

    fig, ax = plt.subplots(figsize=(9, 5.2), facecolor=surface)
    ax.set_facecolor(surface)
    rng = np.random.default_rng(0)
    for y, v in enumerate(order):
        color = blue if v in correct else orange
        xs = [r["rougeL"] for r in rows if r["variant"] == v]
        ax.scatter(xs, y + rng.uniform(-0.18, 0.18, len(xs)), s=14, color=color, alpha=0.35, linewidths=0)
        ax.scatter(mean(v), y, s=90, color=color, edgecolors=surface, linewidths=2, zorder=3)
        ax.annotate(f"{mean(v):.3f}", (mean(v), y), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=8.5, color=ink)

    ax.axvline(1.0, color=muted, linestyle="--", linewidth=1)
    ax.text(0.998, -0.75, "unchanged report = 1.0", ha="right", va="center", fontsize=8.5, color=muted)
    ax.axhline(len(correct) - 0.5, color="#e4e3df", linewidth=1)
    for y, text, color in [(-0.75, "Correct rewordings (facts unchanged)", blue),
                           (len(correct) - 0.25, "Wrong reports (one fact flipped)", orange)]:
        ax.text(0.005, y, text, transform=ax.get_yaxis_transform(), fontsize=9, color=color,
                va="center", weight="bold")

    ax.set_yticks(range(len(order)), [LABELS[v] for v in order], fontsize=9.5, color=ink)
    ax.set_ylim(len(order) - 0.5, -1.0)  # correct rewordings on top
    ax.set_xlim(min(r["rougeL"] for r in rows) - 0.01, 1.008)
    ax.set_xlabel("ROUGE-L against the original report (higher = judged more similar)", fontsize=9.5, color=muted)
    ax.tick_params(colors=muted, length=0)
    ax.grid(axis="x", color="#e4e3df", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.suptitle("ROUGE-L scores wrong pathology reports above correct ones", x=0.02, ha="left",
                 fontsize=13, weight="bold", color=ink)
    fig.text(0.02, 0.905, f"{len({r['case_id'] for r in rows})} TCGA breast cancer reports. "
             "Each dot is one report; the large dot is the mean.", fontsize=9, color=muted)
    fig.tight_layout(rect=(0, 0, 0.99, 0.9))
    fig.savefig(ROOT / "results" / "metric_paradox.png", dpi=200, facecolor=surface)


if __name__ == "__main__":
    main()
