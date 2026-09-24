"""Experiment 2b: does ROUGE reward fixing MedGemma's real mistakes?

  uv run src/05_correct_and_rescore.py  ->  summary table

Exp 2a flipped facts in the pathologist's reports. Here the errors are MedGemma's own.
For every fact MedGemma contradicted (04_annotations.csv), results/05_corrections.csv
holds the smallest edit that makes it right, e.g. "N1" -> "N0" or "moderately" ->
"poorly differentiated", applied everywhere the claim appears. Both versions of the
report are scored against the pathologist's report. If ROUGE-L barely moves, the
metric can't tell MedGemma's wrong report from a corrected one.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"


def read(name):
    csv.field_size_limit(10**9)
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))


def correct(text, rules):
    for r in rules:
        text, n = re.subn(r["find"], r["replace"], text)
        assert n, f"{r['case_id']} {r['fact']}: '{r['find']}' not found"
    return text


def main():
    reports = {r["case_id"]: r for r in read("03_medgemma_reports.csv")}
    rules = defaultdict(list)
    for r in read("05_corrections.csv"):
        rules[r["case_id"]].append(r)
    rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    score = lambda text, ref: (rouge.score(ref, text)["rougeL"].fmeasure, sacrebleu.sentence_bleu(text, [ref]).score / 100)

    rows = []
    for case, case_rules in rules.items():
        generated, ref = reports[case]["generated"], reports[case]["reference"]
        (r0, b0), (r1, b1) = score(generated, ref), score(correct(generated, case_rules), ref)
        rows.append({"case_id": case, "facts_fixed": " ".join(sorted({r["fact"] for r in case_rules})),
                     "rougeL_medgemma": round(r0, 4), "rougeL_corrected": round(r1, 4), "rougeL_change": round(r1 - r0, 4),
                     "bleu_medgemma": round(b0, 4), "bleu_corrected": round(b1, 4), "bleu_change": round(b1 - b0, 4)})

    mean = lambda k, rs=rows: sum(r[k] for r in rs) / len(rs)
    all_rouge = [score(r["generated"], r["reference"])[0] for r in reports.values()]
    n_facts = sum(len(r["facts_fixed"].split()) for r in rows)
    print(f"{len(rows)} reports with at least one contradicted fact ({n_facts} facts fixed)\n")
    print(f"{'':<22}{'ROUGE-L':>9}{'BLEU':>8}")
    print(f"{'MedGemma as written':<22}{mean('rougeL_medgemma'):>9.3f}{mean('bleu_medgemma'):>8.3f}")
    print(f"{'after fixing errors':<22}{mean('rougeL_corrected'):>9.3f}{mean('bleu_corrected'):>8.3f}")
    print(f"{'change':<22}{mean('rougeL_change'):>+9.3f}{mean('bleu_change'):>+8.3f}")
    print(f"\nLargest ROUGE-L gain from a fix: {max(r['rougeL_change'] for r in rows):+.3f}")
    print(f"Reports where the fix lowered ROUGE-L: {sum(r['rougeL_change'] < 0 for r in rows)} of {len(rows)}")
    print(f"For scale: MedGemma's ROUGE-L across all {len(all_rouge)} reports ranges "
          f"{min(all_rouge):.3f} to {max(all_rouge):.3f}")


if __name__ == "__main__":
    main()
