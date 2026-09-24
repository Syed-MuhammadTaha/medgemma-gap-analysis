"""Experiment 1 analysis: how often does MedGemma get the six critical facts right?

  uv run src/04_fact_analysis.py  ->  results/04_fact_accuracy.png

Reads the hand-made annotation table results/04_annotations.csv:
one row per case x fact with what the pathologist reported, what
MedGemma stated, a verdict, and MedGemma's exact words as evidence. Rows with
verdict "observed" record other failures seen while reading; they are not counted here.
Repetition loops are counted directly from results/03_medgemma_reports.csv.
"""

import csv
import re
from collections import Counter
from pathlib import Path

from utils import plots

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "results"
FACTS = {"diagnosis": "Diagnosis (ductal / lobular)", "grade": "Grade", "lvi": "Lymphovascular invasion",
         "margins": "Margins", "nodes": "Lymph node status", "laterality": "Left / right"}
VISIBLE = ["diagnosis", "grade", "lvi"]  # judged from tissue; the rest need other slides or clinical records
VERDICTS = {"correct": "Stated, matches pathologist", "wrong": "Stated, contradicts pathologist",
            "not stated": "Not stated", "no reference": "Pathologist's report has no clear answer"}


def read(name):
    with open(RESULTS / name) as f:
        return list(csv.DictReader(f))



def main():
    facts, reports = read("04_annotations.csv"), read("03_medgemma_reports.csv")
    counts = {f: Counter(r["verdict"] for r in facts if r["item"] == f) for f in FACTS}

    print(f"{'fact':<26}{'correct':>8}{'wrong':>7}{'not stated':>12}{'no ref':>8}   wrong when stated")
    for f, label in FACTS.items():
        c = counts[f]
        stated = c["correct"] + c["wrong"]
        rate = f"{c['wrong'] / stated:.0%} of {stated}" if stated else "-"
        print(f"{label:<26}{c['correct']:>8}{c['wrong']:>7}{c['not stated']:>12}{c['no reference']:>8}   {rate}")

    plots.fact_accuracy(counts, FACTS, VISIBLE, VERDICTS, len(reports), RESULTS / "04_fact_accuracy.png")
    print("Chart -> results/04_fact_accuracy.png")


if __name__ == "__main__":
    main()
