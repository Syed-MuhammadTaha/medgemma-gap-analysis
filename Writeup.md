> **A note on my background.** I am not a histopathologist. I am an ML/DL researcher who worked on report generation from whole-slide images in earlier research, and that is where the idea for this project comes from: the observation that reports can score well on text metrics while getting clinical facts wrong is my own, from that work. The pathology details in this write-up, such as which facts matter clinically, how they are graded and what the terms mean, were worked out with the help of an AI assistant and checked against published sources where I could. They have not been reviewed by a pathologist.

## Summary

Given tissue from a breast cancer slide, MedGemma writes a confident pathology report and fills in facts it has no way of knowing. The metric its model card uses for these reports, ROUGE, can't tell.

- **The model gets critical facts wrong.** When MedGemma states a fact that decides treatment, it contradicts the pathologist 41 to 67% of the time. It states a margin status in 15 of 53 cases, although one slide can't establish it, and 10 of those 15 are wrong (Exp 1).
- **The metric can't see it.** In pathologists' own reports, ROUGE-L scores a report with one flipped fact above a correct rewording in 92 to 100% of cases (Exp 2a).
- **Fixing the errors doesn't register.** Correcting every contradicted fact in MedGemma's reports moves ROUGE-L by +0.006 on average, and in 11 of 37 reports the corrected version scores lower (Exp 2b).

## Model

I evaluate MedGemma 1.5 4B (`google/medgemma-1.5-4b-it`).

Why MedGemma:

- It is built on Gemma 3, Google's open frontier model family, and adapted to medical text and images. So it shows how a current frontier model behaves once it is specialised for a domain.
- It has 4B parameters, inside the 0.6B to 6B range.
- Its image encoder saw histopathology, microscope images of tissue, during training. If it fails on tissue, it isn't because it has never seen tissue.
- Its model card reports one number for whole-slide pathology reports. That number is WSI-Path, 49.4, up from 2.2 for MedGemma 1, and it is scored with ROUGE only. Chest X-ray reports on the same card get RadGraph F1, a metric built on clinical facts. So the model's headline pathology result comes from exactly the metric I test in Exp 2. A whole-slide image (WSI) is a gigapixel scan of one glass tissue slide.
- It reads a whole slide as a set of patches. The technical report describes the pipeline: 896 px tissue patches on a grid, up to 126 per slide, kept in spatial order, at 5x, 10x or 20x magnification (5x is the most zoomed out). I copy that pipeline, so the test is fair to the model.

## Data

|         | Source                                                                                                                           | License     |
| ------- | -------------------------------------------------------------------------------------------------------------------------------- | ----------- |
| Reports | TCGA-Reports (Kefeli et al. 2024), 9,523 OCR'd pathology reports from The Cancer Genome Atlas (TCGA), a public US cancer dataset | CC BY 4.0   |
| Slides  | TCGA-BRCA (the breast cancer part of TCGA) diagnostic slides, NCI Genomic Data Commons                                           | Open access |

I kept one organ, breast, so the same six facts apply to every case. Of 1,062 breast cancer patients with an open slide, 53 have a report that states all six facts, is a reasonable length and has little OCR noise. The reference text is the report's final diagnosis section, which is what WSI-Path scores. 33 of the 53 reports have a detectable section; the other 20 use the full report.

For all 53 cases I cut patches the way MedGemma's technical report describes: 896 px patches on a grid over the tissue, randomly subsampled to a cap and kept in spatial order. I use 5x, one of Google's three magnifications, because each patch then covers about 1.8 mm of tissue, so the cap covers most of a slide. I cut up to 64 patches per slide (Google's cap is 126). Every experiment uses the same 53 cases.

## What counts as an error

| Fact                          | What it means                                                                                  | Example flip         | Determinable from one slide? |
| ----------------------------- | ---------------------------------------------------------------------------------------------- | -------------------- | ---------------------------- |
| Diagnosis                     | The cancer type. Ductal and lobular are the two main breast cancer types                       | ductal to lobular    | Yes                          |
| Lymphovascular invasion (LVI) | Cancer cells inside blood or lymph vessels, a sign it can spread                               | absent to present    | Sometimes                    |
| Margins                       | Whether cancer reaches the cut edge of the removed tissue. Positive means some was left behind | negative to positive | No                           |
| Lymph node count              | How many nearby lymph nodes contain cancer                                                     | 0/3 to 1/3           | No                           |
| Grade                         | How abnormal the cancer cells look, from 1 (close to normal) to 3                              | 2 to 3               | Yes                          |
| Laterality                    | Left or right breast                                                                           | left to right        | No                           |

The last column matters for Exp 1. Laterality isn't in the tissue at all, lymph nodes are on other slides, and margin status needs every edge of the removed tissue, not one slide. So any value the model states for these three is invented.

## Experiment 1: MedGemma states facts it can't know

### Setup

I fed each slide to MedGemma the way Google's technical report describes its own whole-slide pipeline. A tissue mask finds the tissue, a grid of 896 px patches is laid over it, the patches are randomly subsampled to a cap, and they stay in spatial order, row by row. Google picks 5x, 10x or 20x per slide. I use 5x for every slide, where one patch covers about 1.8 mm of tissue, so a few dozen patches span most of a slide. H&E is the standard pink-and-purple stain used on almost every tissue slide. All patches of a case go into one prompt with the instruction "These are tissue patches from one H&E slide of a breast surgical specimen, in spatial order. Write the FINAL DIAGNOSIS section of the pathology report." Decoding is greedy, up to 400 new tokens.

I ran inference on my own RTX 5070 (12 GB) in bf16, the model's native precision. My first run used 64 patches per slide and ran out of GPU memory. MedGemma's image encoder (SigLIP) processes every image in the prompt as one batch, and its MLP activation of shape (images, 4096, 4304) alone needs about 2 GB at 64 images. Two changes made it fit. I encode the images 4 at a time and concatenate the results; SigLIP encodes each image independently, so the language model receives exactly the same image tokens. And I cap each slide at 32 patches, picked evenly across the 64 so they still span the whole slide. That is a quarter of Google's cap of 126. Slides with little tissue have fewer patches: 38 of the 53 cases used 32, the rest 6 to 31.

### Scoring

Extracting facts from MedGemma's free text with keyword rules would be unreliable: its reports loop, bury stages inside lists and phrase things loosely. So the facts were annotated by reading. For every case I recorded, per fact, what the pathologist reported, what MedGemma stated, and a verdict, together with MedGemma's exact words as evidence (`results/04_annotations.csv`, 318 fact rows). The first pass was done with Claude Opus reading each generated report next to the reference; a script checks that every quoted phrase appears verbatim in MedGemma's output, and I spot-checked a sample of cases against the quotes (TODO: check ~10 cases and state the number here). Other failures seen while reading are in the same table, as rows marked "observed", also with quotes.

Verdicts: "matches" and "contradicts" compare MedGemma's statement with the pathologist's report. "Not stated" means MedGemma said nothing about the fact. "Invasive carcinoma" without a type counts as not stating the diagnosis. Grade counts only when stated for the invasive tumour (grade 1 to 3, or well, moderately or poorly differentiated), not for DCIS (pre-invasive cancer still confined to the ducts) or for descriptions of individual nuclei.

### Results

![What MedGemma says about six treatment-critical facts](results/04_fact_accuracy.png)

When MedGemma commits to a fact that decides treatment, it is wrong 41 to 67% of the time. The worst is margins: it states a margin status in 15 cases, although one slide can't establish the margin status of a whole specimen, and 10 of those 15 contradict the pathologist. Nearly always the invented margin is "tumor present at the inked surgical margin". Surgeons ink the edge of the removed tissue, so tumour at the ink means cancer was left in the patient, which would send them back to surgery.

### What else MedGemma does

Reading the reports also shows failure modes that any ML reader will recognise.

| Failure mode                                | Cases | What it looks like here                                                                                                                                   |
| ------------------------------------------- | ----- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Degenerate repetition under greedy decoding | 15    | One sentence repeated up to 26 times until the token limit                                                                                                |
| Misses the main finding                     | 8     | Calls a cancer benign: "Fibroadenoma."                                                                                                                    |
| States facts that aren't in the input       | 8     | A full cancer stage (T = tumour size, N = lymph nodes, M = spread to other organs), which needs scans and other slides the model never saw: "T4 N1(c) M0" |
| States results of tests it can't see        | 7     | Lab test results that need extra staining, not an image like these: "The tumor is triple negative."                                                       |
| Invents the worst-case outcome              | 4     | Says the cancer has spread to other organs: "T4 N0 M1"                                                                                                    |
| Claims inputs it never had                  | 4     | "based on clinical information provided", when the prompt had none                                                                                        |
| Leaks text from a neighbouring domain       | 4     | A mammography score in a pathology report: "BI-RADS Category 5:"                                                                                          |

Each case and its quote is listed in `results/04_annotations.csv` (rows marked "observed"); repetition loops are counted from the raw outputs.

Together with the results above, this points to one blind spot. MedGemma writes the report its language prior expects for a breast cancer case, and fills every section of that template whether or not the image supports it. The critical facts are exactly where this shows up, and they are one or two words long.

## Experiment 2: Why nobody notices

Exp 1 shows MedGemma getting treatment-critical facts wrong, yet its model card reports steady progress on whole-slide reports: WSI-Path went from 2.2 to 49.4, scored with ROUGE. This experiment tests whether ROUGE can see these errors at all, in two directions: flipping facts in the pathologists' own reports (2a), and fixing facts in MedGemma's reports (2b).

### 2a. ROUGE prefers a one-fact flip over a correct rewording

I took each of the 53 reference reports and made two kinds of variants.

- Correct rewordings. A light one changes only spelling and format ("tumor" to "tumour", "one" to "1"). A fuller one also swaps pathology synonyms ("infiltrating" for "invasive", "not seen" for "not identified").
- Wrong reports. Each one flips a single fact from the error table, everywhere it appears.

Then I scored every variant against the original with ROUGE-L (BLEU as a side check).

An unchanged report scores ROUGE-L 1.000 against itself. The table shows the score after each kind of change. "Reports affected" is how many of the 53 reports state that fact in a form the rules can flip, which is why it varies.

| Variant                         | Correct? | Reports affected | Phrases changed (avg) | ROUGE-L | Wrong report beats a correct rewording |
| ------------------------------- | -------- | ---------------- | --------------------- | ------- | -------------------------------------- |
| Spelling/format only            | Yes      | 52               | 10.6                  | 0.971   |                                        |
| + pathology synonyms            | Yes      | 53               | 22.0                  | 0.923   |                                        |
| Lymphovascular invasion flipped | No       | 28               | 1.3                   | 0.998   | 100%                                   |
| Grade flipped                   | No       | 31               | 1.4                   | 0.996   | 98%                                    |
| Node count flipped              | No       | 43               | 2.0                   | 0.993   | 99%                                    |
| Margins flipped                 | No       | 37               | 1.9                   | 0.992   | 100%                                   |
| Ductal/lobular flipped          | No       | 52               | 3.6                   | 0.989   | 94%                                    |
| Left/right flipped              | No       | 53               | 4.9                   | 0.984   | 92%                                    |

![ROUGE-L for correct rewordings vs wrong reports](results/02_metric_paradox.png)

The "phrases changed" column explains the result. ROUGE-L drops in proportion to how many words change, not to what they mean. A flip touches one or two words, so it barely moves the score. A correct rewording touches ten or more, so it loses more. Writing "tumour" instead of "tumor" costs more ROUGE-L than calling negative margins positive. Every flipped fact scores higher on average than either correct rewording, and in 92 to 100% of same-report pairs ROUGE-L prefers the wrong report.

### 2b. Fixing MedGemma's own errors doesn't move ROUGE

For each of the 52 facts MedGemma contradicted in Exp 1 (in 37 reports), I made the smallest edit that makes the claim match the pathologist, for example "N1" to "N0" or "moderately" to "poorly differentiated", everywhere the claim appears. The edits are listed in `results/05_corrections.csv`. Then I scored MedGemma's report as written and the corrected report against the pathologist's report.

|                                      | ROUGE-L | BLEU   |
| ------------------------------------ | ------- | ------ |
| MedGemma as written                  | 0.086   | 0.003  |
| After fixing every contradicted fact | 0.092   | 0.003  |
| Change                               | +0.006  | +0.000 |

Fixing every treatment-critical error raises ROUGE-L by 0.006 on average. In 11 of the 37 reports the corrected version scores lower than the wrong one. For scale, MedGemma's reports range from 0.030 to 0.175 ROUGE-L among themselves, so the difference between a report that sends a patient back to surgery and one that doesn't is lost in the noise of wording.

These ROUGE-L values are much lower than the 49.4 on Google's WSI-Path benchmark because the references differ: TCGA reports are long, OCR'd and full of template text, while WSI-Path uses its own curated final-diagnosis text. The absolute numbers aren't comparable. The point is the size of the change.

### What this means

2a and 2b look at the same blind spot from both sides. Near a perfect score, a wrong fact costs almost nothing; at MedGemma's real scores, a corrected fact gains almost nothing. A model can improve on ROUGE without getting a single margin, node or grade right, and it can get them all right without improving on ROUGE. So a WSI-Path score of 49.4 tells us how closely MedGemma copies pathologist wording, not whether its reports are correct, and the errors in Exp 1 stay invisible.

## Path forward

The experiments point to two problems: the model states facts its input doesn't support, and the metric used to score it can't tell. Fixing the model without fixing the metric means nobody would see the improvement, so the metric comes first.

**1. Score WSI reports on facts, not words.** Pathologists already fill in a structured checklist for each cancer, such as the CAP synoptic report: diagnosis, grade, margins, nodes and so on. A WSI report metric should extract those fields from the generated and reference reports, with an LLM or a trained tagger, and score them field by field, like RadGraph F1 does for chest X-rays. It should report three numbers, not one: accuracy on facts the input can support, how often the model states facts the input can't support, and how often it correctly says "not assessable". Exp 2a then becomes a unit test for any proposed metric: it must score a one-fact flip below a harmless rewording.

**2. Stop training the model to invent.** Slide-to-report training pairs a slide, or a sample of its patches, with a report written about the whole case, including lymph nodes on other slides and clinical history. The target contains information the input doesn't, which teaches the model to fill those fields from its language prior. That is exactly the behaviour in Exp 1. The data fix is to rewrite targets so that fields the input can't support read "not assessable from the provided tissue" instead of the case-level answer.

**3. Make the loss notice the words that matter.** Cross-entropy weighs "tumour" and "negative" the same, just like ROUGE. Two changes would help. Up-weight the loss on tokens that carry a checklist field (negations, numbers, grades, margin status, laterality). And train on counterfactual pairs: the flip generator from Exp 2a produces a wrong-by-one-fact copy of every report for free, which can serve as a hard negative, through an unlikelihood loss on the flipped report or as the rejected answer in DPO.

**4. Reinforcement learning with a checkable reward.** Because the checklist fields can be compared automatically, the reward can be computed rather than collected from raters, in the style of RL with verifiable rewards (for example GRPO): reward each correct field and each correct "not assessable", penalise stated fields the input can't support, and penalise repetition. Human or pathologist preference feedback (RLHF) is then only needed for the close calls the checklist can't settle.

**5. Generate the checklist first, then the prose.** Decode the structured fields first, constrained to valid values including "not assessable", with each field linked to the patches that support it, for example through the attention weights of a multiple-instance pooling layer. Then write the prose report from those fields. This makes every stated fact traceable to image evidence and removes the free-text loops seen in Exp 1.

## Limitations

- MedGemma's model card lists TCGA among its training data, so it may have seen these slides or reports. Exp 2a doesn't involve the model, so this doesn't affect it.
- The reports were OCR'd from scanned PDFs and contain typos.
- Flips and rewordings come from word-level rules, so a flip is skipped when a report states a fact in a form the rules don't cover. That is why the "Reports affected" column varies.
- The rewordings are mild. A real model's correct report would differ from the reference much more, so Exp 2a understates the problem.
- Random patches may contain no tumor. A tumor-targeted set would separate "can't see it" from "sees it and gets it wrong".
- MedGemma sees at most 32 patches at 5x per slide, fewer than Google's cap of 126 and at one magnification only, because of GPU memory.
- 53 cases is a small sample, suitable for showing the pattern, not for precise rates.
- The Exp 2b corrections are minimal word edits written by hand. They fix the claim but can leave a report internally inconsistent, for example a corrected stage next to an unchanged tumour size.
- Exp 1 facts were annotated by an LLM (Claude) with verbatim evidence and spot-checked by me (TODO: number of cases), not by a pathologist. Grading close calls ("high-grade carcinoma" as grade 3; IDC, invasive ductal carcinoma, versus "invasive carcinoma of no special type", which is the newer name for the same thing) follow the rules stated above.

## Code

| File                               | What it does                                                                                                                              |
| ---------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `src/01_prepare_data.py`           | Downloads reports, selects cases (`data/00_cases.csv`), streams slides and cuts patches (`data/patches/`)                                    |
| `src/02_metric_paradox.py`         | Experiment 2a: flips and rewordings of the reference reports, writes `results/02_metric_paradox.csv` and `.png`                          |
| `src/03_generate_reports.py`       | Experiment 1: MedGemma writes a report per case from up to 32 patches (images encoded 4 at a time), writes `results/03_medgemma_reports.csv` |
| `src/04_fact_analysis.py`          | Experiment 1 scoring: counts `results/04_annotations.csv`, draws `results/04_fact_accuracy.png`                                           |
| `src/05_correct_and_rescore.py`    | Experiment 2b: fixes MedGemma's contradicted facts using `results/05_corrections.csv` and prints the rescored summary                     |
| `src/utils/plots.py`               | All charts, in one shared style                                                                                                           |
| `src/utils/dataset_download.ipynb` | Runs `01_prepare_data.py` on Colab's fast connection                                                                                      |
