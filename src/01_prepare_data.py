"""Build the evaluation set: breast cancer reports paired with tissue patches.

  uv run prepare_data.py  ->  data/cases.csv + data/patches/<case>/

1. Download TCGA-Reports (Kefeli et al. 2024, CC BY 4.0): 9,523 OCR'd reports.
2. Keep TCGA-BRCA cases that have an open diagnostic slide on GDC and whose
   report states all six critical facts. Save the FINAL DIAGNOSIS section as
   the reference (MedGemma's WSI-Path benchmark scores final diagnosis text).
3. Stream each slide (~1 GB) from GDC and cut patches the way the
   MedGemma 1.5 technical report describes: 896px patches on a grid over the
   tissue mask, randomly subsampled to a cap, kept in spatial (row-by-row)
   order. We use 5x, one of Google's three magnifications. The cap is
   64 (Google uses 126). Then delete the slide.
   Resumable: a case folder with thumbnail.jpg is finished.
"""

import csv
import io
import json
import re
import shutil
import urllib.request
import zipfile
from pathlib import Path

DATA = Path(__file__).parent / "data"
REPORTS_URL = "https://github.com/jkefeli/tcga-path-reports/raw/main/TCGA_Reports.csv.zip"
GDC = "https://api.gdc.cancer.gov"

MAX_PATCHES, PATCH_PX, TARGET_MPP = 64, 896, 2.0  # 896px = MedGemma input size; 2 um/px = ~5x
MIN_TISSUE = 0.5  # a grid cell is kept if at least half of it is tissue

CRITICAL_FACTS = {
    "diagnosis": r"\b(ductal|lobular)\s+carcinoma",
    "grade": r"(nottingham|histologic grade|grade\s*[123]|grade:?\s*i{1,3}\b)",
    "margins": r"margins?[^.]{0,60}(negative|free|uninvolved|positive|involved|close)",
    "nodes": r"\b\d{1,2}\s*/\s*\d{1,2}\b",
    "lvi": r"(lymphovascular|angiolymphatic|vascular)\s+invasion",
    "laterality": r"\b(left|right)\s+breast",
}


def download_reports():
    path = DATA / "TCGA_Reports.csv"
    if not path.exists():
        print("Downloading TCGA-Reports ...")
        zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(REPORTS_URL).read())).extract(path.name, DATA)
    csv.field_size_limit(10**9)
    with open(path) as f:
        return {r["patient_filename"].split(".")[0]: r["text"] for r in csv.DictReader(f)}


def brca_slides():
    """Smallest open diagnostic slide per TCGA-BRCA patient."""
    filters = {"op": "and", "content": [
        {"op": "=", "content": {"field": f, "value": v}} for f, v in [
            ("cases.project.project_id", "TCGA-BRCA"), ("data_type", "Slide Image"),
            ("experimental_strategy", "Diagnostic Slide"), ("access", "open")]]}
    body = json.dumps({"filters": filters, "size": 5000,
                       "fields": "file_id,file_name,file_size,cases.submitter_id"}).encode()
    req = urllib.request.Request(f"{GDC}/files", body, {"Content-Type": "application/json"})
    slides = {}
    for s in json.load(urllib.request.urlopen(req))["data"]["hits"]:
        case = s["cases"][0]["submitter_id"]
        if case not in slides or s["file_size"] < slides[case]["file_size"]:
            slides[case] = s
    return slides


def diagnosis_section(text):
    """Text after 'FINAL DIAGNOSIS:' (or 'Diagnosis:') up to the gross/microscopic description."""
    starts = [(("final" not in text[m.start() - 15:m.start()].lower()), m.end())
              for m in re.finditer(r"\bdiagnosis\s*:", text, re.I)
              if not re.search(r"\b(pre|post|op)\b|outside|tissue|clinical", text[m.start() - 15:m.start()], re.I)]
    if not starts:
        return text, "full_report"
    start = min(starts)[1]
    end = re.compile(r"\b(gross|microscopic)\s+description|clinical\s+(history|information)|"
                     r"intraoperative|electronically\s+signed|comment\s*:", re.I).search(text, start)
    section = text[start:end.start() if end else None].strip(" .:-")
    return (section, "diagnosis_section") if len(section.split()) >= 40 else (text, "full_report")


def ocr_noise(text):
    words = re.findall(r"[a-z]+", text.lower())
    return sum(len(w) <= 2 for w in words) / max(len(words), 1)


def build_cases():
    reports, slides = download_reports(), brca_slides()
    cases = []
    for case, slide in slides.items():
        text = reports.get(case, "")
        if not (800 < len(text) < 6000 and ocr_noise(text) < 0.25):
            continue
        if not all(re.search(p, text, re.I) for p in CRITICAL_FACTS.values()):
            continue
        reference, source = diagnosis_section(text)
        cases.append({"case_id": case, "slide_file_id": slide["file_id"], "slide_file_name": slide["file_name"],
                      "slide_gb": round(slide["file_size"] / 1e9, 3), "noise": ocr_noise(text),
                      "reference_source": source, "reference": reference})
    cases.sort(key=lambda c: c.pop("noise"))  # cleanest OCR first
    with open(DATA / "cases.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cases[0].keys())
        w.writeheader()
        w.writerows(cases)
    print(f"{len(slides)} BRCA patients with slides -> {len(cases)} cases stating all six facts -> data/cases.csv")
    return cases


def download(case, svs, tries=3):
    """Stream a slide from GDC; retry if the connection drops before the full file arrives."""
    part = svs.with_suffix(".part")
    for attempt in range(1, tries + 1):
        print(f"  downloading {case['slide_gb']} GB (try {attempt}) ...", flush=True)
        try:
            with urllib.request.urlopen(f"{GDC}/data/{case['slide_file_id']}") as r, open(part, "wb") as f:
                expected = int(r.headers.get("Content-Length", 0))
                shutil.copyfileobj(r, f, length=8 << 20)
            if not expected or part.stat().st_size == expected:
                part.rename(svs)
                return
            print(f"  incomplete: {part.stat().st_size / 1e9:.2f} of {expected / 1e9:.2f} GB")
        except OSError as e:
            print(f"  download error: {e}")
    part.unlink(missing_ok=True)


def extract_patches(case):
    import numpy as np
    import openslide
    from PIL import Image

    out = DATA / "patches" / case["case_id"]
    if (out / "thumbnail.jpg").exists():
        return
    out.mkdir(parents=True, exist_ok=True)
    svs = DATA / case["slide_file_name"]
    if not svs.exists():
        download(case, svs)
    try:
        slide = openslide.OpenSlide(str(svs))
    except openslide.OpenSlideError as e:
        svs.unlink(missing_ok=True)
        print(f"  SKIPPED, slide won't open ({e}); rerun to retry")
        return
    mpp = float(slide.properties.get(openslide.PROPERTY_NAME_MPP_X, 0.25))
    size0 = round(PATCH_PX * TARGET_MPP / mpp)  # patch size in full-resolution pixels
    thumb = slide.get_thumbnail((2048, 2048)).convert("RGB")
    scale = slide.dimensions[0] / thumb.size[0]
    cell = max(1, round(size0 / scale))

    # tissue = stained (saturated) and not glass-white; grid cells that are mostly tissue
    rgb = np.asarray(thumb, dtype=np.float32) / 255
    tissue = (rgb.max(2) - rgb.min(2) > 0.07) & (rgb.mean(2) < 0.9)
    cells = [(r, c) for r in range(tissue.shape[0] // cell) for c in range(tissue.shape[1] // cell)
             if tissue[r * cell:(r + 1) * cell, c * cell:(c + 1) * cell].mean() >= MIN_TISSUE]
    pick = sorted(np.random.default_rng(0).choice(len(cells), size=min(MAX_PATCHES, len(cells)), replace=False))

    # read from the pyramid level closest to 5x instead of full resolution (much faster)
    level = slide.get_best_level_for_downsample(size0 / PATCH_PX)
    read_px = round(size0 / slide.level_downsamples[level])
    for i, idx in enumerate(pick):  # sorted indices = row-by-row spatial order
        r, c = cells[idx]
        x, y = round(c * cell * scale), round(r * cell * scale)
        patch = slide.read_region((x, y), level, (read_px, read_px)).convert("RGB")
        patch.resize((PATCH_PX, PATCH_PX), Image.LANCZOS).save(out / f"patch_{i:02d}_x{x}_y{y}.jpg", quality=95)
    slide.close()
    svs.unlink()
    thumb.save(out / "thumbnail.jpg", quality=90)  # written last = case finished
    print(f"  {len(pick)} of {len(cells)} tissue patches saved, slide deleted")


def main():
    DATA.mkdir(exist_ok=True)
    if (DATA / "cases.csv").exists():
        with open(DATA / "cases.csv") as f:
            cases = list(csv.DictReader(f))
    else:
        cases = build_cases()
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['case_id']}", flush=True)
        extract_patches(case)


if __name__ == "__main__":
    main()
