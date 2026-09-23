"""Experiment 2, step 1: MedGemma writes a report for each case from its patches.

  uv run generate_reports.py                    # bf16, up to 64 patches per case
  uv run generate_reports.py --max-images 32    # if the GPU runs out of memory
  uv run generate_reports.py --8bit             # or: 8-bit weights, all patches

Reads data/patches/<case>/ (from prepare_data.py) and data/cases.csv.
Writes results/medgemma_reports.csv, one row per case, saved as it goes and
resumable: cases already in the file are skipped.
Needs a Hugging Face token with access to google/medgemma-1.5-4b-it
(`hf auth login`, or the HF_TOKEN environment variable).
"""

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

ROOT = Path(__file__).parent
OUT = ROOT / "results" / "medgemma_reports.csv"
MODEL_ID = "google/medgemma-1.5-4b-it"
SYSTEM = "You are an expert pathologist."
PROMPT = ("These are tissue patches from one H&E slide of a breast surgical specimen, in spatial order. "
          "Write the FINAL DIAGNOSIS section of the pathology report.")


def evenly_spaced(items, k):
    """Keep k items spread across the list, preserving spatial order."""
    return items if len(items) <= k else [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--8bit", dest="eight_bit", action="store_true", help="load weights in 8-bit to save memory")
    args = parser.parse_args()

    quant = BitsAndBytesConfig(load_in_8bit=True) if args.eight_bit else None
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16,
                                                        device_map="auto", quantization_config=quant)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    csv.field_size_limit(10**9)
    with open(ROOT / "data" / "cases.csv") as f:
        references = {c["case_id"]: c["reference"] for c in csv.DictReader(f)}
    done = set()
    if OUT.exists():
        with open(OUT) as f:
            done = {r["case_id"] for r in csv.DictReader(f)}
    OUT.parent.mkdir(exist_ok=True)

    with open(OUT, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "n_patches", "weights", "generated", "reference"])
        if not done:
            writer.writeheader()
        for case_dir in sorted((ROOT / "data" / "patches").iterdir()):
            if case_dir.name in done or not (case_dir / "thumbnail.jpg").exists():
                continue
            paths = evenly_spaced(sorted(case_dir.glob("patch_*.jpg")), args.max_images)
            messages = [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                {"role": "user", "content": [{"type": "text", "text": PROMPT}]
                 + [{"type": "image", "image": Image.open(p).convert("RGB")} for p in paths]},
            ]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=400, do_sample=False)
            report = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
            torch.cuda.empty_cache()

            writer.writerow({"case_id": case_dir.name, "n_patches": len(paths),
                             "weights": "8bit" if args.eight_bit else "bf16",
                             "generated": report, "reference": references[case_dir.name]})
            f.flush()
            print(f"=== {case_dir.name} ({len(paths)} patches)\n{report}\n", flush=True)


if __name__ == "__main__":
    main()
