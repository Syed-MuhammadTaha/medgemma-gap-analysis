"""Experiment 1: MedGemma writes a final diagnosis for each case from its patches.

  uv run src/03_generate_reports.py  ->  results/03_medgemma_reports.csv

Each case's patches (from 01_prepare_data.py) go into one prompt, in spatial order.
Runs in bf16 on a 12 GB RTX 5070 with two memory fixes: images are encoded 4 at
a time (same output, much lower peak) and at most 32 patches per case.
"""

import csv
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path(__file__).parent.parent
MODEL_ID = "google/medgemma-1.5-4b-it"
MAX_IMAGES = 32
VISION_BATCH = 4
SYSTEM = "You are an expert pathologist."
PROMPT = ("These are tissue patches from one H&E slide of a breast surgical specimen, in spatial order. "
          "Write the FINAL DIAGNOSIS section of the pathology report.")


def evenly_spaced(items, k):
    """Keep k items spread across the list, preserving spatial order."""
    return items if len(items) <= k else [items[round(i * (len(items) - 1) / (k - 1))] for i in range(k)]


def encode_images_in_batches(model):
    """MedGemma encodes all images of a prompt in one batch, and SigLIP's MLP activation
    (n_images, 4096, 4304) runs out of memory. Encode VISION_BATCH images at a time and
    concatenate: images are encoded independently, so the output is the same."""
    encode = model.model.get_image_features

    def batched(pixel_values, **kwargs):
        parts = [encode(pixel_values[i:i + VISION_BATCH], **kwargs) for i in range(0, len(pixel_values), VISION_BATCH)]
        parts[0].pooler_output = torch.cat([p.pooler_output for p in parts])
        return parts[0]

    model.model.get_image_features = batched


def main():
    model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
    encode_images_in_batches(model)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    csv.field_size_limit(10**9)
    with open(ROOT / "data" / "00_cases.csv") as f:
        references = {c["case_id"]: c["reference"] for c in csv.DictReader(f)}

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / "03_medgemma_reports.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["case_id", "n_patches", "generated", "reference"])
        writer.writeheader()
        for case_dir in sorted((ROOT / "data" / "patches").iterdir()):
            paths = evenly_spaced(sorted(case_dir.glob("patch_*.jpg")), MAX_IMAGES)
            images = [{"type": "image", "image": Image.open(p).convert("RGB")} for p in paths]
            messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                        {"role": "user", "content": [{"type": "text", "text": PROMPT}] + images}]
            inputs = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                                   return_dict=True, return_tensors="pt").to(model.device, dtype=torch.bfloat16)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=400, do_sample=False)  # greedy
            report = processor.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

            writer.writerow({"case_id": case_dir.name, "n_patches": len(paths),
                             "generated": report, "reference": references[case_dir.name]})
            f.flush()
            print(f"=== {case_dir.name} ({len(paths)} patches)\n{report}\n", flush=True)


if __name__ == "__main__":
    main()
