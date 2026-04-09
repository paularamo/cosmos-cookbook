# Caption Generation — Cosmos Reason2 + Metadata Injection

Detailed reference for the caption pipeline: Cosmos Reason2 setup, prompt template, metadata injection logic, and output format.

---

## Model

| Model | HuggingFace ID | VRAM | Best For |
|---|---|---|---|
| Cosmos-Reason2-8B | `nvidia/Cosmos-Reason2-8B` | 32 GB | Best quality, large GPU |
| Cosmos-Reason2-2B | `nvidia/Cosmos-Reason2-2B` | 24 GB | Memory-constrained |
| Cosmos-Reason1-7B | `nvidia/Cosmos-Reason1-7B` | 24 GB | Legacy / Qwen2.5-VL base |

**Input FPS for video:** always use `fps=4` (4 frames sampled per second — this is what the model was trained on, even though your clip runs at 16 FPS).

---

## Installation

```bash
pip install transformers>=4.57.0 torch accelerate av
huggingface-cli login   # accept NVIDIA Open Model License
```

---

## Prompt Template

The system prompt primes Cosmos Reason to produce a training-quality description.
The user prompt includes a metadata block prepended before the video question.

```python
SYSTEM_PROMPT = """You are a precise video captioning assistant for robotics and field robotics training data.
Describe the video content in factual, detailed language suitable for training a video generation model.
Focus on: scene environment, visible objects, camera perspective, motion patterns, lighting, and spatial relationships, ego object around the camera visible in the image.
Be concise. Do not speculate beyond what is visible. Maximum 200 words."""

USER_PROMPT_TEMPLATE = """Context metadata:
- Weather: {weather}
- Time of day: {hour_of_day}
- Turf variety: {turf_variety}
- Objects detected in scene: {object_list}

Describe this video clip in detail, incorporating the context above.
Include: camera angle, visible objects and their positions, motion, lighting conditions, and any notable actions."""
```

When a metadata field is absent it is omitted from the context block (not rendered as "None").

---

## Caption Assembly Logic

```
Final caption = metadata_prefix + " " + cosmos_reason_description
```

**Metadata prefix** (injected before the VLM text):
```
{turf_variety} field under {weather} conditions at {hour_of_day}. Objects present: {object_list}.
```

Example assembled caption (≤ 300 words):
```
Bermuda grass field under overcast conditions at 14:30. Objects present: grass, vehicle, person.
A wide-angle camera mounted on a ground vehicle captures a close-up view of the turf surface.
A robotic arm end-effector descends slowly toward a small white object resting on the grass.
The motion is steady and precise. Background includes a parked utility vehicle on the left.
Lighting is diffuse with no harsh shadows. Camera is fixed, no pan or tilt.
```

---

## Object Extraction from Segmentation Mask

The seg mask MP4 is decoded frame-by-frame; unique pixel colors are extracted
and mapped to class names via `colormap.json`.

**Default colormap** (COCO-style subset):
```json
{
  "0,128,0":    "grass",
  "128,64,128": "road",
  "70,70,70":   "building",
  "220,20,60":  "person",
  "0,0,142":    "vehicle",
  "107,142,35": "vegetation",
  "119,11,32":  "motorcycle",
  "0,60,100":   "bus",
  "0,80,100":   "train",
  "0,0,230":    "bicycle",
  "255,0,0":    "fire hydrant",
  "128,0,0":    "wall"
}
```

Logic:
1. Sample N frames evenly from the seg mask video (default N=5)
2. Quantize each frame's unique colors to nearest colormap entry (tolerance 15)
3. Union of classes across all sampled frames → `object_list`
4. Sort by frequency (most prominent first)
5. Cap at 8 objects to keep caption concise

---

## generate_captions.py — Full Script

```python
#!/usr/bin/env python3
"""
generate_captions.py — Cosmos Reason2 video captioning with metadata injection.

Usage:
    python scripts/generate_captions.py \
        --videos_dir   dataset_root/videos/ \
        --metadata_dir dataset_root/metadata/ \
        --seg_dir      dataset_root/control_seg/ \
        --colormap     dataset_root/colormap.json \
        --output_dir   dataset_root/captions/ \
        --model        nvidia/Cosmos-Reason2-8B \
        --device       cuda \
        --fps          4
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


# ── Colormap helpers ──────────────────────────────────────────────────────────

COCO_COLORMAP = {
    (0, 128, 0):    "grass",
    (128, 64, 128): "road",
    (70, 70, 70):   "building",
    (220, 20, 60):  "person",
    (0, 0, 142):    "vehicle",
    (107, 142, 35): "vegetation",
    (119, 11, 32):  "motorcycle",
    (0, 60, 100):   "bus",
    (0, 80, 100):   "train",
    (0, 0, 230):    "bicycle",
}


def load_colormap(path: str | None) -> dict:
    if not path or not Path(path).exists():
        return COCO_COLORMAP
    raw = json.loads(Path(path).read_text())
    return {tuple(int(x) for x in k.split(",")): v for k, v in raw.items()}


def nearest_color(pixel: tuple, colormap: dict, tol: int = 15) -> str | None:
    r, g, b = pixel
    best, best_dist = None, float("inf")
    for (cr, cg, cb), name in colormap.items():
        d = (r - cr) ** 2 + (g - cg) ** 2 + (b - cb) ** 2
        if d < best_dist:
            best_dist = d
            best = name
    return best if best_dist <= tol * tol * 3 else None


def extract_objects_from_seg(seg_path: str, colormap: dict, n_samples: int = 5) -> list[str]:
    cap = cv2.VideoCapture(seg_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        cap.release()
        return []

    indices = [int(i * total / n_samples) for i in range(n_samples)]
    counts: dict[str, int] = {}

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame_rgb.shape[:2]
        # Sample every 8th pixel for speed
        pixels = frame_rgb[::8, ::8].reshape(-1, 3)
        for px in pixels:
            name = nearest_color(tuple(int(x) for x in px), colormap)
            if name:
                counts[name] = counts.get(name, 0) + 1

    cap.release()
    sorted_objects = sorted(counts, key=counts.get, reverse=True)
    return sorted_objects[:8]


# ── Metadata loading ───────────────────────────────────────────────────────────

def load_metadata(metadata_dir: str, stem: str) -> dict:
    path = Path(metadata_dir) / f"{stem}.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


# ── Cosmos Reason inference ────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a precise video captioning assistant for robotics and field robotics training data. "
    "Describe the video content in factual, detailed language suitable for training a video generation model. "
    "Focus on: scene environment, visible objects, camera perspective, motion patterns, lighting, and spatial relationships. "
    "Be concise. Do not speculate beyond what is visible. Maximum 200 words."
)

USER_PROMPT_TEMPLATE = """\
Context metadata:
{meta_block}
Describe this video clip in detail, incorporating the context above.
Include: camera angle, visible objects and their positions, motion, lighting conditions, and any notable actions."""


def build_meta_block(meta: dict, objects: list[str]) -> str:
    lines = []
    if meta.get("weather"):
        lines.append(f"- Weather: {meta['weather']}")
    if meta.get("hour_of_day"):
        lines.append(f"- Time of day: {meta['hour_of_day']}")
    if meta.get("turf_variety"):
        lines.append(f"- Turf variety: {meta['turf_variety']}")
    if objects:
        lines.append(f"- Objects detected in scene: {', '.join(objects)}")
    if meta.get("location"):
        lines.append(f"- Location: {meta['location']}")
    if meta.get("custom_tags"):
        lines.append(f"- Tags: {', '.join(meta['custom_tags'])}")
    return "\n".join(lines) if lines else "(no metadata available)"


def build_metadata_prefix(meta: dict, objects: list[str]) -> str:
    parts = []
    if meta.get("turf_variety") and meta.get("weather") and meta.get("hour_of_day"):
        parts.append(
            f"{meta['turf_variety']} field under {meta['weather']} conditions "
            f"at {meta['hour_of_day']}."
        )
    elif meta.get("weather"):
        parts.append(f"{meta['weather']} conditions.")
    if objects:
        parts.append(f"Objects present: {', '.join(objects)}.")
    return " ".join(parts)


def run_cosmos_reason(video_path: str, meta: dict, objects: list[str],
                      model, processor, fps: int = 4) -> str:
    meta_block = build_meta_block(meta, objects)
    user_text = USER_PROMPT_TEMPLATE.format(meta_block=meta_block)

    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": f"file://{os.path.abspath(video_path)}", "fps": fps},
            {"type": "text", "text": user_text},
        ]},
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
        fps=fps,
    ).to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.05,
            do_sample=True,
        )

    input_len = inputs["input_ids"].shape[1]
    trimmed = generated_ids[:, input_len:]
    text = processor.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

    # Strip <think>...</think> reasoning block if present
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    return text


# ── Caption assembly ───────────────────────────────────────────────────────────

def word_count(text: str) -> int:
    return len(text.split())


def truncate_to_words(text: str, max_words: int = 290) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def assemble_caption(meta: dict, objects: list[str], description: str) -> dict:
    prefix = build_metadata_prefix(meta, objects)
    prompt = f"{prefix} {description}".strip() if prefix else description
    prompt = truncate_to_words(prompt, max_words=290)

    negative_parts = ["blurry", "overexposed", "motion blur"]
    if meta.get("weather") in ("sunny", "midday"):
        negative_parts.append("harsh shadows")
    negative_prompt = ", ".join(negative_parts)

    return {"prompt": prompt, "negative_prompt": negative_prompt}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate captions using Cosmos Reason2")
    parser.add_argument("--videos_dir",   required=True)
    parser.add_argument("--metadata_dir", required=True)
    parser.add_argument("--seg_dir",      default=None, help="Segmentation mask dir (optional)")
    parser.add_argument("--colormap",     default=None, help="Path to colormap.json")
    parser.add_argument("--output_dir",   required=True)
    parser.add_argument("--model",        default="nvidia/Cosmos-Reason2-8B")
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--fps",          type=int, default=4)
    parser.add_argument("--overwrite",    action="store_true", help="Overwrite existing captions")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    colormap = load_colormap(args.colormap)

    videos = sorted(Path(args.videos_dir).glob("*.mp4"))
    if not videos:
        print(f"No MP4 files found in {args.videos_dir}")
        sys.exit(1)

    print(f"Loading {args.model}...")
    import transformers
    model = transformers.AutoModelForVision2Seq.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        attn_implementation="sdpa",
    )
    processor = transformers.AutoProcessor.from_pretrained(args.model)
    model.eval()
    print("Model loaded.")

    for video_path in videos:
        stem = video_path.stem
        out_path = out_dir / f"{stem}.json"

        if out_path.exists() and not args.overwrite:
            print(f"SKIP (exists): {stem}")
            continue

        # Load metadata
        meta = load_metadata(args.metadata_dir, stem)

        # Extract objects from seg mask
        objects = []
        if args.seg_dir:
            seg_path = str(Path(args.seg_dir) / f"{stem}.mp4")
            if Path(seg_path).exists():
                objects = extract_objects_from_seg(seg_path, colormap)

        # Run Cosmos Reason
        print(f"Processing: {stem} | meta={bool(meta)} | objects={objects[:3]}...")
        try:
            description = run_cosmos_reason(
                str(video_path), meta, objects, model, processor, fps=args.fps
            )
        except Exception as e:
            print(f"  ERROR on {stem}: {e}")
            continue

        caption = assemble_caption(meta, objects, description)
        out_path.write_text(json.dumps(caption, indent=2, ensure_ascii=False))
        print(f"  → {word_count(caption['prompt'])} words | {out_path}")

    print("\nCaption generation complete.")


if __name__ == "__main__":
    main()
```

---

## Caption Quality Checks

After generation, inspect a sample:

```bash
# Show 5 random captions
for f in $(ls dataset_root/captions/*.json | shuf | head -5); do
  echo "=== $f ===" && python3 -c "import json; d=json.load(open('$f')); print(d['prompt'])" && echo
done

# Word count distribution
python3 -c "
import json, glob, statistics
wc = [len(json.load(open(f))['prompt'].split()) for f in glob.glob('dataset_root/captions/*.json')]
print(f'Count: {len(wc)}  Min: {min(wc)}  Max: {max(wc)}  Mean: {statistics.mean(wc):.0f}')
"
```

Expected: most captions 80–200 words. Anything > 290 was auto-truncated.

---

## Hardware & Time Estimates

| Model | GPU | Seconds per clip (93 frames) |
|---|---|---|
| Cosmos-Reason2-8B | H100 80GB | ~8–15 s |
| Cosmos-Reason2-8B | A100 80GB | ~12–20 s |
| Cosmos-Reason2-2B | RTX 4090 24GB | ~15–25 s |
| Cosmos-Reason1-7B | A100 40GB | ~10–18 s |

For 1000 clips on A100: ~3–5 hours.
Run overnight or on a brev GPU instance (see brev-cli skill).
