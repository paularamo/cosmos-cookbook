#!/usr/bin/env python3
"""
visualize_controls.py — Generate a self-contained HTML validation report
showing all control signal variants for an input image or video frame.

Also scans the dataset directory for pre-existing control files (edge, blur,
depth, seg, captions, metadata) and renders a presence/absence inventory table
at the top of the report.  When a prior depth or seg control file is found in
the dataset it is shown directly alongside (or instead of) the model output.

Panels generated:
  Inventory │ Table: which control signals already exist in the dataset
  Original  │ Source image / extracted video frame
  Edge      │ Canny (very-loose / loose / default / tight / very-tight)
             │  + "From dataset" preview if control_edge/<stem>.mp4 exists
  Blur      │ Gaussian blur (σ=3 / 5 / 7 / 10 / 15)
             │  + "From dataset" preview if control_blur/<stem>.mp4 exists
  Depth     │ DepthAnything V2 Large (vitl) colorized — OR existing dataset frame
  Seg       │ SAM 3 / SAM 2 automatic masks — OR existing dataset frame

Usage:
    # Basic — auto-infers dataset root if input is inside videos/ or images/
    python scripts/visualize_controls.py --input dataset_root/videos/clip.mp4

    # Explicit dataset root
    python scripts/visualize_controls.py --input clip.mp4 --dataset_root /data/myproject

    # Specific frame + depth model size
    python scripts/visualize_controls.py --input clip.mp4 --frame 30 --depth_model vitb

    # Fast local preview — skip GPU-heavy models
    python scripts/visualize_controls.py --input sample.jpg --no_depth --no_sam

Dependencies:
    pip install opencv-python numpy Pillow
    pip install torch torchvision transformers          # depth panel
    pip install git+https://github.com/facebookresearch/sam2.git  # seg panel
"""

import argparse
import base64
import sys
import traceback
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────────

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Dataset sub-directories that hold control signals and metadata
CONTROL_DIRS = {
    "edge":     "control_edge",
    "blur":     "control_blur",
    "depth":    "control_depth",
    "seg":      "control_seg",
}
AUX_DIRS = {
    "captions": "captions",
    "metadata": "metadata",
}
AUX_EXTS = {
    "captions": ".json",
    "metadata": ".json",
}

# ── Image I/O ─────────────────────────────────────────────────────────────────

def load_frame(path: str, frame_idx: int = 0) -> np.ndarray:
    """Load a still image or extract one frame from a video. Returns BGR array."""
    p = Path(path)
    if p.suffix.lower() in IMAGE_EXTS:
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Cannot read image: {path}")
        return img
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(frame_idx, total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise ValueError(f"Cannot read frame {frame_idx} from {path}")
    return frame


def img_to_base64(bgr: np.ndarray) -> str:
    """Encode BGR array as base64 PNG data URI for HTML embedding."""
    _, buf = cv2.imencode(".png", bgr)
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


# ── Dataset inventory scan ────────────────────────────────────────────────────

def infer_dataset_root(input_path: Path) -> Path | None:
    """
    If the input file lives inside a directory named 'videos' or 'images',
    treat its grandparent as the dataset root.  Otherwise return None.
    """
    parent_name = input_path.parent.name.lower()
    if parent_name in {"videos", "images"}:
        return input_path.parent.parent
    return None


def scan_dataset_controls(dataset_root: Path, stem: str, frame_idx: int) -> dict:
    """
    Check which control signal files already exist in dataset_root for the
    given clip stem.  Loads the relevant frame from each found file.

    Returns a dict keyed by signal type, each value a dict with:
        dir_path     : Path  — expected control directory
        dir_exists   : bool
        file_path    : Path  — expected control file
        file_exists  : bool
        size_mb      : float | None
        frame        : np.ndarray | None  — loaded BGR frame, or None
        n_frames     : int | None         — total frames in control video
    """
    results = {}

    for signal, subdir in CONTROL_DIRS.items():
        dir_path  = dataset_root / subdir
        file_path = dir_path / f"{stem}.mp4"
        dir_exists  = dir_path.is_dir()
        file_exists = file_path.is_file()
        size_mb     = file_path.stat().st_size / 1_048_576 if file_exists else None
        frame       = None
        n_frames    = None

        if file_exists:
            try:
                frame = load_frame(str(file_path), frame_idx)
                cap = cv2.VideoCapture(str(file_path))
                n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            except Exception as exc:
                print(f"  [scan] Warning: could not read {file_path}: {exc}")

        results[signal] = {
            "dir_path":   dir_path,
            "dir_exists": dir_exists,
            "file_path":  file_path,
            "file_exists": file_exists,
            "size_mb":    size_mb,
            "frame":      frame,
            "n_frames":   n_frames,
        }

    # Also check auxiliary dirs (captions, metadata) — no frame loading needed
    for key, subdir in AUX_DIRS.items():
        ext       = AUX_EXTS[key]
        dir_path  = dataset_root / subdir
        file_path = dir_path / f"{stem}{ext}"
        results[key] = {
            "dir_path":    dir_path,
            "dir_exists":  dir_path.is_dir(),
            "file_path":   file_path,
            "file_exists": file_path.is_file(),
            "size_mb":     file_path.stat().st_size / 1_048_576 if file_path.is_file() else None,
            "frame":       None,
            "n_frames":    None,
        }

    return results


# ── Canny edge variants ───────────────────────────────────────────────────────

CANNY_PRESETS = [
    ("Very Loose",  20,  50),
    ("Loose",       30,  80),
    ("Default",     50, 150),   # matches check_control.py default
    ("Tight",       80, 200),
    ("Very Tight", 100, 300),
]


def generate_canny_variants(bgr: np.ndarray) -> list:
    """Return [(label, low, high, BGR edge image), ...]."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    results = []
    for label, lo, hi in CANNY_PRESETS:
        edges = cv2.Canny(gray, lo, hi)
        results.append((label, lo, hi, cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)))
    return results


# ── Blur variants ─────────────────────────────────────────────────────────────

BLUR_PRESETS = [
    ("Light",    3),
    ("Medium",   5),
    ("Default",  7),   # matches check_control.py default
    ("Heavy",   10),
    ("Max",     15),
]


def generate_blur_variants(bgr: np.ndarray) -> list:
    """Return [(label, sigma, blurred_BGR), ...]."""
    results = []
    for label, sigma in BLUR_PRESETS:
        ksize = sigma * 6 + 1
        blurred = cv2.GaussianBlur(bgr, (ksize, ksize), sigma)
        results.append((label, sigma, blurred))
    return results


# ── Depth — DepthAnything V2 ──────────────────────────────────────────────────

_DEPTH_MODELS = {
    "vits": "depth-anything/Depth-Anything-V2-Small-hf",
    "vitb": "depth-anything/Depth-Anything-V2-Base-hf",
    "vitl": "depth-anything/Depth-Anything-V2-Large-hf",
}


def generate_depth_map(bgr: np.ndarray, model_size: str = "vitl") -> tuple | None:
    """
    Run DepthAnything V2 via HuggingFace transformers.
    Returns (hf_model_id, colorized_depth_bgr) or None on failure.
    Colormap INFERNO: dark = near, bright = far.
    """
    hf_id = _DEPTH_MODELS.get(model_size, _DEPTH_MODELS["vitl"])
    try:
        import torch
        from transformers import pipeline as hf_pipeline
        from PIL import Image as PILImage

        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"  [depth] Loading {hf_id} on {device} …", flush=True)
        estimator = hf_pipeline(
            task="depth-estimation",
            model=hf_id,
            device=0 if device == "cuda" else -1,
        )
        pil_rgb = PILImage.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        result   = estimator(pil_rgb)
        depth_np = np.array(result["depth"])
        depth_u8 = cv2.normalize(depth_np, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored  = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
        print("  [depth] Done.", flush=True)
        return hf_id, colored

    except ImportError as exc:
        print(f"  [depth] SKIP — missing dependency: {exc}")
        print("  Install: pip install transformers torch")
        return None
    except Exception as exc:
        print(f"  [depth] ERROR: {exc}")
        traceback.print_exc()
        return None


# ── Segmentation — SAM 3 / SAM 2 ─────────────────────────────────────────────

_MASK_COLORS_RGB = [
    (255,  56,  56), (255, 157,  56), (255, 255,  56), ( 56, 255, 101),
    ( 56, 182, 255), ( 56,  56, 255), (187,  56, 255), (255,  56, 182),
    (255, 255, 255), (128, 128,   0), (  0, 255, 128), (128,   0, 255),
]


def _colorize_masks(bgr: np.ndarray, masks: list) -> np.ndarray:
    overlay = bgr.copy().astype(np.float32)
    for i, m in enumerate(masks):
        seg = m["segmentation"]
        r, g, b = _MASK_COLORS_RGB[i % len(_MASK_COLORS_RGB)]
        patch = np.zeros_like(bgr, dtype=np.float32)
        patch[seg] = (b, g, r)
        overlay = cv2.addWeighted(overlay, 1.0, patch, 0.45, 0)
    result = np.clip(overlay, 0, 255).astype(np.uint8)
    for i, m in enumerate(masks):
        mask_u8 = m["segmentation"].astype(np.uint8) * 255
        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        r, g, b = _MASK_COLORS_RGB[i % len(_MASK_COLORS_RGB)]
        cv2.drawContours(result, contours, -1, (b, g, r), 1)
    return result


def generate_segmentation(bgr: np.ndarray) -> tuple | None:
    """
    Try SAM 3 → SAM 2.1 → SAM 2 in order.
    Returns (model_id, n_masks, colorized_bgr) or None on failure.
    """
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rgb    = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    candidates = [
        ("SAM 3",   "facebook/sam3-hiera-large"),
        ("SAM 2.1", "facebook/sam2.1-hiera-large"),
        ("SAM 2",   "facebook/sam2-hiera-large"),
    ]
    for version_name, model_id in candidates:
        try:
            print(f"  [seg] Trying {version_name} ({model_id}) …", flush=True)
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
            try:
                generator = SAM2AutomaticMaskGenerator.from_pretrained(
                    model_id, device=device,
                    points_per_side=32, pred_iou_thresh=0.80,
                    stability_score_thresh=0.90,
                )
            except TypeError:
                from sam2.build_sam import build_sam2
                sam_model = build_sam2(model_id, device=device)
                generator = SAM2AutomaticMaskGenerator(
                    sam_model, points_per_side=32,
                    pred_iou_thresh=0.80, stability_score_thresh=0.90,
                )
            print("  [seg] Running automatic mask generation …", flush=True)
            masks = generator.generate(rgb)
            if not masks:
                print("  [seg] No masks returned.")
                return None
            masks = sorted(masks, key=lambda m: m["area"], reverse=True)
            colored = _colorize_masks(bgr, masks)
            print(f"  [seg] {len(masks)} masks with {version_name}.")
            return model_id, len(masks), colored
        except ImportError:
            print(f"  [seg] {version_name} not available, trying next …")
            continue
        except Exception as exc:
            print(f"  [seg] {version_name} failed: {exc}")
            continue

    print("  [seg] SKIP — no SAM version available.")
    print("  Install: pip install git+https://github.com/facebookresearch/sam2.git")
    return None


# ── HTML helpers ──────────────────────────────────────────────────────────────

_CSS = """
:root {
  --bg:      #0f1117;
  --surface: #1a1d27;
  --border:  #2a2d3e;
  --accent:  #7c6af7;
  --text:    #e2e4f0;
  --muted:   #8890a0;
  --good:    #4ade80;
  --warn:    #f59e0b;
  --bad:     #f87171;
  --card-r:  8px;
  --gap:     14px;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  background: var(--bg); color: var(--text);
  font-family: 'Inter', 'Segoe UI', system-ui, sans-serif;
  font-size: 14px; line-height: 1.5; padding: 28px 32px;
}
code {
  font-family: 'Fira Code', 'Cascadia Code', monospace;
  background: rgba(255,255,255,0.07);
  padding: 1px 6px; border-radius: 3px; font-size: 0.9em;
}
a { color: var(--accent); text-decoration: none; }
h1 { font-size: 21px; font-weight: 700; }
h2 {
  font-size: 13px; font-weight: 700; letter-spacing: 0.08em;
  text-transform: uppercase; color: var(--accent);
  margin: 28px 0 10px; padding-bottom: 8px;
  border-bottom: 1px solid var(--border);
}
.subtitle { color: var(--muted); font-size: 13px; margin-top: 3px; }
.meta {
  display: flex; gap: 28px; flex-wrap: wrap;
  margin: 14px 0 0; font-size: 12px; color: var(--muted);
}
.meta span b { color: var(--accent); font-weight: 600; margin-right: 4px; }
.section-desc {
  font-size: 12px; color: var(--muted); margin-bottom: 12px; line-height: 1.6;
}

/* ── inventory table ── */
.inv-wrap {
  overflow-x: auto; margin-bottom: 4px;
}
.inv-table {
  width: 100%; border-collapse: collapse;
  font-size: 12px; background: var(--surface);
  border: 1px solid var(--border); border-radius: var(--card-r);
  overflow: hidden;
}
.inv-table th {
  background: rgba(124,106,247,0.12);
  color: var(--accent); font-weight: 700;
  padding: 9px 14px; text-align: left;
  border-bottom: 1px solid var(--border);
  white-space: nowrap;
}
.inv-table td {
  padding: 8px 14px; vertical-align: middle;
  border-bottom: 1px solid var(--border);
  color: var(--text);
}
.inv-table tr:last-child td { border-bottom: none; }
.inv-table tr:hover td { background: rgba(255,255,255,0.03); }
.inv-table td.mono { font-family: 'Fira Code','Cascadia Code',monospace; font-size: 11px; color: var(--muted); }
.inv-table img.thumb {
  width: 80px; height: 45px;
  object-fit: cover; border-radius: 4px;
  border: 1px solid var(--border); display: block;
}
.inv-table .no-thumb {
  width: 80px; height: 45px; border-radius: 4px;
  border: 1px dashed var(--border);
  display: flex; align-items: center; justify-content: center;
  font-size: 10px; color: var(--muted);
}

/* ── badges ── */
.badge {
  display: inline-block; padding: 2px 8px; border-radius: 10px;
  font-size: 11px; font-weight: 700; white-space: nowrap;
}
.badge.present  { background: rgba(74,222,128,0.12);  color: var(--good); border: 1px solid rgba(74,222,128,0.3); }
.badge.missing  { background: rgba(248,113,113,0.10); color: var(--bad);  border: 1px solid rgba(248,113,113,0.3); }
.badge.default  { background: rgba(124,106,247,0.12); color: var(--accent); border: 1px solid rgba(124,106,247,0.3); }
.badge.dataset  { background: rgba(78,205,196,0.12);  color: #4ecdc4; border: 1px solid rgba(78,205,196,0.3); }
.badge.nodir    { background: rgba(255,255,255,0.04); color: var(--muted); border: 1px solid var(--border); }

/* ── card grid ── */
.grid { display: flex; flex-wrap: wrap; gap: var(--gap); margin-bottom: 4px; }
.card {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--card-r); overflow: hidden;
  flex: 1 1 160px; max-width: 320px;
  display: flex; flex-direction: column;
}
.card.lg { flex: 1 1 320px; max-width: 600px; }
.card img { width: 100%; display: block; }
.card-foot {
  padding: 8px 12px; font-size: 11px;
  border-top: 1px solid var(--border);
}
.card-title { font-weight: 700; color: var(--text); margin-bottom: 2px; }
.card-sub   { color: var(--muted); line-height: 1.4; }

.unavail {
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--card-r); padding: 20px 22px; font-size: 12px;
  flex: 1 1 auto;
}
.unavail .title  { font-weight: 700; color: var(--warn); margin-bottom: 6px; }
.unavail .reason { color: var(--muted); margin-bottom: 10px; }
.unavail code {
  display: block; padding: 6px 10px; margin-top: 4px;
  background: rgba(245,158,11,0.08); border: 1px solid rgba(245,158,11,0.2);
  color: var(--warn); border-radius: 4px; font-size: 12px;
}

footer {
  margin-top: 40px; padding-top: 14px;
  border-top: 1px solid var(--border);
  font-size: 11px; color: var(--muted);
  display: flex; justify-content: space-between;
}
"""


def _card(uri: str, title: str, sub: str = "", large: bool = False) -> str:
    cls     = "card lg" if large else "card"
    sub_htm = f'<div class="card-sub">{sub}</div>' if sub else ""
    return (
        f'<div class="{cls}">'
        f'<img src="{uri}" alt="{title}">'
        f'<div class="card-foot">'
        f'<div class="card-title">{title}</div>{sub_htm}'
        f'</div></div>'
    )


def _unavail(title: str, reason: str, cmd: str) -> str:
    return (
        f'<div class="unavail">'
        f'<div class="title">{title}</div>'
        f'<div class="reason">{reason}</div>'
        f'<code>{cmd}</code>'
        f'</div>'
    )


# ── Inventory table rendering ─────────────────────────────────────────────────

_SIGNAL_META = {
    "edge":     ("Edge",        "control_edge/",     "Canny edges",                  "auto-generated"),
    "blur":     ("Blur",        "control_blur/",     "Gaussian blur",                "auto-generated"),
    "depth":    ("Depth",       "control_depth/",    "Depth map",                    "external tool"),
    "seg":      ("Seg",         "control_seg/",      "Segmentation mask",            "external tool"),
    "captions": ("Captions",    "captions/",         "Caption JSON",                 "generate_captions.py"),
    "metadata": ("Metadata",    "metadata/",         "Metadata JSON",                "user-supplied"),
}

_CONTROL_SIGNALS = ["edge", "blur", "depth", "seg"]
_ALL_KEYS        = ["edge", "blur", "depth", "seg", "captions", "metadata"]


def _build_inventory_table(dataset_controls: dict | None, stem: str) -> str:
    if dataset_controls is None:
        return """
<h2>Dataset Control Signal Inventory</h2>
<p class="section-desc" style="color:var(--warn)">
  No dataset root detected. Pass <code>--dataset_root /path/to/dataset</code>
  or place the input file inside a <code>videos/</code> subdirectory for
  automatic discovery.
</p>
"""

    # Count present control signals for the summary line
    n_present = sum(
        1 for k in _CONTROL_SIGNALS
        if dataset_controls[k]["file_exists"]
    )

    rows = ""
    for key in _ALL_KEYS:
        info           = dataset_controls[key]
        label, subdir, desc, how = _SIGNAL_META[key]
        dir_badge      = (
            '<span class="badge present">exists</span>'
            if info["dir_exists"]
            else '<span class="badge nodir">absent</span>'
        )
        if info["file_exists"]:
            file_badge = '<span class="badge present">present</span>'
            size_str   = f'{info["size_mb"]:.1f} MB' if info["size_mb"] is not None else "—"
            frames_str = str(info["n_frames"]) if info["n_frames"] is not None else "—"
            if key in _CONTROL_SIGNALS:
                extra = f'<span style="color:var(--muted);font-size:10px">{frames_str} frames</span>'
            else:
                extra = ""
        else:
            file_badge = '<span class="badge missing">missing</span>'
            size_str   = "—"
            extra      = ""

        # Thumbnail
        if info.get("frame") is not None:
            thumb_uri = img_to_base64(info["frame"])
            thumb_htm = f'<img class="thumb" src="{thumb_uri}" alt="{label} preview">'
        elif info["file_exists"] and key not in _CONTROL_SIGNALS:
            # JSON files — show a small text indicator
            thumb_htm = '<div class="no-thumb">JSON</div>'
        else:
            thumb_htm = '<div class="no-thumb">—</div>'

        rows += f"""
    <tr>
      <td><strong>{label}</strong><br><span style="color:var(--muted);font-size:10px">{desc}</span></td>
      <td class="mono">{subdir}<br>{dir_badge}</td>
      <td class="mono">{stem}{'  .mp4' if key in _CONTROL_SIGNALS else '.json'}<br>{file_badge}</td>
      <td>{size_str}<br><span style="color:var(--muted);font-size:10px">{extra}</span></td>
      <td><span style="color:var(--muted);font-size:10px">{how}</span></td>
      <td>{thumb_htm}</td>
    </tr>"""

    summary_color = "var(--good)" if n_present == 4 else ("var(--warn)" if n_present > 0 else "var(--bad)")
    summary = (
        f'<b style="color:{summary_color}">{n_present} / 4</b> control signal types present for this clip.'
    )

    return f"""
<h2>Dataset Control Signal Inventory</h2>
<p class="section-desc">
  Scanned dataset root: <code>{dataset_controls["edge"]["dir_path"].parent}</code>
  &nbsp;·&nbsp; Clip stem: <code>{stem}</code>
  &nbsp;·&nbsp; {summary}
</p>
<div class="inv-wrap">
<table class="inv-table">
  <thead>
    <tr>
      <th>Signal</th>
      <th>Directory</th>
      <th>File (this clip)</th>
      <th>Size</th>
      <th>How generated</th>
      <th>Preview (frame)</th>
    </tr>
  </thead>
  <tbody>{rows}
  </tbody>
</table>
</div>
"""


# ── Full HTML assembly ─────────────────────────────────────────────────────────

def build_html(
    filename: str,
    stem: str,
    is_video: bool,
    resolution: str,
    frame_info: str,
    original_bgr: np.ndarray,
    canny_variants: list,
    blur_variants: list,
    depth_result,
    seg_result,
    dataset_controls: dict | None,
) -> str:
    ts        = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    orig_uri  = img_to_base64(original_bgr)
    itype     = "Video" if is_video else "Image"

    # ── header ────────────────────────────────────────────────────────────────
    header = f"""
<h1>Control Signal Validation Report</h1>
<p class="subtitle">Cosmos Transfer 2.5 — Stage 3: Control Signal Inspection</p>
<div class="meta">
  <span><b>Input</b>{filename}</span>
  <span><b>Type</b>{itype}</span>
  <span><b>Resolution</b>{resolution}</span>
  <span><b>Frame</b>{frame_info}</span>
  <span><b>Generated</b>{ts}</span>
</div>
"""

    # ── inventory table ───────────────────────────────────────────────────────
    inventory_html = _build_inventory_table(dataset_controls, stem)

    # ── original ──────────────────────────────────────────────────────────────
    original_html = f"""
<h2>Original</h2>
<div class="grid">{_card(orig_uri, filename, f'{resolution} · {itype}', large=True)}</div>
"""

    # ── canny ─────────────────────────────────────────────────────────────────
    # Dataset frame for edge (if present) shown first
    ds_edge = dataset_controls["edge"] if dataset_controls else None
    ds_edge_card = ""
    if ds_edge and ds_edge["file_exists"] and ds_edge["frame"] is not None:
        ds_edge_card = _card(
            img_to_base64(ds_edge["frame"]),
            'From Dataset &nbsp;<span class="badge dataset">dataset</span>',
            f'control_edge/{stem}.mp4 · {ds_edge["size_mb"]:.1f} MB · {ds_edge["n_frames"]} frames',
            large=True,
        )

    canny_cards = ds_edge_card
    for label, lo, hi, img in canny_variants:
        is_default = label == "Default"
        badge      = ' &nbsp;<span class="badge default">check_control default</span>' if is_default else ""
        canny_cards += _card(
            img_to_base64(img),
            label + badge,
            f"cv2.Canny(gray, low={lo}, high={hi})",
        )

    canny_html = f"""
<h2>Canny Edge Detection — Threshold Variations</h2>
<p class="section-desc">
  Applied frame-by-frame as per <code>check_control.py generate_edge_control()</code>.
  Default: <code>low=50, high=150</code>.
  Lower thresholds detect more (noisier) edges; higher values keep only strong contours.
  Saved to <code>control_edge/</code> as grayscale MP4.
</p>
<div class="grid">{canny_cards}</div>
"""

    # ── blur ──────────────────────────────────────────────────────────────────
    ds_blur = dataset_controls["blur"] if dataset_controls else None
    ds_blur_card = ""
    if ds_blur and ds_blur["file_exists"] and ds_blur["frame"] is not None:
        ds_blur_card = _card(
            img_to_base64(ds_blur["frame"]),
            'From Dataset &nbsp;<span class="badge dataset">dataset</span>',
            f'control_blur/{stem}.mp4 · {ds_blur["size_mb"]:.1f} MB · {ds_blur["n_frames"]} frames',
            large=True,
        )

    blur_cards = ds_blur_card
    for label, sigma, img in blur_variants:
        ksize      = sigma * 6 + 1
        is_default = label == "Default"
        badge      = ' &nbsp;<span class="badge default">check_control default</span>' if is_default else ""
        blur_cards += _card(
            img_to_base64(img),
            label + badge,
            f"GaussianBlur(σ={sigma}, kernel {ksize}×{ksize})",
        )

    blur_html = f"""
<h2>Gaussian Blur — Sigma Variations</h2>
<p class="section-desc">
  Applied frame-by-frame as per <code>check_control.py generate_blur_control()</code>.
  Default: <code>sigma=7</code> (kernel 43×43).
  Used as a <em>vis</em> control signal — keep weight ≤ 0.6 to avoid temporal noise.
</p>
<div class="grid">{blur_cards}</div>
"""

    # ── depth ─────────────────────────────────────────────────────────────────
    ds_depth = dataset_controls["depth"] if dataset_controls else None

    depth_cards = ""
    depth_note  = ""

    # Prior dataset frame
    if ds_depth and ds_depth["file_exists"] and ds_depth["frame"] is not None:
        depth_cards += _card(
            img_to_base64(ds_depth["frame"]),
            'From Dataset &nbsp;<span class="badge dataset">dataset</span>',
            f'control_depth/{stem}.mp4 · {ds_depth["size_mb"]:.1f} MB · {ds_depth["n_frames"]} frames',
            large=True,
        )

    # Model-computed frame
    if depth_result is not None:
        hf_id, depth_bgr = depth_result
        short_id = hf_id.split("/")[-1]
        depth_cards += _card(
            img_to_base64(depth_bgr),
            f'Model Output &nbsp;<span class="badge default">DepthAnything V2</span>',
            f'{short_id} · INFERNO colormap · dark=near bright=far',
            large=True,
        )
    elif not (ds_depth and ds_depth["file_exists"]):
        depth_cards += _unavail(
            "Depth Map Unavailable",
            "No pre-existing control_depth/ file found and DepthAnything V2 not run.",
            "pip install transformers torch  # then re-run without --no_depth",
        )

    depth_html = f"""
<h2>Depth Estimation — DepthAnything V2</h2>
<p class="section-desc">
  Colorized with <code>COLORMAP_INFERNO</code>: dark (near) → bright (far).
  For training, save the <em>grayscale</em> version to <code>control_depth/</code>
  (0 = near, 255 = far).{depth_note}
</p>
<div class="grid">
  {_card(orig_uri, "Original", "Reference", large=True)}
  {depth_cards}
</div>
"""

    # ── segmentation ──────────────────────────────────────────────────────────
    ds_seg = dataset_controls["seg"] if dataset_controls else None

    seg_cards = ""

    if ds_seg and ds_seg["file_exists"] and ds_seg["frame"] is not None:
        seg_cards += _card(
            img_to_base64(ds_seg["frame"]),
            'From Dataset &nbsp;<span class="badge dataset">dataset</span>',
            f'control_seg/{stem}.mp4 · {ds_seg["size_mb"]:.1f} MB · {ds_seg["n_frames"]} frames',
            large=True,
        )

    if seg_result is not None:
        model_id, n_masks, seg_bgr = seg_result
        short_id = model_id.split("/")[-1]
        seg_cards += _card(
            img_to_base64(seg_bgr),
            f'Model Output &nbsp;<span class="badge default">SAM</span>',
            f'{short_id} · {n_masks} masks · contours shown',
            large=True,
        )
    elif not (ds_seg and ds_seg["file_exists"]):
        seg_cards += _unavail(
            "Segmentation Unavailable",
            "No pre-existing control_seg/ file found and SAM not run.",
            "pip install git+https://github.com/facebookresearch/sam2.git",
        )

    seg_html = f"""
<h2>Segmentation — SAM 3 / SAM 2 (Automatic Mask Generation)</h2>
<p class="section-desc">
  Each colored region is a distinct segment.  For Cosmos Transfer, convert to
  COCO-palette color-coded masks in <code>control_seg/</code> for object
  extraction in caption generation.
</p>
<div class="grid">
  {_card(orig_uri, "Original", "Reference", large=True)}
  {seg_cards}
</div>
"""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Control Report — {filename}</title>
<style>{_CSS}</style>
</head>
<body>
{header}
{inventory_html}
{original_html}
{canny_html}
{blur_html}
{depth_html}
{seg_html}
<footer>
  <span>Cosmos Transfer 2.5 — Control Signal Validator</span>
  <span>Stage 3 · <a href="../references/data-validation.md">data-validation.md</a></span>
</footer>
</body>
</html>"""


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate HTML control signal validation report for Cosmos Transfer 2.5."
    )
    parser.add_argument("--input",        required=True,
                        help="Input image (jpg/png) or video (mp4)")
    parser.add_argument("--dataset_root", default=None,
                        help="Dataset root directory (auto-inferred if input is inside videos/)")
    parser.add_argument("--output",       default=None,
                        help="Output HTML path (default: <stem>_control_report.html)")
    parser.add_argument("--frame",        type=int, default=0,
                        help="Frame index to extract from video (default: 0)")
    parser.add_argument("--depth_model",  default="vitl",
                        choices=["vits", "vitb", "vitl"],
                        help="DepthAnything V2 model size (default: vitl)")
    parser.add_argument("--no_depth",     action="store_true",
                        help="Skip DepthAnything V2 inference")
    parser.add_argument("--no_sam",       action="store_true",
                        help="Skip SAM segmentation inference")
    args = parser.parse_args()

    inp = Path(args.input)
    if not inp.exists():
        print(f"ERROR: not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    stem      = inp.stem
    out_path  = args.output or f"{stem}_control_report.html"
    is_video  = inp.suffix.lower() not in IMAGE_EXTS

    print(f"Input : {inp.name}  ({'video' if is_video else 'image'})")
    print(f"Output: {out_path}")

    # ── Dataset root resolution ────────────────────────────────────────────
    if args.dataset_root:
        dataset_root = Path(args.dataset_root)
        if not dataset_root.is_dir():
            print(f"WARNING: --dataset_root not found: {dataset_root}", file=sys.stderr)
            dataset_root = None
    else:
        dataset_root = infer_dataset_root(inp)
        if dataset_root:
            print(f"Dataset: {dataset_root}  (auto-inferred from input path)")
        else:
            print("Dataset: not detected — pass --dataset_root to enable inventory scan")

    # ── Dataset inventory ──────────────────────────────────────────────────
    dataset_controls = None
    if dataset_root:
        print(f"\n[0/4] Scanning dataset controls for clip '{stem}' …")
        dataset_controls = scan_dataset_controls(dataset_root, stem, args.frame)
        for signal in _CONTROL_SIGNALS:
            info = dataset_controls[signal]
            status = "FOUND" if info["file_exists"] else ("dir missing" if not info["dir_exists"] else "file missing")
            print(f"  {signal:6s}: {status}" + (f"  ({info['size_mb']:.1f} MB, {info['n_frames']} frames)" if info["file_exists"] else ""))

    # ── Load source frame ──────────────────────────────────────────────────
    bgr        = load_frame(str(inp), args.frame)
    h, w       = bgr.shape[:2]
    resolution = f"{w} × {h} px"
    frame_info = f"Frame {args.frame}" if is_video else "N/A (static image)"
    print(f"\nFrame : {frame_info}  {resolution}")

    # ── Compute control variants ───────────────────────────────────────────
    print("\n[1/4] Canny edge variants …")
    canny_variants = generate_canny_variants(bgr)

    print("[2/4] Gaussian blur variants …")
    blur_variants = generate_blur_variants(bgr)

    depth_result = None
    if not args.no_depth:
        print(f"[3/4] DepthAnything V2 ({args.depth_model}) …")
        depth_result = generate_depth_map(bgr, args.depth_model)
    else:
        print("[3/4] Depth — skipped (--no_depth)")

    seg_result = None
    if not args.no_sam:
        print("[4/4] SAM segmentation …")
        seg_result = generate_segmentation(bgr)
    else:
        print("[4/4] SAM — skipped (--no_sam)")

    # ── Build & write HTML ─────────────────────────────────────────────────
    print("\nBuilding HTML report …")
    html = build_html(
        filename=inp.name,
        stem=stem,
        is_video=is_video,
        resolution=resolution,
        frame_info=frame_info,
        original_bgr=bgr,
        canny_variants=canny_variants,
        blur_variants=blur_variants,
        depth_result=depth_result,
        seg_result=seg_result,
        dataset_controls=dataset_controls,
    )
    Path(out_path).write_text(html, encoding="utf-8")
    print(f"\nDone → {out_path}")


if __name__ == "__main__":
    main()
