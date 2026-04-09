#!/usr/bin/env python3
"""
check_control.py — Check if control signal videos exist for every source clip.
Auto-generates edge or blur control signals if missing. Warns for depth/seg.

Usage:
    python scripts/check_control.py \
        --videos_dir   dataset_root/videos/ \
        --control_dir  dataset_root/control_edge/ \
        --control_type edge \
        --auto_generate

    # Check depth (warns but cannot auto-generate)
    python scripts/check_control.py \
        --videos_dir  dataset_root/videos/ \
        --control_dir dataset_root/control_depth/ \
        --control_type depth

    # Also validate alignment (dimensions + frame count)
    python scripts/check_control.py \
        --videos_dir   dataset_root/videos/ \
        --control_dir  dataset_root/control_edge/ \
        --control_type edge \
        --validate_alignment
"""

import argparse
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np


# ── Video probing ─────────────────────────────────────────────────────────────

def probe_video(path: str) -> dict:
    """Return dict with width, height, fps_str, nb_frames (or None on failure)."""
    try:
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return {}
        info = {
            "width":  int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps":    cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }
        cap.release()
        return info
    except Exception:
        return {}


# ── Edge control generation ───────────────────────────────────────────────────

def generate_edge_control(src_path: str, dst_path: str,
                           low_thresh: int = 50, high_thresh: int = 150) -> bool:
    """
    Apply Canny edge detection to every frame of src_path, write grayscale
    edge-map video to dst_path. Returns True on success.
    """
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {src_path}")
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, low_thresh, high_thresh)
        # Convert grayscale edges back to BGR for VideoWriter
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        out.write(edges_bgr)

    cap.release()
    out.release()

    # Re-encode with libx264 for compatibility (mp4v → H.264)
    tmp = dst_path + ".tmp.mp4"
    try:
        import subprocess
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", dst_path,
             "-c:v", "libx264", "-crf", "18", "-preset", "fast", tmp],
            capture_output=True
        )
        if result.returncode == 0:
            Path(tmp).replace(dst_path)
    except FileNotFoundError:
        pass  # ffmpeg not available; keep mp4v encoding

    return Path(dst_path).exists()


# ── Blur control generation ───────────────────────────────────────────────────

def generate_blur_control(src_path: str, dst_path: str, sigma: int = 7) -> bool:
    """Gaussian blur every frame and write to dst_path."""
    cap = cv2.VideoCapture(src_path)
    if not cap.isOpened():
        return False

    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(dst_path, fourcc, fps, (width, height))

    ksize = sigma * 6 + 1  # kernel size must be odd
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        blurred = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        out.write(blurred)

    cap.release()
    out.release()

    tmp = dst_path + ".tmp.mp4"
    try:
        result = subprocess.run(
            ["ffmpeg", "-y", "-i", dst_path,
             "-c:v", "libx264", "-crf", "18", "-preset", "fast", tmp],
            capture_output=True
        )
        if result.returncode == 0:
            Path(tmp).replace(dst_path)
    except FileNotFoundError:
        pass

    return Path(dst_path).exists()


# ── Alignment check ───────────────────────────────────────────────────────────

def check_alignment(src_info: dict, ctrl_info: dict, name: str) -> list[str]:
    issues = []
    for field in ("width", "height", "frames"):
        sv, cv_ = src_info.get(field), ctrl_info.get(field)
        if sv != cv_:
            issues.append(f"{field}: source={sv} vs control={cv_}")
    # FPS tolerance 0.1
    sf, cf = src_info.get("fps", 0), ctrl_info.get("fps", 0)
    if abs(sf - cf) > 0.1:
        issues.append(f"fps: source={sf:.2f} vs control={cf:.2f}")
    return issues


# ── Main ──────────────────────────────────────────────────────────────────────

AUTO_GENERATE_TYPES = {"edge", "blur"}
MANUAL_ONLY_TYPES   = {"depth", "seg"}

MANUAL_HINTS = {
    "depth": (
        "Depth maps cannot be auto-generated. Options:\n"
        "  • Depth Anything V2: https://github.com/DepthAnything/Depth-Anything-V2\n"
        "  • ZoeDepth: https://github.com/isl-org/ZoeDepth\n"
        "  Run depth estimation on each video and save as matching MP4."
    ),
    "seg": (
        "Segmentation masks cannot be auto-generated. Options:\n"
        "  • Grounded-SAM: https://github.com/IDEA-Research/Grounded-Segment-Anything\n"
        "  • Mask2Former: https://github.com/facebookresearch/Mask2Former\n"
        "  Run segmentation on each video and save color-coded masks as matching MP4."
    ),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--videos_dir",        required=True)
    parser.add_argument("--control_dir",       required=True)
    parser.add_argument("--control_type",      required=True,
                        choices=["edge", "depth", "seg", "blur"])
    parser.add_argument("--auto_generate",     action="store_true",
                        help="Auto-generate missing edge/blur control videos")
    parser.add_argument("--validate_alignment", action="store_true",
                        help="Check dimensions and frame counts match source")
    parser.add_argument("--edge_low",  type=int, default=50)
    parser.add_argument("--edge_high", type=int, default=150)
    parser.add_argument("--blur_sigma", type=int, default=7)
    args = parser.parse_args()

    videos_dir  = Path(args.videos_dir)
    control_dir = Path(args.control_dir)
    control_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(videos_dir.glob("*.mp4"))
    if not videos:
        print(f"No MP4 files found in {args.videos_dir}")
        sys.exit(1)

    can_auto = args.control_type in AUTO_GENERATE_TYPES

    if args.control_type in MANUAL_ONLY_TYPES:
        print(f"\nNOTE: control_type='{args.control_type}' requires external tools.")
        print(MANUAL_HINTS[args.control_type])
        print()

    missing, generated, ok, alignment_errors = 0, 0, 0, 0

    for video_path in videos:
        stem = video_path.stem
        ctrl_path = control_dir / f"{stem}.mp4"

        if not ctrl_path.exists():
            missing += 1
            if can_auto and args.auto_generate:
                print(f"GENERATE: {stem} → {ctrl_path}")
                if args.control_type == "edge":
                    success = generate_edge_control(
                        str(video_path), str(ctrl_path),
                        args.edge_low, args.edge_high
                    )
                else:  # blur
                    success = generate_blur_control(
                        str(video_path), str(ctrl_path), args.blur_sigma
                    )
                if success:
                    generated += 1
                    print(f"  OK: generated {ctrl_path}")
                else:
                    print(f"  ERROR: failed to generate {ctrl_path}")
            else:
                tag = "MISSING" if not can_auto else "MISSING (run with --auto_generate)"
                print(f"{tag}: {ctrl_path}")
        else:
            ok += 1
            if args.validate_alignment:
                src_info  = probe_video(str(video_path))
                ctrl_info = probe_video(str(ctrl_path))
                issues = check_alignment(src_info, ctrl_info, stem)
                if issues:
                    alignment_errors += 1
                    print(f"ALIGNMENT ERROR: {stem}")
                    for issue in issues:
                        print(f"  • {issue}")

    print()
    print(f"Results for control_type='{args.control_type}':")
    print(f"  Found:     {ok}")
    print(f"  Missing:   {missing - generated}")
    print(f"  Generated: {generated}")
    if args.validate_alignment:
        print(f"  Alignment errors: {alignment_errors}")

    total_issues = (missing - generated) + alignment_errors
    if total_issues > 0:
        sys.exit(1)
    else:
        print("All control videos OK.")


if __name__ == "__main__":
    main()
