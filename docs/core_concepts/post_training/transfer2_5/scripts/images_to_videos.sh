#!/usr/bin/env bash
# images_to_videos.sh — Convert still images to 93-frame 16 FPS MP4 clips.
#
# ⚠️  DEPRECATED — NOT RECOMMENDED FOR COSMOS TRANSFER TRAINING
# Cosmos Transfer 2.5 is a video generation and temporal reasoning model.
# Still images looped as video carry no motion or inter-frame information,
# which degrades training quality. Use real video clips instead.
# This script is retained for pipeline testing purposes only.
#
# Usage:
#   bash images_to_videos.sh <input_dir> <output_dir> [frames] [fps] [width] [height]
#
# Defaults: 93 frames, 16 FPS, 1280x720
# Supports: jpg, jpeg, png (case-insensitive)
#
# Example:
#   bash images_to_videos.sh images/ videos/
#   bash images_to_videos.sh images/ videos/ 186 16 1280 720

set -euo pipefail

INPUT_DIR="${1:?Usage: $0 <input_dir> <output_dir> [frames=93] [fps=16] [width=1280] [height=720]}"
OUTPUT_DIR="${2:?Usage: $0 <input_dir> <output_dir>}"
FRAMES="${3:-93}"
FPS="${4:-16}"
WIDTH="${5:-1280}"
HEIGHT="${6:-720}"

mkdir -p "$OUTPUT_DIR"

CONVERTED=0
SKIPPED=0
ERRORS=0

for img in "$INPUT_DIR"/*.{jpg,jpeg,png,JPG,JPEG,PNG}; do
    [[ -f "$img" ]] || continue
    stem=$(basename "$img" | sed 's/\.[^.]*$//')
    out="$OUTPUT_DIR/${stem}.mp4"

    if [[ -f "$out" ]]; then
        echo "SKIP (exists): $out"
        SKIPPED=$((SKIPPED+1))
        continue
    fi

    ffmpeg -y \
        -loop 1 \
        -i "$img" \
        -vf "fps=${FPS},scale=${WIDTH}:${HEIGHT}:flags=lanczos,format=yuv420p" \
        -frames:v "$FRAMES" \
        -c:v libx264 \
        -crf 18 \
        -preset fast \
        -an \
        "$out" 2>/dev/null && {
            echo "OK: $img → $out  (${FRAMES} frames @ ${FPS}fps, ${WIDTH}x${HEIGHT})"
            CONVERTED=$((CONVERTED+1))
        } || {
            echo "ERROR: $img"
            ERRORS=$((ERRORS+1))
        }
done

echo ""
echo "Done. Converted: $CONVERTED | Skipped: $SKIPPED | Errors: $ERRORS"
echo "Output: $OUTPUT_DIR"

# Validate frame counts
if [[ $CONVERTED -gt 0 ]]; then
    echo ""
    echo "Validating frame counts..."
    BAD=0
    for f in "$OUTPUT_DIR"/*.mp4; do
        fc=$(ffprobe -v error -select_streams v:0 -count_frames \
            -show_entries stream=nb_read_frames -of csv=p=0 "$f" 2>/dev/null)
        mod=$((fc % 93))
        if [[ "$mod" -ne 0 ]]; then
            echo "  WARN: $f has $fc frames (not ×93)"
            BAD=$((BAD+1))
        fi
    done
    [[ $BAD -eq 0 ]] && echo "  All frame counts OK."
fi
