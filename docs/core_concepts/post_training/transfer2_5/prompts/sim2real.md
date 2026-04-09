# Sim2Real Workflow — Cosmos Transfer 2.5

Post-training Cosmos Transfer 2.5 to convert synthetic (CARLA, Isaac, Omniverse) video to photorealistic output.

---

## Goal

Adapt the base Cosmos Transfer 2.5 model to your specific synthetic dataset so that edge/seg-controlled transfer produces your target real-world visual style.

---

## Recommended Control Signal

- **Primary:** `edge` (weight 0.6) — preserves scene structure across the synthetic/real domain gap
- **Secondary:** `seg` (weight 0.4) — maintains semantic layout (road, sky, vehicles)

---

## Step-by-Step

### 1. Prepare Paired Data

Collect paired or domain-matched clips:
- **Synthetic source:** CARLA/Isaac renders at exactly 16 FPS, 720P
- **Real target:** dashcam/street footage at 16 FPS, 720P
- Clips need NOT be pixel-paired; domain-level correspondence is sufficient

Minimum viable dataset: **50 synthetic clips** + real clips used as style reference in captions

### 2. Generate Edge Control from Synthetic

```bash
# Auto-generated from synthetic clips during inference
# Or pre-generate with Canny:
python tools/generate_control.py \
  --type edge \
  --input_dir dataset_root/synthetic_videos/ \
  --output_dir dataset_root/control_edge/
```

### 3. Write Captions

Each caption should describe the real-world equivalent, not the synthetic render:

```json
{
  "prompt": "Photorealistic urban street scene. Daytime, overcast sky. Wet asphalt road with lane markings. Modern vehicles parked along the curb. Wide-angle dashcam perspective. Natural lighting with soft shadows.",
  "negative_prompt": "synthetic, cartoon, CG render, unrealistic colors, flat lighting"
}
```

### 4. Config

```python
# experiments/sim2real_carla.py
data = dict(
    train_dataset=dict(
        data_path="/path/to/dataset_root",
        control_types=["edge", "seg"],
        control_weights=[0.6, 0.4],
        control_dirs=["control_edge", "control_seg"],
        caption_dir="captions",
        video_dir="synthetic_videos",
        resolution=(720, 1280),
        num_frames=93,
        fps=16,
    ),
    batch_size=1,
    num_workers=4,
)
training = dict(
    max_iter=1000,
    save_iter=200,
    learning_rate=1e-5,
)
```

### 5. Train

```bash
export EXP=sim2real_carla
IMAGINAIRE_OUTPUT_ROOT=/large/storage \
torchrun --nproc_per_node=8 --master_port=12341 \
  -m scripts.train \
  --config=cosmos_transfer2/_src/transfer2/configs/vid2vid_transfer/config.py \
  -- experiment=${EXP} job.wandb_mode=disabled
```

### 6. Validate at 25%, 50%, 75%, 100%

Convert checkpoint and run on a held-out synthetic clip:
```bash
python scripts/convert_distcp_to_pt.py ${CKPT}/model ${CKPT}
python examples/inference.py \
  --params_file test_spec.json \
  --checkpoint_path ${CKPT}/checkpoint_ema_bfloat16.pt \
  -o outputs/validation/
```

Compare: Does the output look like your target real-world domain? Is structure preserved?

---

## Expected Results

- Loss should drop within 100–200 iterations
- Visual output starts showing real-world textures by 50% training
- Overfitting risk: monitor held-out clips; stop if identical-to-input
