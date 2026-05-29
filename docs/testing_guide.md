# Testing the VLM Exploration Pipeline

## Sample Scene

The test image (`sample_scene.png`) shows three objects on a white table — a perfect controlled scene for the pipeline:

![Sample scene for manipulation testing](/home/twinxblaze/.gemini/antigravity-cli/brain/289930a3-e94b-40e0-bf8d-fb0f4d71b4f3/sample_scene_1779541061655.png)

| Object | Position | Relevance |
|---|---|---|
| Water bottle | Left | Tall cylindrical — good bimanual grasp target |
| Red mug | Right | Short, wide — handle visible, orientation matters |
| Wooden block | Back center | Rigid cube — good for rotation tasks |

---

## Step 0 — Activate environment

Always activate your virtual environment first:

```bash
cd ~/fyp
source vla_env/bin/activate
```

---

## Test 1 — Full pipeline (Florence-2 + SmolVLM2)

```bash
python run_pipeline.py \
  --image sample_scene.png \
  --task "Pick up the water bottle with both arms and rotate it 90 degrees clockwise" \
  --ground-phrase "the water bottle" \
  --save output/test1_result.jpg
```

### What to expect

```
╔══════════════════════════════════════════════════╗
║  XLeRobot VLM Exploration Pipeline               ║
╚══════════════════════════════════════════════════╝

╔══════════════════════════════════════════════════╗
║  System Info                                     ║
╚══════════════════════════════════════════════════╝
  GPU : NVIDIA GeForce RTX 5060 Laptop GPU
  VRAM: 8.5 GB free / 8.5 GB total

╔══════════════════════════════════════════════════╗
║  Florence-2 Perception  [large]                  ║
╚══════════════════════════════════════════════════╝
[Florence-2] Loading microsoft/Florence-2-large on cuda...
[Florence-2] Ready.

  [1/4] Object Detection
  Detected 3 object(s):
    • bottle                 bbox: [85.0, 120.0, 340.0, 950.0]
    • cup                    bbox: [520.0, 380.0, 850.0, 900.0]
    • wood block             bbox: [380.0, 50.0, 620.0, 320.0]

  [2/4] Scene Caption
  → A clear plastic water bottle on the left, a red ceramic mug on the right,
    and a wooden cube block in the background center on a white surface.

  [3/4] Phrase Grounding  → 'the water bottle'
  Grounded 'the water bottle' → 1 region(s):
    • [85.0, 120.0, 340.0, 950.0]

  [4/4] Dense Region Captions
  3 region(s) captioned:
    • clear plastic water bottle with white cap
    • red ceramic coffee mug with black liquid inside
    • small natural wood cube block

╔══════════════════════════════════════════════════╗
║  SmolVLM2 Planning  [2b]                         ║
╚══════════════════════════════════════════════════╝
[SmolVLM2] Loading HuggingFaceTB/SmolVLM2-Instruct...
[SmolVLM2] Ready.

  [1/3] Scene Description
  The scene shows three objects on a white surface. A clear plastic water
  bottle stands upright on the left side. A red ceramic mug sits on the
  right, with its handle facing right. A small wooden cube is positioned
  in the upper center background area.

  [2/3] Bimanual Action Plan
  Task: "Pick up the water bottle with both arms and rotate it 90 degrees clockwise"

  SCENE: Clear water bottle on the left side, standing upright.

  PLAN:
  Step 1 | LEFT ARM:  Move to left side of bottle at mid-height, open gripper
  Step 1 | RIGHT ARM: Move to right side of bottle at mid-height, open gripper
  Step 2 | BOTH ARMS: Close grippers simultaneously to grasp bottle firmly
  Step 3 | BOTH ARMS: Lift bottle 5-10cm off the surface
  Step 4 | BOTH ARMS: Rotate clockwise 90 degrees (left arm moves forward,
                       right arm moves backward)
  Step 5 | BOTH ARMS: Lower bottle back to surface
  Step 6 | BOTH ARMS: Open grippers and retract arms

  [3/3] Grasp Point Suggestions  → 'the water bottle'
  LEFT ARM should grasp the left side of the bottle at approximately
  1/3 height from the bottom, where the bottle is widest.
  RIGHT ARM should grasp the right side at the same height, mirroring
  the left arm, to ensure balanced grip during rotation.

╔══════════════════════════════════════════════════╗
║  Saving Results                                  ║
╚══════════════════════════════════════════════════╝
  Saved → output/test1_result_detections.jpg
  Saved → output/test1_result_grounding.jpg
  Saved → output/test1_result.jpg       ← composite (original | OD | grounding)
```

> [!NOTE]
> **First run**: Models download automatically (~3GB Florence-large, ~4.5GB SmolVLM2). This takes several minutes depending on your connection. Subsequent runs are instant (cached in `~/.cache/huggingface/`).

---

## Test 2 — Perception only (faster, good for iterating on Florence-2)

```bash
python run_pipeline.py \
  --image sample_scene.png \
  --ground-phrase "the red mug" \
  --perception-only \
  --save output/test2_mug.jpg
```

**Why**: Lets you test Florence-2 grounding quickly without waiting for SmolVLM2. Good when iterating over different phrases or images.

---

## Test 3 — Try the wooden block (rotation task)

```bash
python run_pipeline.py \
  --image sample_scene.png \
  --task "Grasp the wooden block with both arms and move it 15cm to the left" \
  --ground-phrase "the wooden block" \
  --save output/test3_block.jpg
```

**Why**: Tests a translation (move left/right) instead of rotation — different bimanual coordination pattern.

---

## How to interpret the outputs

### Florence-2 detections ✅ / ⚠️

| Result | What it means |
|---|---|
| All 3 objects detected with correct labels | ✅ Model understands the scene |
| Bounding boxes look tight around objects | ✅ Good spatial precision |
| Objects detected but labelled wrong (e.g. "bottle" → "container") | ⚠️ Normal — labels vary, shape is correct |
| A grounding phrase returns no bbox | ⚠️ Try rephrasing (e.g. "water bottle" instead of "the bottle") |
| Detects 5+ phantom objects | ⚠️ Common with cluttered backgrounds |

### SmolVLM2 plan ✅ / ⚠️

| Result | What it means |
|---|---|
| Plan has LEFT ARM / RIGHT ARM steps | ✅ Bimanual prompt is working |
| Step descriptions are physically plausible | ✅ Model understands manipulation |
| Plan is vague ("move arm to object") | ⚠️ Normal for 2B model — rephrase task more specifically |
| Arms assigned to wrong sides of object | ⚠️ Note this — important for FYP — may need prompt tuning |
| Completely ignores the image | ⚠️ Rare; try with `--smolvlm-size 500m` to rule out memory issue |

### Output images

Three images are saved:
- `*_detections.jpg` — all detected objects with colored bounding boxes
- `*_grounding.jpg` — the specific grounded object highlighted in red
- `*_result.jpg` — side-by-side composite of all three views

---

## Timing expectations

| Stage | First run (download) | Subsequent runs |
|---|---|---|
| Florence-2-large load | ~2 min | ~10 sec |
| SmolVLM2 load | ~3 min | ~15 sec |
| Florence-2 inference | ~2–5 sec | ~2–5 sec |
| SmolVLM2 inference | ~5–15 sec | ~5–15 sec |
| **Total end-to-end** | **~10 min first run** | **~40 sec** |

---

## Common issues

> [!WARNING]
> **CUDA out of memory**: Unlikely on 8.5GB, but if it happens, add `--smolvlm-size 500m` to use the smaller SmolVLM2 model (~1GB instead of ~4.5GB).

> [!WARNING]
> **"trust_remote_code" warning**: Florence-2 requires `trust_remote_code=True`. This is already set in the code — safe to proceed.

> [!NOTE]
> **HuggingFace rate limit**: If downloads fail, set your HF token: `export HF_TOKEN=your_token_here` (same token already in `vlm_test.py`).

---

## Next steps for your FYP

Once the pipeline validates well on static images:

1. **Use your robot's actual camera** — replace `sample_scene.png` with frames from the robot
2. **Benchmark grounding accuracy** — test Florence-2-large vs base on real scenes
3. **Refine the bimanual prompt** — tune `_BIMANUAL_CONTEXT` in `smolvlm_planner.py` for your specific arm setup
4. **Extract bounding box → 3D coordinates** — this is the next big bridge to actual robot control
