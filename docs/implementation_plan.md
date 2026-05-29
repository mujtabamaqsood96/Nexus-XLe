# Implementation Plan: Next Steps

The Spatial Chain-of-Thought Prompting Framework is now fully operational. The pipeline reliably produces structured `SCENE → ANALYSIS → PLAN` outputs with correct bimanual formatting, TARGET tags, and spatial axis reasoning.

---

## Immediate Next: Collaborative Scenarios Testing

Three hard-to-program physical tasks to test with the Spatial Chain-of-Thought prompt:

### Scenario 1: Collaborative Handover (Reach Limitation)
* **Goal**: Hand over an object from one arm to another to extend reach.
* **Task Phrase**: `Pick up the red mug with the Left Arm and hand it over to the Right Arm so it can be placed on the far right.`
* **Expected Spatial Reasoning**:
  - Left Arm translates along X/Y to reach the mug, grasps it, translates to central handover zone.
  - Right Arm translates to central zone, grasps mug, Left Arm releases.
  - Right Arm translates and places the mug.

### Scenario 2: Asymmetric Lifting (Coordination & Balance)
* **Goal**: Lift a long/heavy object simultaneously using both arms.
* **Task Phrase**: `Lift the wooden block from both ends simultaneously and keep it horizontal.`
* **Expected Spatial Reasoning**:
  - Both arms translate along X/Y to grasp opposite ends.
  - Both arms translate along Z-axis concurrently.

### Scenario 3: Coordinated Pouring
* **Goal**: Pour water from the bottle into the mug.
* **Task Phrase**: `Hold the red mug steady with the Left Arm while the Right Arm pours water from the bottle into it.`
* **Expected Spatial Reasoning**:
  - Left Arm stabilizes the mug (support role).
  - Right Arm grasps bottle, lifts above mug (Z-axis), rotates around Y-axis to pour.

---

## Task 3: Success Detection (Before/After Verification)

### Concept
Currently the pipeline verifies objects exist *before* the task. To close the loop fully, we need **post-task verification** comparing Before and After states.

### Implementation Steps

#### 1. Update the Web UI (`app.py`)
* Add a second image input component for the **After** image.
* **Before Image**: Initial scene (used for planning).
* **After Image**: Final scene (after robot attempts the task).

#### 2. Implement `verify_task_success()` in `smolvlm_planner.py`
* Inputs: Before image, After image, original Task text, generated Plan text.
* Outputs:
  - `STATUS`: `SUCCESS` or `FAILURE`
  - `REASON`: Detailed explanation of what changed and why it succeeded/failed.

#### 3. Florence-2 After-Image Verification
* Run Florence-2 object detection on the After image.
* Compare object positions between Before and After.
* Draw green bounding boxes on After image to confirm success visually.

---

## Task 4: Collision Warning (Safety)

* Compute spatial overlaps between LEFT ARM and RIGHT ARM target bounding boxes.
* If significant overlap detected, print warning and halt execution.
* Prevents physical arm collisions during collaborative motion.

---

## Task 5: Closed-Loop MuJoCo Simulation (Future Phase)

* Connect `app.py` to MuJoCo simulation environment using `XLeRobot/simulation/mujoco/`.
* Capture virtual camera feeds from the viewer, pass to VLA pipeline.
* Send joint velocity commands back to actuators.
* Inject adversarial robot model for rogue arm simulation.

---

## Technical Notes for Next Chat Session

### Current Architecture
* **Perception**: Florence-2-Large (`pipeline/florence_perception.py`) — Object Detection, Phrase Grounding, Segmentation, OCR
* **Planning**: SmolVLM2-2.2B-Instruct (`pipeline/smolvlm_planner.py`) — Scene Description, Bimanual Planning, Visual QA
* **Verification**: Closed-loop in `app.py` — multi-source evidence + phrase grounding
* **UI**: Gradio Web UI (`app.py`) on `http://127.0.0.1:7860`
* **CLI**: `run_pipeline.py` for command-line testing

### Key Parameters
* `repetition_penalty = 1.05` for bimanual planning (NOT higher — causes hallucinations)
* `do_sample = False` for deterministic greedy decoding
* `max_new_tokens = 400` for planning, `300` for scene description
* Post-processing via `_postprocess_bimanual_plan()` catches single-arm plans

### Environment
* Python 3.12, `vla_env` virtualenv
* CUDA-enabled, RTX 5060 Laptop GPU (8.5GB VRAM)
* WSL2 on Windows
* Activate: `source ~/fyp/vla_env/bin/activate`
* Run: `python app.py` → opens `http://127.0.0.1:7860`
