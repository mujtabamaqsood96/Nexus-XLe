# Task Tracker

## Completed Tasks

- `[x]` Build the Verification Loop
  - `[x]` Update `_BIMANUAL_CONTEXT` in `smolvlm_planner.py` to request `[TARGET: <noun>]` tags.
  - `[x]` Update `app.py` to import `re`.
  - `[x]` Add parsing logic in `app.py` to extract `[TARGET: xxx]` strings from the generated plan.
  - `[x]` Add verification loop to automatically call `florence.ground_phrase` on the targets.
  - `[x]` Add UI feedback (warning messages if missing, green bounding boxes if found).
- `[x]` Test Verification Loop and deliver to user
  - `[x]` Handle phrase grounding hallucination (do not draw orange boxes on incorrect objects).
  - `[x]` Resolve overlapping bounding boxes by dynamically suppressing standard detections.
  - `[x]` Ensure missed objects (e.g. wooden block) are drawn correctly in green.
- `[x]` Implement Advanced FYP Features
  - `[x]` Build pixel-level referring expression segmentation for each detected object in the scene (panoptic segmentation style).
  - `[x]` Build bimanual grasp point overlay showing Left Arm (Cyan) and Right Arm (Pink) target crosshairs on verified targets.
  - `[x]` Connect controls as checkboxes in the Gradio Web UI.
- `[x]` Implement "Spatial Chain-of-Thought" Prompting Framework
  - `[x]` Update `smolvlm_planner.py` to force an Analysis Phase (Translation/Rotation axis).
  - `[x]` Ensure VLM outputs spatial geometry analysis before planning arm movements.
  - `[x]` Fix repetition_penalty conflict causing hallucinations (reduced from 1.3 to 1.05).
  - `[x]` Add scene-grounded planning (inject `scene_desc` into planner prompt).
  - `[x]` Shorten prompt to prevent Stochastic Parrot problem with 2.2B model.
  - `[x]` Fix few-shot example object collision (changed example objects to green cylinder).
  - `[x]` Add bimanual post-processing safety net (`_postprocess_bimanual_plan()`).
  - `[x]` Add input logging to `app.py` (text fields + toggle states).

## Next Tasks

- `[ ]` Define and Test Specific Collaborative Scenarios
  - `[ ]` Test Scenario 1: Collaborative Handover — `Pick up the red mug with the Left Arm and hand it over to the Right Arm.`
  - `[ ]` Test Scenario 2: Asymmetric Lifting — `Lift the wooden block from both ends simultaneously.`
  - `[ ]` Test Scenario 3: Coordinated Pouring — `Hold the red mug steady with the Left Arm while the Right Arm pours water from the bottle into it.`
  - `[ ]` Document results and model limitations for FYP report.
- `[ ]` Task 3: Success Detection (Before/After Verification)
  - `[ ]` Update `app.py` UI to accept a 'Before' and 'After' image.
  - `[ ]` Implement `verify_task_success()` in `smolvlm_planner.py` to compare Before/After images.
  - `[ ]` Output structured `STATUS: SUCCESS/FAILURE` + `REASON: ...` format.
  - `[ ]` Run Florence-2 on After image to verify final object positions.
- `[ ]` Task 4: Collision Warning (Safety)
  - `[ ]` Compute spatial overlaps between LEFT ARM and RIGHT ARM target bounding boxes.
  - `[ ]` Print warning and halt execution if significant overlap detected.
- `[ ]` Task 5: Closed-Loop MuJoCo Simulation (Future Phase)
  - `[ ]` Connect `app.py` to MuJoCo simulation environment.
  - `[ ]` Capture virtual camera feeds from MuJoCo viewer.
  - `[ ]` Send joint velocity commands back to actuators.
