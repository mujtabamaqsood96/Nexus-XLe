# 🎓 FYP Context & Research Roadmap

**Project Code:** 26M005  
**Project Title:** Vision Language Action (VLA) for Collaborative Robot Control  
**Supervisors:** Ts. Dr. Ho Tatt Wei & Dr. Ahmad Bukhari Aujih  
**Collaboration:** Adelaide University  

---

## 📅 Academic Timeline & Deliverables (FYP 1)

*   **Week 6**: Progress Assessment 1 (Supervisor Evaluation of VLA setup)
*   **Week 7 (June 24, 2026)**: **Proposal Defence (Oral Presentation)** — *Feasibility, methodology, and prototype demonstrations.*
*   **Week 9**: Progress Assessment 2 (Ethics and responsibility evaluation)
*   **Week 10 (July 17, 2026)**: Draft Interim Report (Turnitin similarity < 25%)
*   **Week 12 (July 31, 2026)**: **Final Interim Report Submission** (50% of total grade)

---

## 🎯 Project Objectives vs. Current Codebase Mapping

Below is how our current Python codebase maps directly to your supervisor's core research objectives:

### Objective 1: Train & Develop VLA Control (Translation/Orientation)
*   **Current Progress**: 
    *   `pipeline/smolvlm_planner.py` uses SmolVLM2 to formulate natural language plans for object translation and orientation (e.g. *"Place the wooden block on the left side of the cup"*).
    *   `pipeline/florence_perception.py` uses Florence-2 to detect objects (`<OD>`) and ground them spatially (`<CAPTION_TO_PHRASE_GROUNDING>`).
    *   `app.py` computes coordinates and visualizes bimanual grasp points (`L GRASP` in Cyan and `R GRASP` in Pink) to extract translation and orientation control vectors.
*   **Ready for Proposal Defence?**: Yes! You have a fully interactive web dashboard (`app.py`) to demonstrate how VLA planning translates to physical spatial targets.

### Objective 2: Simulate Disruption (Rogue Arm)
*   **Concept**: A disruptive secondary arm acts in the environment to knock objects away, mislabel them, or block trajectories.
*   **Next Phase Plan**: Use our visualizers to detect unexpected bounding box/mask movements (e.g. object moves when the good arm is still static, signifying an external disruption).

### Objective 3: Task Decomposition (Map-Reduce Framework)
*   **Concept**: Decomposing a complex user task (Map) into discrete subtasks (e.g. grasp, lift, translate, place) and coordinating them across two arms (Reduce).
*   **Current Progress**: The SmolVLM2 planner decomposes commands into structured `Step X | LEFT ARM` / `Step X | RIGHT ARM` steps. 

### Objective 4: Mitigation & Safety Strategies
*   **Current Progress**: 
    *   **Verification Loop (Task 2)**: The pipeline cross-references plan targets against multi-source visual evidence to block the robot from reaching for hallucinated objects (e.g., catching a "screwdriver" hallucination).
    *   **Dynamic Suppression**: Suppresses overlapping bounding boxes to avoid coordinate ambiguity.
*   **Next Phase Plan**: **Task 4 (Collision Warning)** will compute spatial overlaps to mitigate physical arm collisions under rogue disruptions.

---

## 🛠️ Codebase Overview for AI Assistants

This repository contains an active prototype built with a PyTorch/CUDA-enabled environment (`vla_env`).

### Core Modules
1.  **`app.py`**: Gradio-based Web UI. Runs on `http://127.0.0.1:7860`. Coordinates the pipeline, verification loop, and rendering.
2.  **`run_pipeline.py`**: Command Line utility for running tasks offline.
3.  **`pipeline/florence_perception.py`**: Wrapper for Microsoft Florence-2-Large (object detection, segmentation, text grounding, OCR).
4.  **`pipeline/smolvlm_planner.py`**: Wrapper for SmolVLM2-2.2B-Instruct (bimanual plan generation, visual QA).
5.  **`pipeline/visualizer.py`**: Renders annotations, masks, and side-by-side composites.

### Custom Features Built
*   **Auto-Panoptic Segmentation**: Checking `Pixel-Level Segmentation (All Objects)` segments all detected objects automatically.
*   **Visual Grasp Points**: Verified target objects get custom Cyan (`L GRASP`) and Pink (`R GRASP`) crosshairs drawn on their boundaries.
*   **Hallucination Block**: Suspicious target objects (no text evidence in scene captions/descriptions) are flagged `⚠️ SUSPICIOUS` in logs and **no bounding box is drawn** to prevent mislabeling.

---

## 🐛 Bug Log & Resolved Issues

1.  **Gradio 6.0 UserWarning**: (Resolved) Theme parameter moved from Blocks constructor to the `launch(theme=...)` call.
2.  **Overlapping Bounding Boxes**: (Resolved) Implemented spatial IoU/containment check (`check_overlap`). If a verified plan target overlaps with a standard detection, the standard box is suppressed.
3.  **Grounding Hallucinations**: (Resolved) Florence-2 phrase grounding is overly eager and always returns a box even if the object is missing (e.g., pointing "screwdriver" at a bottle). Fixed by cross-referencing against the combined text evidence string.

---

## 🗺️ Next Tasks to Implement

### Task 3: Success Detection (Verification)
*   **Goal**: Upload a "Before" image and an "After" image of the workspace.
*   **Implementation**: Ask SmolVLM2: *"Compare these two images. Did the robot successfully complete [Task]?"*

### Task 4: Collision Warning (Safety)
*   **Goal**: Prevent dual-arm collisions during collaborative motion.
*   **Implementation**: If bounding boxes for `LEFT ARM` and `RIGHT ARM` targets (or grippers) overlap significantly, print a warning and halt execution.
