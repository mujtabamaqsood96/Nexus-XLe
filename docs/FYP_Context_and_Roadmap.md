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

## 🧠 Core Methodology: Zero-Shot Generalization Framework

A major challenge in collaborative robotics is that traditional control systems require training low-level imitation learning or reinforcement learning policies on thousands of hours of demonstrations for specific objects. If the object changes shape, shifts 5cm, or a new item is introduced, the system fails.

**Nexus-XLe solves this by designing a Zero-Shot Generalization Framework:**
1. **No Target Dataset Training**: Rather than training or fine-tuning models from scratch, we leverage pre-trained, multi-modal **foundation models** (SmolVLM2 & Florence-2) as zero-shot planners.
2. **Open-Vocabulary Perception**: Florence-2 performs open-vocabulary object detection, text-grounding, and referring expression segmentation. It can locate and segment arbitrary, unseen objects (e.g. customized blocks, cups, tools) without needing target-specific training.
3. **Brain-to-Motor Translation Layer**:
   - SmolVLM2 plans the step-by-step bimanual coordination.
   - Florence-2 grounds the physical pixel coordinates of the targets.
   - The cyan/pink grasp overlays extract the 2D spatial pixel targets, which map directly to the robot's inverse kinematics (IK) solver to control the joints of the dual SO-101 arms.
4. **Generalization Safety (Verification Loop)**: Since foundational models are prone to hallucinating actions or targets, the closed-loop Verification Loop checks and verifies that plan targets actually exist in the physical space, creating a safe, training-free, and generalizable collaborative environment.

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
*   **Status**: Implemented in `app.py` — upload an **After Image** and enable **Task 3: Verify Success (Before vs After)**.

### Task 4: Collision Warning (Safety)
*   **Goal**: Prevent dual-arm collisions during collaborative motion.
*   **Implementation**: If bounding boxes for `LEFT ARM` and `RIGHT ARM` targets (or grippers) overlap significantly, print a warning and halt execution.
*   **Status**: Implemented in `app.py` — enable **Task 4: Collision Warning (Safety)** to halt on overlapping L/R verified targets.
