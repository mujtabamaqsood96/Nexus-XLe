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

Furthermore, autonomous single-agent scenarios are still not well established—only routine or predictable tasks are handled well, while tasks requiring adaptation remain largely unsolved. Collaborative robotics introduces scenarios with even more adaptability and variability. This presents an opportunity to use multiple smaller, low-spec robots rather than a large single robot, utilizing the reasoning and generalization capabilities of VLMs to handle tasks that are difficult to program conventionally.

**Nexus-XLe solves this by designing a Zero-Shot Generalization Framework:**
1. **No Target Dataset Training**: Rather than training or fine-tuning models from scratch, we leverage pre-trained, multi-modal **foundation models** (SmolVLM2 & Florence-2) as zero-shot planners.
2. **Visual Feedback Guidance**: We rely heavily on real-time visual feedback to guide the action model, allowing the system to adapt dynamically to changing states without relying on brittle pre-programmed trajectories.
3. **Map-Reduce Task Decomposition**: We apply a novel Map-Reduce approach to robotics, where complex tasks are decomposed (Map) into discrete subtasks, and then coordinated across multiple robots (Reduce).
4. **Open-Vocabulary Perception**: Florence-2 performs open-vocabulary object detection, text-grounding, and referring expression segmentation. It can locate and segment arbitrary, unseen objects (e.g. customized blocks, cups, tools) without needing target-specific training.
5. **Brain-to-Motor Translation Layer**:
   - SmolVLM2 plans the step-by-step bimanual coordination.
   - Florence-2 grounds the physical pixel coordinates of the targets.
   - The cyan/pink grasp overlays extract the 2D spatial pixel targets, which map directly to the robot's inverse kinematics (IK) solver to control the joints of the dual SO-101 arms.
6. **Generalization Safety (Verification Loop)**: Since foundational models are prone to hallucinating actions or targets, the closed-loop Verification Loop checks and verifies that plan targets actually exist in the physical space, creating a safe, training-free, and generalizable collaborative environment.

---

## 🎯 Project Objectives vs. Current Codebase Mapping

Below is how our current Python codebase maps directly to your supervisor's core research objectives:

### Objective 1: Develop a Collaborative Robotics Prompting Framework
*   **Current Progress**: 
    *   `pipeline/smolvlm_planner.py` uses SmolVLM2 to formulate natural language plans for object translation and orientation (e.g. *"Place the wooden block on the left side of the cup"*).
    *   `pipeline/florence_perception.py` uses Florence-2 to detect objects (`<OD>`) and ground them spatially (`<CAPTION_TO_PHRASE_GROUNDING>`).
    *   `app.py` computes coordinates and visualizes bimanual grasp points (`L GRASP` in Cyan and `R GRASP` in Pink) to extract translation and orientation control vectors.
*   **Ready for Proposal Defence?**: Yes! You have a fully interactive web dashboard (`app.py`) to demonstrate how VLA planning translates to physical spatial targets.

### Objective 2: Simulate Disruption & Assess Adaptability
*   **Concept**: A disruptive secondary arm acts in the environment to knock objects away, mislabel them, or block trajectories.
*   **Next Phase Plan**: Use our visualizers to detect unexpected bounding box/mask movements (e.g. object moves when the good arm is still static, signifying an external disruption), relying on visual feedback to guide adaptation.

### Objective 3: Mitigation & Safety Strategies
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
2.  **Overlapping Bounding Boxes**: (Resolved) Implemented spatial IoU/containment check (`check_overlap`). If a verified target overlaps with a standard detection, the standard box is suppressed.
3.  **Grounding Hallucinations**: (Resolved) Florence-2 phrase grounding is overly eager and always returns a box even if the object is missing. Fixed by cross-referencing against the combined text evidence string.
4.  **Repetition Penalty Hallucinations**: (Resolved) `repetition_penalty=1.3` caused SmolVLM2 to skip SCENE/ANALYSIS blocks and hallucinate random objects in TARGET tags (e.g. `paper towel roll`). Root cause: the penalty applied to prompt tokens too, penalizing the model for repeating necessary words like `LEFT ARM`, `Step 1`, and target object names. Fixed by reducing to `1.05`.
5.  **Stochastic Parrot (Example Copying)**: (Resolved) The 2.2B model copied few-shot example actions verbatim when the example objects matched the scene objects. Fixed by using unrelated example objects (green cylinder) and shortening the prompt.
6.  **Single-Arm Plans for Bimanual Tasks**: (Resolved) SmolVLM2-2.2B sometimes ignores "both arms" instructions and only uses one arm. Fixed by adding `_postprocess_bimanual_plan()` post-processing safety net.

---

## 🗺️ Next Tasks to Implement

### Immediate Next: Test Collaborative Scenarios
*   **Goal**: Test 3 hard-to-program bimanual tasks against the Spatial Chain-of-Thought prompt.
*   **Scenarios**: Collaborative Handover, Asymmetric Lifting, Coordinated Pouring.
*   **See**: `docs/implementation_plan.md` for full details.

### Task 3: Success Detection (Before/After Verification)
*   **Goal**: Upload a "Before" image and an "After" image of the workspace.
*   **Implementation**: Implement `verify_task_success()` in `smolvlm_planner.py`. Compare images and output `STATUS: SUCCESS/FAILURE` + `REASON`.

### Task 4: Collision Warning (Safety)
*   **Goal**: Prevent dual-arm collisions during collaborative motion.
*   **Implementation**: Compute spatial overlaps between LEFT ARM and RIGHT ARM target bounding boxes. Halt execution if significant overlap detected.

### Task 5: Closed-Loop Simulation in MuJoCo (Upcoming Phase)
*   **Goal**: Transition our VLA planning and verification pipeline from static 2D images to a real-time 3D physics-based simulation of the dual-arm XLeRobot mobile base.
*   **MuJoCo Setup**: Leverage the official **XLeRobot MuJoCo Controller** (using `mujoco` and `mujoco-viewer` inside `XLeRobot/simulation/mujoco/`) which defines the robot's physical structures, joint limits, and meshes in `xlerobot.xml` and `scene.xml`.
*   **Rogue Arm Simulation**: Inject a secondary adversarial robot model in the MuJoCo simulator environment to block trajectories, displace target objects, or generate visual occlusions.
*   **Closed-Loop Integration**: Connect `app.py` directly to the MuJoCo simulation environment, capturing virtual camera feeds from the viewer, passing them to the VLA verification pipeline, and sending joint velocity commands back to the actuators in MuJoCo.
