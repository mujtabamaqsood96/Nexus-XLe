# 📚 Relevant Literature & Source Review

This document contains a structured analysis of key academic publications relevant to **Project 26M005: Vision Language Action (VLA) for Collaborative Robot Control**. These papers provide the theoretical foundation for our zero-shot generalization framework, bimanual cooperation, and rogue arm mitigation strategies.

---

## 🔍 Quick-Reference Objective Mapping

| Paper Title | Main Topic | Relevance Level | Fyp Objective mapped |
|---|---|---|---|
| **Florence-2** (Xiao et al., 2024) | Vision Foundation Models | **Essential** | Objective 1 (Perception & Grounding) |
| **Exploring Robustness of VLA** (Lu et al., 2025) | Sensor/Adversarial Attacks | **Essential** | Objective 2 & 4 (Rogue Arm & Mitigation) |
| **Efficient VLA Survey** (Yu et al., 2025) | Embedded/Edge VLAs | **Essential** | Methodology (SmolVLM2 Selection) |
| **VLA: Concepts & Progress** (Sapkota et al., 2025) | Embodied AI Overview | **Essential** | Literature Review (Chapters 1 & 2) |
| **Hybrid Fault-Tolerant Control** (Urrea, 2025) | Robotics Resilience | **Very High** | Objective 4 (Mitigation / FTC Math) |
| **Hierarchical Vision-Language** (Zhang et al., 2025) | Collaborative Task Planning | **Very High** | Objective 1 & 3 (Map-Reduce & Hierarchy) |
| **Advances in GRPO** (Liu et al., 2026) | RL Alignment / GRPO | **Moderate** | Future Work (VLM Policy Tuning) |

---

## 📄 Detailed Paper Evaluations

### 1. Florence-2: Advancing a unified representation for a variety of vision tasks
* **Authors**: B. Xiao, H. Wu, W Xu, X. Dai, H. Hu, Y. Lu, et al. (CVPR, 2024)
* **Status**: **Essential Core Citation**
* **Summary**: Introduces Florence-2, a unified vision foundation model that processes multiple spatial tasks (Object Detection, Referring Expression Segmentation, Phrase Grounding, OCR) via a single sequence-to-sequence prompt architecture.
* **Why it is relevant to your FYP**:
  * This is the **actual perception model** running in our codebase (`pipeline/florence_perception.py`).
  * Demonstrates how open-vocabulary visual tasks can be unified without fine-tuning, providing the baseline for our **Zero-Shot Generalization Framework**.
  * Fulfills **Objective 1**: Proves how we locate target objects, extract silhouettes (segmentation), and verify coordinates.

---

### 2. Exploring the Robustness of Vision-Language-Action Models against Sensor Attacks
* **Authors**: X. Lu, J. Chen, S. Xiao, Z. Jin, R. Zhou, X. Ji, et al. (ACM CCS, 2025)
* **Status**: **Essential Safety Citation**
* **Summary**: Examines the vulnerability of VLA models to sensor noise, anomalies, and physical/visual adversarial attacks, evaluating how visual deviations affect low-level action token outputs.
* **Why it is relevant to your FYP**:
  * Fulfills **Objective 2 (Rogue Arm)** & **Objective 4 (Mitigation)**: A "rogue robotic arm" acting disruptively in the environment introduces severe sensor/visual anomalies. 
  * This paper provides the academic precedent for why VLA models degrade under physical interference, justifying our need for the **Closed-Loop Verification Loop** to intercept erroneous plan commands before motor execution.

---

### 3. A survey on efficient vision-language-action models
* **Authors**: Z. Yu, B. Wang, P. Zeng, H. Zhang, J. Zhang, et al. (arXiv, 2025)
* **Status**: **Essential Methodology Citation**
* **Summary**: Reviews the state of the art in compressing, quantizing, and optimizing VLA models for resource-constrained edge devices and real-time robotic hardware control.
* **Why it is relevant to your FYP**:
  * Justifies your choice of the **HuggingFace "Smol" Ecosystem** (SmolVLM2-2.2B/500M and Florence-2-Large).
  * Embodied systems cannot rely on slow, high-latency cloud APIs (like GPT-4o) for real-time motor loop actions. This paper provides the literature support for local, lightweight VLM deployment on our FPGA/GPU setup.

---

### 4. Vision-language-action models: Concepts, progress, applications and challenges
* **Authors**: R. Sapkota, Y. Cao, K. I. Roumeliotis, M. Karkee (arXiv, 2025)
* **Status**: **Essential Literature Review Citation**
* **Summary**: A comprehensive survey of VLA models mapping out architecture progress, training datasets, reinforcement learning integration, and current limitations in real-world deployments.
* **Why it is relevant to your FYP**:
  * The perfect starting reference for your **Interim Report's Literature Review** (Chapter 2). It outlines the evolution of VLA models and the challenges of translating high-level prompts directly into motor actions.

---

### 5. Hybrid Fault-Tolerant Control in Cooperative Robotics: Advances in Resilience and Scalability
* **Authors**: C. Urrea (Actuators, 2025)
* **Status**: **Highly Relevant (Theory & Modeling)**
* **Summary**: Discusses the principles of Fault-Tolerant Control (FTC) in cooperative multi-robot systems, analyzing how systems maintain resilience and stability when individual actuators experience anomalies or physical disruptions.
* **Why it is relevant to your FYP**:
  * Fulfills **Objective 4 (Mitigation)**: A "rogue arm" represents a dynamic disturbance or fault in the collaborative system. 
  * Provides the theoretical framework to explain how the "good arm" can adapt its trajectory, shift tasks, or halt movement when a visual/spatial fault is detected.

---

### 6. A Hierarchical Vision-Language and Reinforcement Learning Framework for Robotic Task and Motion Planning in Collaborative Manipulation
* **Authors**: J. Zhang, C. Mu, X. Xu, L. Ren (IEEE L-RA, 2025)
* **Status**: **Highly Relevant (Architecture Design)**
* **Summary**: Proposes a hierarchical structure that splits collaborative robotics tasks into two layers: (1) Vision-Language models for high-level semantic task planning, and (2) Reinforcement Learning/Kinematics for low-level joint motion planning.
* **Why it is relevant to your FYP**:
  * Fulfills **Objective 1 (Collaboration)** and **Objective 3 (Task Decomposition)**.
  * Validates our codebase architecture: we use SmolVLM2 for the high-level semantic decomposition (`LEFT ARM` / `RIGHT ARM` steps), and Florence-2 / grasp overlays to bridge that plan directly to physical joint coordinates (Task and Motion Planning).

---

### 7. Advances in GRPO for Generation Models: A Survey
* **Authors**: Z. Liu, X. He, Y. Li (arXiv, 2026)
* **Status**: **Moderately Relevant (Future Scope)**
* **Summary**: Surveys Group Relative Policy Optimization (GRPO) — a reinforcement learning alignment technique that optimizes model output quality by comparing groups of candidate responses rather than relying on heavy value-function models (used in DeepSeek R1).
* **Why it is relevant to your FYP**:
  * While GRPO is traditionally used for text reasoning alignment, it represents the state-of-the-art in VLM policy tuning.
  * Can be cited in your **"Future Work"** section of the Interim Report as a potential method to fine-tune SmolVLM2's joint-coordinate generation directly using reinforcement learning on simulation outcomes in NVIDIA Isaac Sim.
