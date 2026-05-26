import gradio as gr
from PIL import Image
import torch
import textwrap
import os
import glob
import re

# Import our pipeline modules
from pipeline.florence_perception import FlorencePerception
from pipeline.smolvlm_planner import SmolVLMPlanner
from pipeline.visualizer import draw_detections, draw_segmentation, draw_grounding

# Global variables for models (loaded on first request to save startup time, or loaded globally)
florence = None
planner = None

def load_models_if_needed():
    global florence, planner
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    if florence is None:
        print("[WebUI] Loading Florence-2...")
        florence = FlorencePerception(model_id="microsoft/Florence-2-large", device=device)
    if planner is None:
        print("[WebUI] Loading SmolVLM2...")
        planner = SmolVLMPlanner(model_id="HuggingFaceTB/SmolVLM2-2.2B-Instruct", device=device)

def check_overlap(box1, box2, threshold=0.4):
    """
    Check if box1 and box2 overlap significantly.
    box = [x1, y1, x2, y2]
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False
        
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    unionArea = box1Area + box2Area - interArea
    if unionArea > 0:
        iou = interArea / float(unionArea)
        if iou >= threshold:
            return True
            
    min_area = min(box1Area, box2Area)
    if min_area > 0 and (interArea / float(min_area)) >= 0.7:
        return True
        
    return False

def check_collision_risk(box1, box2, iou_threshold=0.2, containment_threshold=0.55):
    """
    Collision warning helper for Task 4.
    Uses a lower IoU threshold than suppression, since arms reaching into the same
    workspace region is risky even with modest overlap.
    """
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0:
        return False

    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    unionArea = box1Area + box2Area - interArea

    if unionArea > 0:
        iou = interArea / float(unionArea)
        if iou >= iou_threshold:
            return True

    min_area = min(box1Area, box2Area)
    if min_area > 0 and (interArea / float(min_area)) >= containment_threshold:
        return True

    return False

_PLAN_LINE_RE = re.compile(
    r"^Step\s*(\d+)\s*\|\s*(LEFT ARM|RIGHT ARM|BOTH ARMS)\s*:\s*(.*?)\s*\[TARGET:\s*(.*?)\s*\]\s*$",
    re.IGNORECASE,
)

def parse_plan_arm_targets(plan_text: str):
    """
    Parse SmolVLM2 plan lines into per-step target assignments.
    Returns: {step_num: {'LEFT ARM': str|None, 'RIGHT ARM': str|None, 'BOTH ARMS': str|None}}
    """
    steps = {}
    for raw_line in (plan_text or "").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = _PLAN_LINE_RE.match(line)
        if not match:
            continue

        step_num = int(match.group(1))
        arm = match.group(2).upper()
        target = match.group(4).strip()
        target_value = None if target.lower() == "none" else target

        entry = steps.setdefault(step_num, {"LEFT ARM": None, "RIGHT ARM": None, "BOTH ARMS": None})
        entry[arm] = target_value

    return steps

SEG_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD", "#98D8C8", "#F39C12"]

def draw_grasp_points(image: Image.Image, bbox: list, label: str) -> Image.Image:
    """
    Draw Cyan (Left Arm) and Pink (Right Arm) grasp points on the left and right edges
    of the bounding box for bimanual manipulation.
    """
    from PIL import ImageDraw
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 11)
    except Exception:
        font = ImageFont.load_default()
        
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    
    x1, y1, x2, y2 = bbox
    y_mid = (y1 + y2) / 2
    r = 7  # crosshair circle radius
    
    # Left Arm Grasp Point (Cyan) - Place slightly inside the left edge
    xl = x1 + (x2 - x1) * 0.05
    color_l = "#00F0FF"  # neon cyan
    draw.ellipse([xl - r, y_mid - r, xl + r, y_mid + r], outline=color_l, width=3)
    draw.line([xl - r - 3, y_mid, xl + r + 3, y_mid], fill=color_l, width=2)
    draw.line([xl, y_mid - r - 3, xl, y_mid + r + 3], fill=color_l, width=2)
    
    # Text background box to make label visible
    text_bbox_l = draw.textbbox((xl + 10, y_mid - 7), "L GRASP", font=font)
    draw.rectangle(
        [text_bbox_l[0] - 2, text_bbox_l[1] - 1, text_bbox_l[2] + 2, text_bbox_l[3] + 1],
        fill="black"
    )
    draw.text((xl + 10, y_mid - 7), "L GRASP", fill=color_l, font=font)
    
    # Right Arm Grasp Point (Neon Pink) - Place slightly inside the right edge
    xr = x2 - (x2 - x1) * 0.05
    color_r = "#FF007F"  # neon pink
    draw.ellipse([xr - r, y_mid - r, xr + r, y_mid + r], outline=color_r, width=3)
    draw.line([xr - r - 3, y_mid, xr + r + 3, y_mid], fill=color_r, width=2)
    draw.line([xr, y_mid - r - 3, xr, y_mid + r + 3], fill=color_r, width=2)
    
    text_bbox_r = draw.textbbox((xr - 62, y_mid - 7), "R GRASP", font=font)
    draw.rectangle(
        [text_bbox_r[0] - 2, text_bbox_r[1] - 1, text_bbox_r[2] + 2, text_bbox_r[3] + 1],
        fill="black"
    )
    draw.text((xr - 62, y_mid - 7), "R GRASP", fill=color_r, font=font)
    
    return img

def run_pipeline(
    image: Image.Image,
    after_image: Image.Image,
    task_text: str,
    ground_phrase: str,
    segment_phrase: str,
    ocr_checkbox: bool,
    proposals_checkbox: bool,
    question_text: str,
    segment_all_checkbox: bool,
    grasp_overlay_checkbox: bool,
    success_detection_checkbox: bool,
    collision_warning_checkbox: bool
):
    if image is None:
        return None, "Please upload an image first."
        
    # Ensure RGB
    image = image.convert("RGB")
    
    # Load models dynamically (so UI starts instantly)
    load_models_if_needed()
    
    # We will accumulate drawn layers on this image
    annotated_image = image.copy()
    
    log_output = []
    
    # ==========================
    # 1. Florence-2 Perception
    # ==========================
    log_output.append("--- FLORENCE-2 PERCEPTION ---")
    
    # Object Detection (standard)
    detections = florence.detect_objects(image)
    detections_to_draw = list(detections) if detections else []
    if detections:
        log_output.append(f"Detected {len(detections)} objects.")
        
        # Pixel-Level Segmentation for All Objects
        if segment_all_checkbox:
            log_output.append("Running pixel-level segmentation for all detected objects...")
            for i, det in enumerate(detections):
                label = det["label"]
                # Query segment_object for this label
                seg_result = florence.segment_object(image, label)
                if seg_result and seg_result.get("polygons"):
                    color = SEG_COLORS[i % len(SEG_COLORS)]
                    annotated_image = draw_segmentation(annotated_image, seg_result, color=color, alpha=100)
                    log_output.append(f"  ✓ Segmented '{label}' at pixel level.")
    
    # Grounding (if phrase given)
    if ground_phrase.strip():
        grounding = florence.ground_phrase(image, ground_phrase)
        if grounding and grounding.get("bboxes"):
            annotated_image = draw_grounding(annotated_image, grounding, ground_phrase, color="#FFD166")
            log_output.append(f"Grounded '{ground_phrase}' at {len(grounding['bboxes'])} locations.")
    
    # Segmentation (if phrase given)
    if segment_phrase.strip():
        segmentation = florence.segment_object(image, segment_phrase)
        if segmentation:
            annotated_image = draw_segmentation(annotated_image, segmentation, color="#4ECDC4", alpha=140)
            log_output.append(f"Segmented '{segment_phrase}'.")
            
    # OCR
    if ocr_checkbox:
        ocr = florence.read_text(image)
        if ocr and ocr.get("bboxes"):
            annotated_image = draw_detections(annotated_image, [{"label": l, "bbox": b} for l, b in zip(ocr["labels"], ocr["bboxes"])])
            log_output.append(f"Read {len(ocr['bboxes'])} text regions.")
            
    # Region Proposals
    if proposals_checkbox:
        props = florence.region_proposals(image)
        if props and props.get("bboxes"):
            annotated_image = draw_detections(annotated_image, [{"label": "Proposal", "bbox": b} for b in props["bboxes"]])
            log_output.append(f"Found {len(props['bboxes'])} region proposals.")

    # ==========================
    # 2. SmolVLM2 Planning
    # ==========================
    log_output.append("\n--- SMOLVLM2 PLANNING ---")
    
    # Scene Description
    scene_desc = planner.describe_scene(image)
    log_output.append("SCENE DESCRIPTION:\n" + textwrap.fill(scene_desc, width=60))
    
    # Question Answering
    if question_text.strip():
        answer = planner.answer_question(image, question_text)
        log_output.append("\nQUESTION: " + question_text)
        log_output.append("ANSWER: " + textwrap.fill(answer, width=60))
    
    # Task Plan
    if task_text.strip():
        plan_text = planner.plan_bimanual_action(image, task_text)
        log_output.append("\nBIMANUAL PLAN:\n" + plan_text)
        step_arm_targets = parse_plan_arm_targets(plan_text)
        
        # ==========================
        # 3. VERIFICATION LOOP
        # ==========================
        # Strategy 1: Try to extract explicit [TARGET: ...] tags
        targets = re.findall(r"\[TARGET:\s*(.*?)\]", plan_text, re.IGNORECASE)
        targets = [t for t in targets if t.strip().lower() != "none"]
        
        # Strategy 2 (Fallback): If the model didn't use [TARGET:] tags,
        # extract noun phrases from the plan and cross-reference against
        # Florence-2's detected objects to find what's real and what's hallucinated.
        if not targets:
            # Build a list of known object labels from Florence-2 detection
            known_labels = set()
            if detections:
                for det in detections:
                    known_labels.add(det["label"].lower())
            
            # Extract candidate nouns: look for "the <noun>" or "a <noun>" patterns
            # plus any multi-word object phrases from the plan text
            candidates = re.findall(
                r'(?:the|a|an)\s+([\w\s]+?)(?:\s+(?:with|from|to|on|in|into|onto|off|up|down|and|,|\.|$))',
                plan_text.lower()
            )
            # Also try to match "<verb> the <noun phrase>" patterns  
            candidates += re.findall(
                r'(?:pick up|grasp|grab|hold|place|move|lift|rotate|push|pull|pour)\s+(?:the\s+)?([\w\s]+?)(?:\s+(?:with|from|to|on|in|into|onto|off|and|,|\.|$))',
                plan_text.lower()
            )
            # Deduplicate and clean
            targets = list(set(c.strip() for c in candidates if len(c.strip()) > 2))
        
        # Build evidence text from multiple sources for cross-referencing.
        # OD labels alone are too narrow (e.g. OD says "coffee cup" but plan says "red coffee mug").
        # We gather text from:
        #   1. OD labels (e.g. "bottle", "coffee cup")
        #   2. Florence-2 scene caption (e.g. "a wooden block and a red mug on a table")
        #   3. SmolVLM2 scene description (already generated above)
        known_labels = set()
        if detections:
            for det in detections:
                known_labels.add(det["label"].lower().strip())
        
        # Get Florence-2's detailed caption as additional evidence
        florence_caption = florence.caption(image, detail="more").lower()
        
        # Combine all text evidence into one string for substring searching
        evidence_text = " ".join(known_labels) + " " + florence_caption + " " + scene_desc.lower()
        
        if targets:
            log_output.append("\n--- VERIFICATION LOOP ---")
            log_output.append(f"OD labels: {', '.join(known_labels) if known_labels else 'none'}")
            unique_targets = list(set(targets))
            verified_bboxes = {}
            for target in unique_targets:
                target_lower = target.lower().strip()
                
                # Check 1: Is the target mentioned in ANY evidence source?
                # Split target into individual words and check if most words appear
                target_words = target_lower.split()
                words_found = sum(1 for w in target_words if w in evidence_text)
                evidence_match = words_found >= max(1, len(target_words) - 1)  # allow 1 word miss
                
                # Check 2: Can Florence-2 ground it?
                grounding = florence.ground_phrase(image, target)
                grounded = grounding and grounding.get("bboxes")
                
                if evidence_match and grounded:
                    # High confidence: evidence mentions it AND grounding found it
                    log_output.append(f"✅ VERIFIED [{target}]: Confirmed by scene evidence + grounding.")
                    verified_bboxes[target_lower] = grounding["bboxes"][0]
                    
                    # Draw green box for verified target
                    annotated_image = draw_grounding(annotated_image, grounding, f"✅ {target}", color="#00FF00")
                    
                    # Draw bimanual grasp points if enabled
                    if grasp_overlay_checkbox and grounding.get("bboxes"):
                        bbox = grounding["bboxes"][0]
                        annotated_image = draw_grasp_points(annotated_image, bbox, target)
                    
                    # Check and suppress overlapping standard detections
                    for g_bbox in grounding["bboxes"]:
                        to_remove = []
                        for det in detections_to_draw:
                            if check_overlap(g_bbox, det["bbox"]):
                                to_remove.append(det)
                        for r_det in to_remove:
                            if r_det in detections_to_draw:
                                detections_to_draw.remove(r_det)
                                
                elif grounded and not evidence_match:
                    # Suspicious: grounding returned a box, but no evidence source mentions it
                    # We do not draw a box to avoid wrongly annotating objects (e.g. labeling bottle as screwdriver)
                    log_output.append(f"⚠️  SUSPICIOUS [{target}]: Grounding found a box, but no perception source mentions this object. Possible hallucination!")
                else:
                    # Not found at all
                    log_output.append(f"❌ FAILED [{target}]: Not found in the camera feed. This step is UNSAFE!")

            # ==========================
            # 4. COLLISION WARNING (Task 4)
            # ==========================
            if collision_warning_checkbox and step_arm_targets and verified_bboxes:
                for step_num in sorted(step_arm_targets.keys()):
                    left_target = step_arm_targets[step_num].get("LEFT ARM")
                    right_target = step_arm_targets[step_num].get("RIGHT ARM")
                    if not left_target or not right_target:
                        continue

                    lt = left_target.lower().strip()
                    rt = right_target.lower().strip()
                    if lt == rt:
                        # Both arms on same object: cooperative grasp, not a collision warning
                        continue

                    left_bbox = verified_bboxes.get(lt)
                    right_bbox = verified_bboxes.get(rt)
                    if not left_bbox or not right_bbox:
                        continue

                    if check_collision_risk(left_bbox, right_bbox):
                        log_output.append("\n--- TASK 4: COLLISION WARNING ---")
                        log_output.append(
                            f"⚠️  COLLISION RISK at Step {step_num}: "
                            f"LEFT ARM target '{left_target}' overlaps RIGHT ARM target '{right_target}'."
                        )
                        log_output.append("Execution halted for safety (collision warning).")
                        return annotated_image, "\n".join(log_output)

        # ==========================
        # 5. SUCCESS DETECTION (Task 3)
        # ==========================
        if success_detection_checkbox:
            log_output.append("\n--- TASK 3: SUCCESS DETECTION ---")
            if after_image is None:
                log_output.append("No AFTER image provided — upload an after-state image to verify success.")
            else:
                after_rgb = after_image.convert("RGB")
                verdict = planner.verify_task_success(image, after_rgb, task_text)
                log_output.append(verdict)
            
    else:
        log_output.append("\nNo specific bimanual task requested.")

    # Draw any remaining/non-suppressed standard detections
    if detections_to_draw:
        annotated_image = draw_detections(annotated_image, detections_to_draw)

    return annotated_image, "\n".join(log_output)

# ---------------------------------------------------------------------------
# Gradio UI Design
# ---------------------------------------------------------------------------
with gr.Blocks() as demo:
    gr.Markdown(
        """
        # 🤖 XLeRobot VLM Bimanual Pipeline
        Upload a camera feed image and test the AI reasoning stack for your Final Year Project!
        """
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(type="pil", label="Robot Camera Feed")
            after_image_input = gr.Image(type="pil", label="After Image (Task 3: Success Detection)", value=None)
            
            # Automatically load all images from test_images/
            test_img_paths = glob.glob("test_images/*.*")
            if test_img_paths:
                gr.Examples(
                    examples=test_img_paths,
                    inputs=image_input,
                    label="Or select an existing test image:"
                )
            
            with gr.Group():
                gr.Markdown("### Task & Grounding")
                task_input = gr.Textbox(label="Bimanual Task (SmolVLM2)", placeholder="e.g. Hold the cup and pour water...")
                ground_input = gr.Textbox(label="Ground Phrase (Florence-2)", placeholder="e.g. coffee cup")
                segment_input = gr.Textbox(label="Segment Phrase (Florence-2)", placeholder="e.g. wooden block")
            
            with gr.Group():
                gr.Markdown("### Extra Perception")
                ocr_checkbox = gr.Checkbox(label="Run OCR (Read Text)")
                proposals_checkbox = gr.Checkbox(label="Run Region Proposals (Blind Grasping)")
                segment_all_checkbox = gr.Checkbox(label="Pixel-Level Segmentation (All Objects)", value=False)
                grasp_overlay_checkbox = gr.Checkbox(label="Show Bimanual Grasp Points (L/R Overlay)", value=True)
                success_detection_checkbox = gr.Checkbox(label="Task 3: Verify Success (Before vs After)", value=False)
                collision_warning_checkbox = gr.Checkbox(label="Task 4: Collision Warning (Safety)", value=True)
            
            with gr.Group():
                gr.Markdown("### Ask the AI (VQA)")
                question_input = gr.Textbox(label="General Question (SmolVLM2)", placeholder="e.g. Is the honey jar open or closed?")
            
            with gr.Row():
                submit_btn = gr.Button("Generate AI Plan", variant="primary")
                cancel_btn = gr.Button("Cancel / Stop", variant="stop")
            
        with gr.Column(scale=1):
            image_output = gr.Image(label="Annotated Output (Perception)")
            text_output = gr.Textbox(label="AI Reasoning Logs & Plan", lines=20)
            
    # Connect UI to Python backend
    generate_event = submit_btn.click(
        fn=run_pipeline,
        inputs=[
            image_input, 
            after_image_input,
            task_input, 
            ground_input, 
            segment_input, 
            ocr_checkbox, 
            proposals_checkbox,
            question_input,
            segment_all_checkbox,
            grasp_overlay_checkbox,
            success_detection_checkbox,
            collision_warning_checkbox
        ],
        outputs=[image_output, text_output]
    )
    
    # Allow user to cleanly cancel the ongoing generation
    cancel_btn.click(fn=None, inputs=None, outputs=None, cancels=[generate_event])

if __name__ == "__main__":
    print("Starting Web UI on http://127.0.0.1:7860 ...")
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=gr.themes.Monochrome())
