#!/usr/bin/env python3
"""
XLeRobot VLM Exploration Pipeline
==================================
Combines Florence-2 (visual perception) and SmolVLM2 (bimanual planning)
to explore zero-shot manipulation understanding for the XLeRobot FYP.

Usage
-----
  # From a local image file
  python run_pipeline.py --image path/to/image.jpg \\
                         --task "Pick up the bottle and rotate it 90 degrees clockwise"

  # From a URL (downloads automatically)
  python run_pipeline.py --image https://example.com/scene.jpg \\
                         --task "Move the cup to the left side of the table"

  # Save the annotated output image
  python run_pipeline.py --image scene.jpg --task "..." --save output/result.jpg

  # Use Florence-2-large for better grounding accuracy
  python run_pipeline.py --image scene.jpg --task "..." --florence-size large

  # Use smaller SmolVLM2 (500M) for faster inference
  python run_pipeline.py --image scene.jpg --task "..." --smolvlm-size 500m

  # Run only Florence-2 (skip SmolVLM2 planning)
  python run_pipeline.py --image scene.jpg --perception-only

  # Webcam: capture a frame and run the pipeline
  python run_pipeline.py --webcam --task "Grab the object in front"
"""

import argparse
import sys
import os
from pathlib import Path

# Allow running from the fyp/ directory directly
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.utils import load_image, print_banner, print_detections, print_grounding, print_plan, check_gpu
from pipeline.florence_perception import FlorencePerception
from pipeline.smolvlm_planner import SmolVLMPlanner
from pipeline.visualizer import (
    draw_detections,
    draw_grounding,
    draw_dense_captions,
    draw_segmentation,
    save_or_show,
    composite_results,
)


# -----------------------------------------------------------------------
# Webcam capture
# -----------------------------------------------------------------------

def capture_webcam_frame(camera_index: int = 0):
    """Capture a single frame from the webcam and return as PIL Image."""
    try:
        import cv2
        from PIL import Image
        import numpy as np

        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print(f"  ✗ Could not open camera index {camera_index}")
            sys.exit(1)

        print(f"  Capturing from camera {camera_index}... (press any key in the preview window)")
        ret, frame = cap.read()
        cap.release()

        if not ret:
            print("  ✗ Failed to capture frame")
            sys.exit(1)

        # BGR → RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame_rgb)
        print(f"  Captured frame: {image.width}×{image.height}")
        return image

    except ImportError:
        print("  ✗ OpenCV not installed. Run: pip install opencv-python")
        sys.exit(1)


# -----------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------

def run_pipeline(args):
    print_banner("XLeRobot VLM Exploration Pipeline")
    print()

    # --- GPU check ---
    print_banner("System Info")
    device = "cuda" if check_gpu() else "cpu"

    # --- Load image ---
    print_banner("Loading Image")
    if args.webcam:
        image = capture_webcam_frame(args.camera_index)
    else:
        print(f"  Source: {args.image}")
        image = load_image(args.image)
        print(f"  Size  : {image.width}×{image.height} px")

    # Save a copy of the original for compositing later
    original = image.copy()

    # ================================================================
    # STAGE 1 — Florence-2 Perception
    # ================================================================
    florence_model_id = {
        "base":  "microsoft/Florence-2-base",
        "large": "microsoft/Florence-2-large",
    }.get(args.florence_size, "microsoft/Florence-2-base")

    print_banner(f"Florence-2 Perception  [{args.florence_size}]")
    florence = FlorencePerception(model_id=florence_model_id, device=device)

    # 1a. Object detection
    print("\n  [1/4] Object Detection")
    detections = florence.detect_objects(image)
    print_detections(detections)
    annotated_od = draw_detections(image, detections)

    # 1b. Image caption (brief overview)
    print("\n  [2/4] Scene Caption")
    caption = florence.caption(image, detail="detailed")
    print(f"  → {caption}")

    # 1c. Phrase grounding (if a task/phrase is given)
    annotated_grounding = image.copy()
    if args.ground_phrase or args.task:
        phrase = args.ground_phrase
        if not phrase and args.task:
            # Auto-extract: use the first noun phrase from the task as grounding target
            # Simple heuristic: take words after "the" or "a"
            words = args.task.lower().split()
            phrase = None
            for i, w in enumerate(words):
                if w in ("the", "a", "an") and i + 1 < len(words):
                    phrase = " ".join(words[i:i+3]).rstrip(".,")
                    break
            if not phrase:
                phrase = args.task.split()[0]  # fallback: first word

        print(f"\n  [3/4] Phrase Grounding  → '{phrase}'")
        grounding = florence.ground_phrase(image, phrase)
        print_grounding(grounding, phrase)
        annotated_grounding = draw_grounding(image, grounding, phrase)
    else:
        print("\n  [3/4] Phrase Grounding  (skipped — no --task or --ground-phrase given)")
        grounding = None

    # 1d. Dense region captions
    print("\n  [4/4] Dense Region Captions")
    dense = florence.dense_captions(image)
    if dense:
        labels = dense.get("labels", [])
        print(f"  {len(labels)} region(s) captioned:")
        for label in labels[:6]:  # show first 6
            print(f"    • {label}")
        if len(labels) > 6:
            print(f"    ... (+{len(labels)-6} more)")

    # 1e. Object Segmentation (if requested)
    annotated_segmentation = None
    if args.segment_phrase:
        print(f"\n  [5/5] Object Segmentation  → '{args.segment_phrase}'")
        seg = florence.segment_object(image, args.segment_phrase)
        if seg and seg.get("polygons"):
            # polygons shape is [[[x1, y1, x2, y2, ...]], ...]
            # count total parts
            parts = sum(len(p) for p in seg.get("polygons", []))
            print(f"  Segmented '{args.segment_phrase}' → found {parts} object part(s)")
            annotated_segmentation = draw_segmentation(image, seg, color="#4ECDC4", alpha=140)
        else:
            print(f"  (could not segment phrase: '{args.segment_phrase}')")
    else:
        print("\n  [5/7] Object Segmentation  (skipped — no --segment-phrase given)")

    # 1f. OCR (Read Text)
    if args.ocr:
        print("\n  [6/7] OCR (Text Reading)")
        ocr_result = florence.read_text(image)
        if ocr_result and ocr_result.get("bboxes"):
            print_detections([{"label": l, "bbox": b} for l, b in zip(ocr_result["labels"], ocr_result["bboxes"])])
            # Draw OCR just like detections
            annotated_od = draw_detections(annotated_od, [{"label": l, "bbox": b} for l, b in zip(ocr_result["labels"], ocr_result["bboxes"])])
        else:
            print("  (no text found)")
    else:
        print("\n  [6/7] OCR (Text Reading)  (skipped — no --ocr given)")

    # 1g. Region Proposals
    if args.proposals:
        print("\n  [7/7] Region Proposals (Blind Grasping)")
        props = florence.region_proposals(image)
        if props and props.get("bboxes"):
            # Region proposals often have empty strings for labels
            labels = props.get("labels", [""] * len(props["bboxes"]))
            print_detections([{"label": f"Proposal {i+1}", "bbox": b} for i, b in enumerate(props["bboxes"])])
            annotated_od = draw_detections(annotated_od, [{"label": "", "bbox": b} for b in props["bboxes"]])
        else:
            print("  (no proposals found)")
    else:
        print("\n  [7/7] Region Proposals  (skipped — no --proposals given)")

    # ================================================================
    # STAGE 2 — SmolVLM2 Planning
    # ================================================================
    if not args.perception_only:
        smolvlm_model_id = {
            "2b":   "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
            "500m": "HuggingFaceTB/SmolVLM2-500M-Instruct",
        }.get(args.smolvlm_size, "HuggingFaceTB/SmolVLM2-2.2B-Instruct")

        print_banner(f"SmolVLM2 Planning  [{args.smolvlm_size}]")
        planner = SmolVLMPlanner(model_id=smolvlm_model_id, device=device)

        # 2a. Scene description
        print("\n  [1/3] Scene Description")
        scene_desc = planner.describe_scene(image)
        import textwrap
        for line in scene_desc.splitlines():
            print(textwrap.fill(line, width=70, initial_indent="    ", subsequent_indent="    "))

        # 2b. Bimanual action plan (if task given)
        if args.task:
            print(f"\n  [2/3] Bimanual Action Plan")
            print(f"  Task: \"{args.task}\"")
            print()
            plan = planner.plan_bimanual_action(image, args.task, scene_desc=scene_desc)
            print_plan(plan)

            # 2c. Grasp point suggestions
            if args.ground_phrase:
                print(f"\n  [3/3] Grasp Point Suggestions  → '{args.ground_phrase}'")
                grasp_sug = planner.suggest_grasp_points(image, args.ground_phrase)
                for line in grasp_sug.splitlines():
                    print(textwrap.fill(line, width=70, initial_indent="    ", subsequent_indent="    "))
        else:
            print("\n  (No --task given — skipping action plan and grasp suggestions)")
            print("  Tip: run with --task \"Pick up the bottle\" to get a bimanual plan")
    else:
        print_banner("SmolVLM2 Planning  [SKIPPED]")
        print("  (--perception-only flag set)")

    # ================================================================
    # STAGE 3 — Save / Display Results
    # ================================================================
    print_banner("Saving Results")

    if args.save:
        save_path = Path(args.save)
        
        # Clear the output directory if requested
        if args.clear_output and save_path.parent.exists() and save_path.parent != Path("."):
            import shutil
            shutil.rmtree(save_path.parent)
            print(f"  [Cleaned] Removed old files from {save_path.parent}/")
            
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save individual outputs
        annotated_od_path = save_path.parent / (save_path.stem + "_detections" + save_path.suffix)
        annotated_grounding_path = save_path.parent / (save_path.stem + "_grounding" + save_path.suffix)
        composite_path = save_path

        save_or_show(annotated_od, str(annotated_od_path))
        save_or_show(annotated_grounding, str(annotated_grounding_path))
        
        if annotated_segmentation is not None:
            annotated_seg_path = save_path.parent / (save_path.stem + "_segmentation" + save_path.suffix)
            save_or_show(annotated_segmentation, str(annotated_seg_path))

        # Composite: original | detections | grounding side by side
        comp = composite_results(original, annotated_od, annotated_grounding)
        save_or_show(comp, str(composite_path))
        print(f"\n  All outputs saved to: {save_path.parent}/")
    else:
        print("  Displaying annotated detection image...")
        print("  (Use --save path/to/output.jpg to save instead)")
        annotated_od.show()

    print_banner("Done ✓")


# -----------------------------------------------------------------------
# CLI argument parsing
# -----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="XLeRobot VLM Exploration Pipeline: Florence-2 + SmolVLM2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Image source (mutually exclusive)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--image", "-i",
        type=str,
        help="Path or URL to input image",
    )
    source.add_argument(
        "--webcam", "-w",
        action="store_true",
        help="Capture a frame from webcam",
    )

    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Webcam camera index (default: 0)",
    )

    # Task
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=None,
        help='Manipulation task in natural language, e.g. "Pick up the bottle"',
    )
    parser.add_argument(
        "--ground-phrase", "-g",
        type=str,
        default=None,
        help='Specific phrase to ground with Florence-2, e.g. "the red bottle"',
    )
    parser.add_argument(
        "--segment-phrase", "-seg",
        type=str,
        default=None,
        help='Phrase to segment at pixel-level with Florence-2, e.g. "the red mug"',
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Run OCR to read all text in the image and locate it",
    )
    parser.add_argument(
        "--proposals",
        action="store_true",
        help="Run dense region proposals (finds all graspable objects without text prompts)",
    )

    # Model sizes
    parser.add_argument(
        "--florence-size",
        choices=["base", "large"],
        default="large",
        help="Florence-2 model size (default: large)",
    )
    parser.add_argument(
        "--smolvlm-size",
        choices=["2b", "500m"],
        default="2b",
        help="SmolVLM2 model size (default: 2b)",
    )

    # Flags
    parser.add_argument(
        "--perception-only",
        action="store_true",
        help="Run only Florence-2 (skip SmolVLM2 planning)",
    )
    parser.add_argument(
        "--clear-output", "-c",
        action="store_true",
        help="Delete all old files in the save directory before saving new ones",
    )
    parser.add_argument(
        "--save", "-s",
        type=str,
        default=None,
        help="Path to save the annotated output image (e.g. output/result.jpg)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(args)
