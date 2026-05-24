"""
Florence-2 Perception Module
----------------------------
Wraps microsoft/Florence-2-base for visual grounding tasks:
  - Object detection
  - Phrase grounding (text → bounding box)
  - Dense region captions
  - Image captioning (brief / detailed / more detailed)

All tasks return parsed, structured Python dicts/lists ready to use.
"""

import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM


class FlorencePerception:
    """
    Wrapper around Florence-2 for visual perception tasks.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Options:
        - "microsoft/Florence-2-base"   (~230M params, faster)
        - "microsoft/Florence-2-large"  (~770M params, better accuracy)
    device : str
        "cuda" or "cpu"
    """

    def __init__(
        self,
        model_id: str = "microsoft/Florence-2-base",
        device: str = "cuda",
    ):
        self.device = device
        self.model_id = model_id

        print(f"[Florence-2] Loading {model_id} on {device}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True,
            attn_implementation="eager",
        ).to(device)
        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        print("[Florence-2] Ready.")

    # ------------------------------------------------------------------
    # Internal: run any Florence-2 task
    # ------------------------------------------------------------------

    def _run_task(self, image: Image.Image, task_prompt: str, text_input: str = None):
        """Run a Florence-2 task and return the post-processed result dict."""
        prompt = task_prompt if text_input is None else task_prompt + text_input

        inputs = self.processor(text=prompt, images=image, return_tensors="pt")

        # Move all tensors to device with correct dtypes
        model_inputs = {}
        for k, v in inputs.items():
            if hasattr(v, "to"):
                if k == "pixel_values":
                    model_inputs[k] = v.to(self.device, dtype=torch.float16)
                else:
                    model_inputs[k] = v.to(self.device)
            else:
                model_inputs[k] = v

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=1024,
                num_beams=1,       # greedy — avoids beam_search KV-cache incompatibility
                do_sample=False,
                use_cache=False,   # bypasses the past_key_values crash in newer transformers
            )

        generated_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=False
        )[0]

        parsed = self.processor.post_process_generation(
            generated_text,
            task=task_prompt,
            image_size=(image.width, image.height),
        )
        return parsed

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_objects(self, image: Image.Image) -> list[dict]:
        """
        Detect all objects in the image.

        Returns
        -------
        list of {"label": str, "bbox": [x1, y1, x2, y2]}
        """
        result = self._run_task(image, "<OD>")
        detections = []
        od = result.get("<OD>", {})
        labels = od.get("labels", [])
        bboxes = od.get("bboxes", [])
        for label, bbox in zip(labels, bboxes):
            detections.append({"label": label, "bbox": list(bbox)})
        return detections

    def ground_phrase(self, image: Image.Image, phrase: str) -> dict | None:
        """
        Find a specific object described by text.

        Parameters
        ----------
        phrase : str  e.g. "the red bottle" or "left arm grasping point"

        Returns
        -------
        {"labels": [...], "bboxes": [[x1,y1,x2,y2], ...]} or None
        """
        result = self._run_task(image, "<CAPTION_TO_PHRASE_GROUNDING>", phrase)
        return result.get("<CAPTION_TO_PHRASE_GROUNDING>")

    def segment_object(self, image: Image.Image, phrase: str) -> dict | None:
        """
        Get pixel-level segmentation polygons for a specific object.

        Parameters
        ----------
        phrase : str  e.g. "the red mug"

        Returns
        -------
        {"labels": [...], "polygons": [[[x1,y1, x2,y2, ...], ...], ...]}
        """
        result = self._run_task(image, "<REFERRING_EXPRESSION_SEGMENTATION>", phrase)
        return result.get("<REFERRING_EXPRESSION_SEGMENTATION>")

    def dense_captions(self, image: Image.Image) -> dict | None:
        """
        Generate captions for dense regions in the image.

        Returns
        -------
        {"labels": [...], "bboxes": [...]}  — one caption per region
        """
        result = self._run_task(image, "<DENSE_REGION_CAPTION>")
        return result.get("<DENSE_REGION_CAPTION>")

    def caption(self, image: Image.Image, detail: str = "detailed") -> str:
        """
        Generate an image caption.

        Parameters
        ----------
        detail : "brief" | "detailed" | "more"
        """
        task_map = {
            "brief": "<CAPTION>",
            "detailed": "<DETAILED_CAPTION>",
            "more": "<MORE_DETAILED_CAPTION>",
        }
        task = task_map.get(detail, "<DETAILED_CAPTION>")
        result = self._run_task(image, task)
        return result.get(task, "")

    def region_proposals(self, image: Image.Image) -> dict | None:
        """
        Get candidate regions (no labels) for potential grasp points.

        Returns
        -------
        {"bboxes": [[x1,y1,x2,y2], ...], "labels": ["", "", ...]}
        """
        result = self._run_task(image, "<REGION_PROPOSAL>")
        return result.get("<REGION_PROPOSAL>")

    def read_text(self, image: Image.Image) -> dict | None:
        """
        Extract text from the image along with bounding boxes (OCR).

        Returns
        -------
        {"labels": ["text1", ...], "bboxes": [[x1, y1, x2, y2], ...]}
        """
        result = self._run_task(image, "<OCR_WITH_REGION>")
        ocr = result.get("<OCR_WITH_REGION>")
        if not ocr:
            return None
        
        # Florence-2 OCR outputs 'quad_boxes' (8 coords: x1,y1,x2,y2,x3,y3,x4,y4)
        if "quad_boxes" in ocr and "labels" in ocr:
            bboxes = []
            for qb in ocr["quad_boxes"]:
                # Convert polygon to bounding box
                xs = qb[0::2]
                ys = qb[1::2]
                bboxes.append([min(xs), min(ys), max(xs), max(ys)])
            return {"labels": ocr["labels"], "bboxes": bboxes}
            
        return ocr

    def unload(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
        print("[Florence-2] Unloaded.")
