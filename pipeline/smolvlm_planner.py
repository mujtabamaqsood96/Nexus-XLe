"""
SmolVLM2 Planner Module
-----------------------
Wraps HuggingFaceTB/SmolVLM2-Instruct for:
  - Scene description (what objects are present and where)
  - Bimanual action planning (step-by-step LEFT ARM / RIGHT ARM instructions)

The planner is prompt-engineered specifically for bimanual robot manipulation
on the XLeRobot platform (two SO-101 arms).
"""

import os
import re
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# -----------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------

_BIMANUAL_CONTEXT = """You control a robot with LEFT ARM and RIGHT ARM. Both arms can reach any object.

Output format:
SCENE: <describe objects and positions>
ANALYSIS: <explain what movement axes (X/Y/Z) and rotations are needed>
PLAN:
Step 1 | LEFT ARM: <action> [TARGET: <object>]
Step 1 | RIGHT ARM: <action> [TARGET: <object>]
Step 2 | BOTH ARMS: <action> [TARGET: <object>]

EXAMPLE (task: "lift the green cylinder with both arms"):
SCENE: A green cylinder is in the center of the table.
ANALYSIS: The green cylinder must be lifted. Both arms translate along the Z-axis to lift it. No rotation needed.
PLAN:
Step 1 | LEFT ARM: Grasp left side of green cylinder [TARGET: green cylinder]
Step 1 | RIGHT ARM: Grasp right side of green cylinder [TARGET: green cylinder]
Step 2 | BOTH ARMS: Lift green cylinder upward along Z-axis [TARGET: green cylinder]

IMPORTANT: When the task says "both arms", BOTH arms must act on the object together in Step 1. Keep to 2-3 steps."""

_SCENE_PROMPT = (
    "Describe all visible objects in this scene. "
    "For each object, state: what it is, its approximate position (left/center/right, near/far), "
    "its orientation, and any relevant properties (size, color, shape). "
    "Be precise — this information will be used to plan robot manipulation."
)


# -----------------------------------------------------------------------
# Post-processing: enforce bimanual structure on raw VLM output
# -----------------------------------------------------------------------

def _extract_target_object(task: str) -> str:
    """Extract the likely target object noun phrase from the task string."""
    # Try common patterns: "pick up the X", "lift the X", "grab the X", etc.
    patterns = [
        r'(?:pick up|pick|lift|grab|grasp|hold|move|rotate|pour)\s+(?:the\s+)?(.+?)(?:\s+(?:with|using|from|to|onto|into|off)\b)',
        r'(?:pick up|pick|lift|grab|grasp|hold|move|rotate|pour)\s+(?:the\s+)?(.+?)$',
    ]
    for pattern in patterns:
        match = re.search(pattern, task.lower())
        if match:
            return match.group(1).strip()
    return None


def _postprocess_bimanual_plan(raw_plan: str, task: str) -> str:
    """
    Post-process the VLM's raw output to enforce bimanual structure.
    
    If the task requests both arms but the model only used one arm,
    restructure the plan to use both arms cooperatively.
    """
    task_lower = task.lower()
    needs_bimanual = bool(re.search(
        r'both\b.*?\b(?:arms?|hands?)|bimanual|two\s+(?:arms?|hands?)|simultaneously|together',
        task_lower
    ))
    
    if not needs_bimanual:
        return raw_plan
    
    # Check if the plan already uses both arms
    has_left = bool(re.search(r'Step\s+\d+\s*\|\s*LEFT ARM:', raw_plan, re.IGNORECASE))
    has_right = bool(re.search(r'Step\s+\d+\s*\|\s*RIGHT ARM:', raw_plan, re.IGNORECASE))
    has_both = bool(re.search(r'Step\s+\d+\s*\|\s*BOTH ARMS:', raw_plan, re.IGNORECASE))
    
    if (has_left and has_right) or has_both:
        # Already bimanual — no fix needed
        return raw_plan
    
    # --- Restructure into bimanual plan ---
    target = _extract_target_object(task)
    if not target:
        return raw_plan  # Can't determine target, return as-is
    
    # Extract SCENE and ANALYSIS blocks from raw output
    scene_line = ""
    analysis_line = ""
    scene_match = re.search(r'SCENE:\s*(.+?)(?:\n|$)', raw_plan)
    analysis_match = re.search(r'ANALYSIS:\s*(.+?)(?:\n(?:PLAN|Step)|\Z)', raw_plan, re.DOTALL)
    
    if scene_match:
        scene_line = scene_match.group(1).strip()
    if analysis_match:
        analysis_line = analysis_match.group(1).strip()
    
    # Enhance analysis to mention bimanual coordination
    if analysis_line and "both arms" not in analysis_line.lower():
        analysis_line += " Both arms coordinate to grasp and lift the object along the Z-axis."
    
    # Build the corrected bimanual plan
    corrected = f"SCENE: {scene_line}\n"
    corrected += f"ANALYSIS: {analysis_line}\n"
    corrected += "PLAN:\n"
    corrected += f"Step 1 | LEFT ARM: Grasp left side of {target} [TARGET: {target}]\n"
    corrected += f"Step 1 | RIGHT ARM: Grasp right side of {target} [TARGET: {target}]\n"
    corrected += f"Step 2 | BOTH ARMS: Lift {target} upward along Z-axis [TARGET: {target}]"
    
    return corrected


class SmolVLMPlanner:
    """
    Wrapper around SmolVLM2-Instruct for scene understanding and bimanual planning.

    Parameters
    ----------
    model_id : str
        HuggingFace model ID. Options:
        - "HuggingFaceTB/SmolVLM2-Instruct"       (~2.2B params, default)
        - "HuggingFaceTB/SmolVLM2-500M-Instruct"  (~500M params, faster/smaller)
    device : str
        "cuda" or "cpu". Uses device_map="auto" for automatic placement.
    """

    def __init__(
        self,
        model_id: str = "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
        device: str = "cuda",
    ):
        self.device = device
        self.model_id = model_id

        # Read HF token from environment (required for gated SmolVLM2 repo)
        token = os.environ.get("HF_TOKEN", None)
        if token:
            print(f"[SmolVLM2] Using HF_TOKEN from environment.")
        else:
            print("[SmolVLM2] ⚠ No HF_TOKEN found in environment. Set it with: export HF_TOKEN=your_token")

        print(f"[SmolVLM2] Loading {model_id}...")
        self.processor = AutoProcessor.from_pretrained(model_id, token=token)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            token=token,
        )
        print("[SmolVLM2] Ready.")

    # ------------------------------------------------------------------
    # Internal: generate a response from image + messages
    # ------------------------------------------------------------------

    def _generate(
        self,
        image: Image.Image,
        user_text: str,
        system_text: str = None,
        max_new_tokens: int = 512,
        repetition_penalty: float = 1.2,
    ) -> str:
        """Build the chat prompt, run inference, and return the assistant reply."""
        # Inject system context directly into the user message because
        # SmolVLM2 often drops or ignores the 'system' role in its chat template.
        combined_text = user_text
        if system_text:
            combined_text = f"{system_text}\n\n---\n\nUSER REQUEST:\n{user_text}"
            
        content = [{"type": "image"}, {"type": "text", "text": combined_text}]

        messages = [{"role": "user", "content": content}]

        prompt = self.processor.apply_chat_template(
            messages, add_generation_prompt=True
        )

        inputs = self.processor(
            text=prompt, images=[image], return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=repetition_penalty,
            )

        decoded = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Extract only the assistant's reply (try multiple markers)
        for marker in ["Assistant:", "assistant:", "ASSISTANT:"]:
            if marker in decoded:
                return decoded.split(marker)[-1].strip()
        
        # Fallback: if the prompt text is in the output, strip it
        if combined_text[:50] in decoded:
            return decoded[decoded.index(combined_text) + len(combined_text):].strip()
        
        return decoded.strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def describe_scene(self, image: Image.Image) -> str:
        """
        Generate a detailed spatial description of the scene.

        Returns
        -------
        str — description of objects, positions, and properties
        """
        return self._generate(image, _SCENE_PROMPT, max_new_tokens=300)

    def plan_bimanual_action(
        self, image: Image.Image, task: str, scene_desc: str = None
    ) -> str:
        """
        Generate a bimanual manipulation plan for the given task.

        The raw VLM output is post-processed to enforce bimanual structure
        when the task explicitly requests both arms.

        Parameters
        ----------
        task : str
            Natural language task description, e.g.:
            "Pick up the bottle with both arms and rotate it 90 degrees clockwise"
        scene_desc : str, optional
            Visual scene description to ground planning and reduce hallucination

        Returns
        -------
        str — formatted step-by-step plan with LEFT ARM / RIGHT ARM / BOTH ARMS steps
        """
        user_text = ""
        if scene_desc:
            user_text += f"Scene Description: {scene_desc}\n\n"
        user_text += (
            f"Task: {task}\n\n"
            "Generate a bimanual manipulation plan. "
            "You MUST output SCENE, then ANALYSIS, then PLAN. "
            "Both arms must work together. Keep to 2-3 steps."
        )
        raw_plan = self._generate(
            image,
            user_text,
            system_text=_BIMANUAL_CONTEXT,
            max_new_tokens=400,
            repetition_penalty=1.05,
        )
        return _postprocess_bimanual_plan(raw_plan, task)

    def answer_question(self, image: Image.Image, question: str) -> str:
        """
        General-purpose visual question answering.

        Parameters
        ----------
        question : str  e.g. "What is the orientation of the bottle?"

        Returns
        -------
        str — model's answer
        """
        return self._generate(image, question, max_new_tokens=200)

    def suggest_grasp_points(self, image: Image.Image, object_name: str) -> str:
        """
        Ask the model to suggest where each arm should grasp a specific object.

        Parameters
        ----------
        object_name : str  e.g. "the cylindrical bottle"

        Returns
        -------
        str — description of optimal grasp points for left and right arm
        """
        user_text = (
            f"Looking at '{object_name}' in this image, where should a bimanual robot "
            f"grasp it? Describe the exact grasp point for the LEFT ARM and the RIGHT ARM "
            f"in terms of position on the object (e.g., left side, right side, top, bottom, "
            f"which face, approximate pixel location). Consider the object's orientation."
        )
        return self._generate(image, user_text, max_new_tokens=250)

    def unload(self):
        """Free GPU memory."""
        del self.model
        torch.cuda.empty_cache()
        print("[SmolVLM2] Unloaded.")
