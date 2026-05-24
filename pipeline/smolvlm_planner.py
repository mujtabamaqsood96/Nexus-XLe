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
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# -----------------------------------------------------------------------
# Prompt templates
# -----------------------------------------------------------------------

_BIMANUAL_CONTEXT = """You are a robotics AI assistant helping plan bimanual manipulation tasks for the XLeRobot.

The XLeRobot has:
- LEFT ARM: a 6-DoF SO-101 robot arm on the left side
- RIGHT ARM: a 6-DoF SO-101 arm on the right side
- Both arms can grasp, rotate, translate, and release objects

When planning, always:
1. Identify the target object(s) and their spatial positions
2. Decide which arm handles which side
3. Coordinate both arms for tasks requiring two hands
4. Output concrete physical actions (grasp, release, rotate X degrees, move left/right/up/down by roughly Y cm)

Format your plan EXACTLY as:
SCENE: <one sentence describing the relevant objects and their positions>

PLAN:
Step 1 | LEFT ARM: <action> [TARGET: <noun>]
Step 1 | RIGHT ARM: <action> [TARGET: <noun>]
Step 2 | BOTH ARMS: <action> [TARGET: <noun>]
...

EXAMPLE:
SCENE: A red mug is on the left, a wooden block is on the right.
PLAN:
Step 1 | LEFT ARM: Grasp the red mug [TARGET: red mug]
Step 1 | RIGHT ARM: Grasp the wooden block [TARGET: wooden block]
Step 2 | BOTH ARMS: Wait [TARGET: none]

If an action does not interact with a physical object (like "wait"), use [TARGET: none].
Keep steps concrete and physically achievable."""

_SCENE_PROMPT = (
    "Describe all visible objects in this scene. "
    "For each object, state: what it is, its approximate position (left/center/right, near/far), "
    "its orientation, and any relevant properties (size, color, shape). "
    "Be precise — this information will be used to plan robot manipulation."
)


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
    ) -> str:
        """Build the chat prompt, run inference, and return the assistant reply."""
        content = [{"type": "image"}, {"type": "text", "text": user_text}]

        # Prepend system context into the user message if model doesn't support
        # system role natively (SmolVLM2 supports it via chat template)
        messages = []
        if system_text:
            messages.append({"role": "system", "content": system_text})
        messages.append({"role": "user", "content": content})

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
            )

        decoded = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        # Extract only the assistant's reply
        if "Assistant:" in decoded:
            return decoded.split("Assistant:")[-1].strip()
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

    def plan_bimanual_action(self, image: Image.Image, task: str) -> str:
        """
        Generate a bimanual manipulation plan for the given task.

        Parameters
        ----------
        task : str
            Natural language task description, e.g.:
            "Pick up the bottle with both arms and rotate it 90 degrees clockwise"

        Returns
        -------
        str — formatted step-by-step plan with LEFT ARM / RIGHT ARM / BOTH ARMS steps
        """
        user_text = (
            f"Task: {task}\n\n"
            "Generate a detailed bimanual manipulation plan to complete this task. "
            "Use the exact SCENE and PLAN format described in your instructions."
        )
        return self._generate(
            image,
            user_text,
            system_text=_BIMANUAL_CONTEXT,
            max_new_tokens=512,
        )

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
