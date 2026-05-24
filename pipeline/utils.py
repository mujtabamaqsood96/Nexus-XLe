"""
Shared Utilities
----------------
Image loading and pretty-printing helpers shared across the pipeline.
"""

from PIL import Image
import requests
import sys
import textwrap
import json


def load_image(source: str | Image.Image) -> Image.Image:
    """
    Load an image from a file path, URL, or pass through a PIL Image.

    Parameters
    ----------
    source : str (file path or http/https URL) or PIL.Image.Image

    Returns
    -------
    PIL Image in RGB mode
    """
    if isinstance(source, Image.Image):
        return source.convert("RGB")
    if isinstance(source, str) and source.startswith("http"):
        response = requests.get(source, stream=True, timeout=10)
        response.raise_for_status()
        return Image.open(response.raw).convert("RGB")
    return Image.open(source).convert("RGB")


def print_banner(title: str, width: int = 62):
    """Print a section banner."""
    print()
    print("╔" + "═" * (width - 2) + "╗")
    pad = (width - 2 - len(title)) // 2
    print("║" + " " * pad + title + " " * (width - 2 - pad - len(title)) + "║")
    print("╚" + "═" * (width - 2) + "╝")


def print_detections(detections: list[dict]):
    """Pretty-print Florence-2 object detections."""
    if not detections:
        print("  (no objects detected)")
        return
    print(f"  Detected {len(detections)} object(s):")
    for d in detections:
        bbox = [f"{x:.1f}" for x in d["bbox"]]
        print(f"    • {d['label']:<22}  bbox: [{', '.join(bbox)}]")


def print_grounding(grounding_result: dict | None, phrase: str):
    """Pretty-print Florence-2 phrase grounding result."""
    if grounding_result is None:
        print(f"  (could not ground phrase: '{phrase}')")
        return
    bboxes = grounding_result.get("bboxes", [])
    print(f"  Grounded '{phrase}' → {len(bboxes)} region(s):")
    for bbox in bboxes:
        coords = [f"{x:.1f}" for x in bbox]
        print(f"    • [{', '.join(coords)}]")


def print_plan(plan_text: str, indent: int = 2):
    """Pretty-print a SmolVLM2 text bimanual plan."""
    prefix = " " * indent
    
    # Example of how easily we can parse the text back into a dictionary!
    # Even though it's text, it's highly structured.
    for line in plan_text.strip().splitlines():
        wrapped = textwrap.fill(line, width=70, subsequent_indent=prefix + "  ")
        print(prefix + wrapped)


def check_gpu():
    """Print GPU status and return whether CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram_total = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_free = (
                torch.cuda.get_device_properties(0).total_memory
                - torch.cuda.memory_allocated(0)
            ) / 1e9
            print(f"  GPU : {gpu_name}")
            print(f"  VRAM: {vram_free:.1f} GB free / {vram_total:.1f} GB total")
            return True
        else:
            print("  ⚠ CUDA not available — running on CPU (will be slow)")
            return False
    except ImportError:
        print("  ⚠ PyTorch not found")
        return False
