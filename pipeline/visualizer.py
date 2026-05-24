"""
Visualization Utilities
-----------------------
Draw Florence-2 detection/grounding results on PIL Images.
"""

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path


# A palette of visually distinct colors for bounding boxes
_PALETTE = [
    "#FF6B6B",  # coral red
    "#4ECDC4",  # teal
    "#45B7D1",  # sky blue
    "#96CEB4",  # sage green
    "#FFEAA7",  # light yellow
    "#DDA0DD",  # plum
    "#98D8C8",  # mint
    "#F39C12",  # orange
    "#BB8FCE",  # lavender
    "#85C1E9",  # light blue
    "#F1948A",  # salmon
    "#82E0AA",  # light green
]


def _get_font(size: int = 14):
    """Try to load a clean system font, fall back to default."""
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except Exception:
        return ImageFont.load_default()


def draw_detections(image: Image.Image, detections: list[dict]) -> Image.Image:
    """
    Draw bounding boxes and labels for Florence-2 object detections.

    Parameters
    ----------
    image      : PIL Image
    detections : list of {"label": str, "bbox": [x1, y1, x2, y2]}

    Returns
    -------
    Annotated PIL Image
    """
    img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(img)
    font = _get_font(14)

    for i, det in enumerate(detections):
        color = _PALETTE[i % len(_PALETTE)]
        bbox = det["bbox"]  # [x1, y1, x2, y2]
        label = det["label"]

        # Draw box
        draw.rectangle(bbox, outline=color, width=3)

        # Draw label background
        text_bbox = draw.textbbox((bbox[0], max(0, bbox[1] - 22)), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((bbox[0], max(0, bbox[1] - 22)), label, fill="white", font=font)

    return img


def draw_grounding(
    image: Image.Image,
    grounding_result: dict | None,
    phrase: str,
    color: str = "#FF3B30",
) -> Image.Image:
    """
    Draw bounding boxes for a phrase-grounded object.

    Parameters
    ----------
    image            : PIL Image
    grounding_result : {"labels": [...], "bboxes": [[x1,y1,x2,y2], ...]}
    phrase           : the text phrase that was grounded
    color            : box color hex string

    Returns
    -------
    Annotated PIL Image
    """
    img = image.copy().convert("RGB")
    if grounding_result is None:
        return img

    draw = ImageDraw.Draw(img)
    font = _get_font(14)

    bboxes = grounding_result.get("bboxes", [])
    for bbox in bboxes:
        draw.rectangle(bbox, outline=color, width=4)
        label = f'"{phrase}"'
        text_bbox = draw.textbbox((bbox[0], max(0, bbox[1] - 24)), label, font=font)
        draw.rectangle(
            [text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2],
            fill=color,
        )
        draw.text((bbox[0], max(0, bbox[1] - 24)), label, fill="white", font=font)

    return img


def draw_dense_captions(image: Image.Image, dense_result: dict | None) -> Image.Image:
    """
    Draw dense region captions from Florence-2 DENSE_REGION_CAPTION task.

    Parameters
    ----------
    dense_result : {"labels": [...], "bboxes": [...]}
    """
    if dense_result is None:
        return image.copy()

    detections = []
    labels = dense_result.get("labels", [])
    bboxes = dense_result.get("bboxes", [])
    for label, bbox in zip(labels, bboxes):
        # Truncate long captions for display
        short_label = label[:30] + "..." if len(label) > 30 else label
        detections.append({"label": short_label, "bbox": list(bbox)})

    return draw_detections(image, detections)


def draw_segmentation(
    image: Image.Image,
    segmentation_result: dict | None,
    color: str = "#4ECDC4",
    alpha: int = 120,
) -> Image.Image:
    """
    Draw semi-transparent polygons from Florence-2 REFERRING_EXPRESSION_SEGMENTATION.
    """
    if not segmentation_result:
        return image.copy()

    img = image.copy().convert("RGBA")
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Convert hex color to RGB tuple
    hex_color = color.lstrip("#")
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    fill_color = (*rgb, alpha)
    outline_color = (*rgb, 255)

    polygons_list = segmentation_result.get("polygons", [])
    for polygons in polygons_list:
        # polygons is a list of polygon lists for one object (some objects have multiple disconnected parts)
        for poly in polygons:
            # poly is a flat list of coordinates: [x1, y1, x2, y2, ...]
            # convert to list of tuples: [(x1, y1), (x2, y2), ...]
            points = [(poly[i], poly[i+1]) for i in range(0, len(poly), 2)]
            if len(points) >= 3:
                draw.polygon(points, fill=fill_color, outline=outline_color, width=3)

    # Blend overlay with original image
    out = Image.alpha_composite(img, overlay)
    return out.convert("RGB")

def overlay_mask(
    image: Image.Image,
    mask: np.ndarray,
    color: tuple = (255, 80, 80),
    alpha: float = 0.45,
) -> Image.Image:
    """
    Overlay a binary segmentation mask on an image with transparency.

    Parameters
    ----------
    mask  : numpy bool/uint8 array, same H×W as image
    color : RGB tuple for mask color
    alpha : opacity of the mask overlay (0=transparent, 1=opaque)
    """
    img = image.copy().convert("RGBA")
    mask_uint8 = (mask.astype(np.uint8) * 255)
    overlay = Image.new("RGBA", img.size, (*color, int(alpha * 255)))
    mask_pil = Image.fromarray(mask_uint8, mode="L")
    img.paste(overlay, mask=mask_pil)
    return img.convert("RGB")


def save_or_show(image: Image.Image, path: str = None):
    """
    Save the annotated image to disk or display it.

    Parameters
    ----------
    path : file path to save to. If None, opens image viewer.
    """
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        image.save(path)
        print(f"  Saved → {path}")
    else:
        image.show()


def composite_results(
    original: Image.Image,
    detections_img: Image.Image,
    grounding_img: Image.Image,
    gap: int = 10,
) -> Image.Image:
    """
    Stitch original + detections + grounding side by side for easy comparison.
    """
    w, h = original.size
    total_w = w * 3 + gap * 2
    composite = Image.new("RGB", (total_w, h), color=(30, 30, 30))
    composite.paste(original, (0, 0))
    composite.paste(detections_img, (w + gap, 0))
    composite.paste(grounding_img, (w * 2 + gap * 2, 0))
    return composite
