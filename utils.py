"""Small shared helpers: safe image decoding, color parsing, PIL-to-bytes."""

from __future__ import annotations

import io
from typing import Tuple

import cv2
import numpy as np
from PIL import Image


MAX_IMAGE_DIM = 4096          # longest-side cap for uploaded stills
MAX_IMAGE_MB = 15             # reject larger uploads outright
MAX_VIDEO_MB = 30             # matches .streamlit/config.toml


def read_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw bytes to an RGB uint8 NumPy array, with guardrails."""
    if not data:
        raise ValueError("Empty upload.")
    arr = np.frombuffer(data, dtype=np.uint8)
    img_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise ValueError("Unsupported or corrupt image. Try JPG, PNG, or WEBP.")

    h, w = img_bgr.shape[:2]
    if max(h, w) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / float(max(h, w))
        img_bgr = cv2.resize(
            img_bgr,
            (max(1, int(w * scale)), max(1, int(h * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def pil_to_bytes(img: Image.Image, fmt: str = "PNG") -> bytes:
    """Serialize a PIL image to an in-memory byte string."""
    buf = io.BytesIO()
    save_kwargs = {}
    if fmt.upper() == "PNG":
        save_kwargs["optimize"] = True
    elif fmt.upper() in ("JPG", "JPEG"):
        save_kwargs.update(quality=92, optimize=True)
    img.save(buf, format=fmt, **save_kwargs)
    return buf.getvalue()


def hex_to_rgb(h: str) -> Tuple[int, int, int]:
    """'#rrggbb' (or 'rrggbb') → (r, g, b)."""
    h = h.lstrip("#")
    if len(h) != 6:
        raise ValueError(f"Expected 6-digit hex color, got {h!r}")
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))  # type: ignore[return-value]


def rgb_to_hex(rgb: Tuple[int, int, int]) -> str:
    return "#%02x%02x%02x" % tuple(int(c) for c in rgb)
