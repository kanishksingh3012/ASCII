"""
Vectorized image → ASCII conversion engine.

The engine is deliberately decoupled from rendering: it turns an RGB image
into a 2-D grid of characters plus a matching grid of source-pixel colors.
The renderer consumes that grid. This separation lets us keep pixel math
fully vectorized in NumPy and reuse the same output for .txt, .png, and
.mp4 exports.
"""

from __future__ import annotations

import numpy as np
import cv2

# Paul Bourke-style luminance ramp, ordered from DENSEST glyph to SPARSEST.
# Index 0 = "brightest" glyph when drawn on a dark background (lots of ink).
DEFAULT_CHARSET = (
    "$@B%8&WM#*oahkbdpqwmZO0QLCJUYXzcvunxrjft/"
    "\\|()1{}[]?-_+~<>i!lI;:,\"^`'. "
)


def adjust_tonemap(
    img_rgb: np.ndarray,
    brightness: float,
    contrast: float,
    gamma: float,
) -> np.ndarray:
    """
    Apply brightness, contrast, and gamma to an RGB uint8 image.

    - brightness: additive in [-1, 1] normalized space (0 = no change).
    - contrast:   multiplier around mid-grey 0.5 (1.0 = no change).
    - gamma:      power curve (1.0 = no change, <1 darkens mids, >1 brightens).

    Fully vectorized — works on either a single frame or any array that
    NumPy broadcasts cleanly.
    """
    f = img_rgb.astype(np.float32, copy=False) / 255.0
    f = (f - 0.5) * contrast + 0.5          # contrast pivot at mid-grey
    f = f + brightness                       # additive brightness
    np.clip(f, 0.0, 1.0, out=f)
    if not np.isclose(gamma, 1.0):
        np.power(f, 1.0 / gamma, out=f)
    return (f * 255.0).astype(np.uint8)


def image_to_ascii(
    img_rgb: np.ndarray,
    output_width: int,
    charset: str = DEFAULT_CHARSET,
    aspect_ratio_correction: float = 0.5,
    invert: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Convert an RGB image into an ASCII character grid.

    The aspect_ratio_correction factor (~0.5 by default) compensates for the
    fact that monospace glyph cells are roughly twice as tall as they are
    wide — without it, ASCII output looks squashed vertically.

    Parameters
    ----------
    img_rgb : (H, W, 3) uint8
    output_width : target number of character columns
    charset : dense-to-sparse glyph ramp
    aspect_ratio_correction : vertical compression factor for the row count
    invert : if True, bright pixels map to sparse glyphs (good for
             light-on-dark displays inverted, or dark-on-light themes)

    Returns
    -------
    char_grid  : (h', w')   dtype '<U1'  — the ASCII characters
    color_grid : (h', w', 3) uint8       — resized RGB source (per-cell color)
    """
    if img_rgb.ndim != 3 or img_rgb.shape[2] != 3:
        raise ValueError("image_to_ascii expects an RGB image of shape (H, W, 3).")

    output_width = max(20, int(output_width))
    h, w = img_rgb.shape[:2]
    new_w = output_width
    new_h = max(1, int(h * (new_w / w) * aspect_ratio_correction))

    # INTER_AREA is the right choice when downscaling — it integrates pixels
    # rather than sampling them, so we get smoother luminance mapping.
    resized = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Rec. 601 luminance
    gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0

    chars = np.array(list(charset))
    n = len(chars)
    if n < 2:
        raise ValueError("charset must contain at least 2 characters.")

    if invert:
        # Bright pixel -> sparse glyph (high index, i.e. space-like)
        idx = gray * (n - 1)
    else:
        # Default: bright pixel -> dense glyph (low index, e.g. '$' or '@')
        idx = (1.0 - gray) * (n - 1)

    idx = np.clip(idx, 0, n - 1).astype(np.int32)
    char_grid = chars[idx]  # fancy-indexing, returns contiguous (h', w')

    return char_grid, resized


def ascii_to_text(char_grid: np.ndarray) -> str:
    """Flatten a character grid to a newline-joined plain-text string."""
    return "\n".join("".join(row) for row in char_grid)
