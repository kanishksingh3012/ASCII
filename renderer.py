"""
High-performance renderer: character grid → PIL image.

Strategy
--------
1. Pre-rasterize every unique glyph in the charset into a small grayscale
   mask once, at construction time.
2. At render time, build a uint32 code-point grid from the character grid
   and use it as a fancy index into the stacked mask array. This gives us
   a (H, W, ch, cw) gather in a single NumPy call — no Python loops over
   characters or pixels.
3. Reshape the gathered tiles into a full-resolution alpha mask and alpha-
   composite it over the background, either with a flat foreground color
   (mono mode) or with the per-cell source color upsampled via np.repeat
   (true color mode).

All blending happens in uint16 integer math for speed and low memory.
"""

from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont


# Candidate monospace font paths. The Streamlit Cloud / Hugging Face base
# images ship DejaVu under the usual Debian path once packages.txt installs
# `fonts-dejavu-core`. Other paths are fallbacks for local dev.
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Menlo.ttc",
    "/System/Library/Fonts/Monaco.ttf",
    "C:\\Windows\\Fonts\\consola.ttf",
    "C:\\Windows\\Fonts\\cour.ttf",
    "./fonts/DejaVuSansMono.ttf",
]


def _load_font(size: int) -> ImageFont.ImageFont:
    for path in FONT_CANDIDATES:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    # Matplotlib bundles DejaVu, so this works if matplotlib is installed.
    try:
        import matplotlib  # type: ignore

        mpl_font = os.path.join(
            matplotlib.get_data_path(), "fonts/ttf/DejaVuSansMono.ttf"
        )
        if os.path.exists(mpl_font):
            return ImageFont.truetype(mpl_font, size)
    except Exception:
        pass
    # Last resort: a tiny bitmap font. Output will be small but functional.
    return ImageFont.load_default()


class AsciiRenderer:
    """
    Pre-builds per-glyph masks for fast, vectorized grid rendering.

    A renderer is keyed by (charset, font_size). Reuse one as long as those
    settings don't change — the ctor does the expensive rasterization work.
    """

    def __init__(self, charset: str, font_size: int = 12):
        self.font = _load_font(font_size)

        # Measure advance width precisely. getlength is the modern PIL API.
        try:
            self.char_w = max(1, int(round(self.font.getlength("M"))))
        except AttributeError:
            bbox = self.font.getbbox("M")
            self.char_w = max(1, bbox[2] - bbox[0])
        ascent, descent = self.font.getmetrics()
        self.char_h = max(1, ascent + descent)

        # De-duplicate while preserving order
        unique_chars = list(dict.fromkeys(charset))

        # Build a stacked mask tensor: (N, char_h, char_w)
        masks = np.zeros((len(unique_chars), self.char_h, self.char_w), dtype=np.uint8)
        for i, ch in enumerate(unique_chars):
            tile = Image.new("L", (self.char_w, self.char_h), 0)
            ImageDraw.Draw(tile).text((0, 0), ch, font=self.font, fill=255)
            masks[i] = np.array(tile)
        self.masks = masks

        # Build a code-point lookup table so we can convert a char grid to
        # mask indices with a single vectorized gather.
        max_ord = max(ord(c) for c in unique_chars)
        self.lut = np.zeros(max_ord + 1, dtype=np.int32)
        for i, ch in enumerate(unique_chars):
            self.lut[ord(ch)] = i

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _char_grid_to_idx(self, char_grid: np.ndarray) -> np.ndarray:
        """
        Map a '<U1' character grid to mask indices via a uint32 reinterpret.

        A '<U1' array stores one Unicode code point per element as a 32-bit
        integer under the hood, so a plain `.view(np.uint32)` gives us the
        code-point grid without any Python-level iteration.
        """
        contig = np.ascontiguousarray(char_grid)
        codes = contig.view(np.uint32).reshape(contig.shape)
        # Clip in case someone hand-edits the charset at runtime; unknown
        # chars silently map to whichever glyph owns index 0.
        np.clip(codes, 0, self.lut.shape[0] - 1, out=codes)
        return self.lut[codes]

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def render(
        self,
        char_grid: np.ndarray,
        color_grid: Optional[np.ndarray],
        bg_color: Tuple[int, int, int],
        fg_color: Optional[Tuple[int, int, int]],
        mode: str = "mono",
    ) -> Image.Image:
        """
        Render a character grid to a PIL RGB image.

        mode="mono"  → every glyph is drawn in fg_color.
        mode="color" → every glyph inherits the matching cell from color_grid.
        """
        H, W = char_grid.shape
        ch_h, ch_w = self.char_h, self.char_w
        out_h, out_w = H * ch_h, W * ch_w

        idx_grid = self._char_grid_to_idx(char_grid)                 # (H, W)
        tiles = self.masks[idx_grid]                                  # (H, W, ch_h, ch_w)
        full_mask = tiles.transpose(0, 2, 1, 3).reshape(out_h, out_w) # (H*ch_h, W*ch_w)

        alpha = full_mask.astype(np.uint16)                           # 0..255
        inv_alpha = np.uint16(255) - alpha

        bg_arr = np.array(bg_color, dtype=np.uint16).reshape(1, 1, 3)

        if mode == "color":
            if color_grid is None:
                raise ValueError("True Color mode requires a color_grid.")
            # Upsample per-cell color from (H, W, 3) -> (H*ch_h, W*ch_w, 3)
            color_up = np.repeat(
                np.repeat(color_grid, ch_h, axis=0), ch_w, axis=1
            ).astype(np.uint16)
            out = (color_up * alpha[..., None] + bg_arr * inv_alpha[..., None]) // 255
        else:
            if fg_color is None:
                raise ValueError("Monochrome mode requires a fg_color.")
            fg_arr = np.array(fg_color, dtype=np.uint16).reshape(1, 1, 3)
            out = (fg_arr * alpha[..., None] + bg_arr * inv_alpha[..., None]) // 255

        return Image.fromarray(out.astype(np.uint8), mode="RGB")
