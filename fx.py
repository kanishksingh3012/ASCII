"""
Post-processing visual effects for the lo-fi / Dreamcore aesthetic.

All functions take and return RGB uint8 arrays so they can be stacked
freely in any order.
"""

from __future__ import annotations

import numpy as np
import cv2


def apply_scanlines(
    img: np.ndarray, intensity: float = 0.35, period: int = 2
) -> np.ndarray:
    """
    Darken every `period`-th horizontal row to simulate CRT scanlines.

    intensity 0 → no effect; 1 → scanline rows become black.
    """
    if intensity <= 0:
        return img
    mask = np.ones(img.shape[0], dtype=np.float32)
    mask[::period] = max(0.0, 1.0 - float(intensity))
    out = img.astype(np.float32) * mask[:, None, None]
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_noise(
    img: np.ndarray, amount: float = 0.08, monochrome: bool = True
) -> np.ndarray:
    """
    Add Gaussian noise for a grainy, lo-fi look.

    `amount` is the noise std-dev in normalized [0, 1] space; 0.05–0.15 is
    a tasteful range. `monochrome=True` applies one noise channel across
    RGB, which preserves the source palette better than per-channel noise.
    """
    if amount <= 0:
        return img
    rng = np.random.default_rng()
    sigma = float(amount) * 255.0
    if monochrome:
        n = rng.normal(0, sigma, img.shape[:2]).astype(np.float32)[..., None]
    else:
        n = rng.normal(0, sigma, img.shape).astype(np.float32)
    out = img.astype(np.float32) + n
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_glow(img: np.ndarray, strength: float = 0.3, sigma: float = 2.0) -> np.ndarray:
    """
    Soft phosphor glow: blend a Gaussian-blurred copy over the original.

    Evokes the bloom of an old CRT and pairs well with Matrix / Amber themes.
    """
    if strength <= 0:
        return img
    blur = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    out = img.astype(np.float32) + blur.astype(np.float32) * float(strength)
    return np.clip(out, 0, 255).astype(np.uint8)


def apply_vignette(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    """Optional circular vignette to focus the eye on the center."""
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2.0, w / 2.0
    dist = np.sqrt(((xx - cx) / cx) ** 2 + ((yy - cy) / cy) ** 2)
    mask = np.clip(1.0 - dist * float(strength), 0.0, 1.0)
    out = img.astype(np.float32) * mask[..., None]
    return np.clip(out, 0, 255).astype(np.uint8)
