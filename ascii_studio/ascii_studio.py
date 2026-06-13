"""
ASCII Art Studio — Reflex port.

The UI is fully custom (assets/styles.css); the conversion pipeline reuses the
original NumPy backend modules unchanged (ascii_engine, renderer, fx, themes,
utils, video_processor). Run:  reflex run
"""

from __future__ import annotations

import base64
import os
import tempfile
import time

import numpy as np
import reflex as rx
from PIL import Image

from ascii_engine import (
    DEFAULT_CHARSET,
    adjust_tonemap,
    ascii_to_text,
    image_to_ascii,
)
from fx import apply_glow, apply_noise, apply_scanlines
from renderer import AsciiRenderer
from themes import THEMES
from utils import (
    MAX_IMAGE_MB,
    MAX_VIDEO_MB,
    hex_to_rgb,
    pil_to_bytes,
    read_image_bytes,
    rgb_to_hex,
)
from video_processor import process_video

CREATOR_HANDLE = "kanishk.io"
CREATOR_INSTAGRAM_URL = "https://instagram.com/kanishk.io"
THEME_NAMES = list(THEMES.keys())

# ---------------------------------------------------------------------------
# Renderer cache (renderers are expensive; key by charset + font size)
# ---------------------------------------------------------------------------
_RENDERER_CACHE: dict[tuple[str, int], AsciiRenderer] = {}


def get_renderer(charset: str, font_size: int) -> AsciiRenderer:
    key = (charset, font_size)
    r = _RENDERER_CACHE.get(key)
    if r is None:
        if len(_RENDERER_CACHE) > 8:
            _RENDERER_CACHE.clear()
        r = AsciiRenderer(charset=charset, font_size=font_size)
        _RENDERER_CACHE[key] = r
    return r


def _png_data_uri(arr: np.ndarray) -> str:
    img = Image.fromarray(arr.astype(np.uint8))
    return "data:image/png;base64," + base64.b64encode(pil_to_bytes(img, "PNG")).decode()


def _sample_image() -> np.ndarray:
    """Synthetic gradient + rings, identical in spirit to the old sample."""
    yy, xx = np.meshgrid(np.arange(360), np.arange(640), indexing="ij")
    r = ((xx * 255) // 640).astype(np.uint8)
    g = ((yy * 255) // 360).astype(np.uint8)
    b = (180 - ((xx + yy) % 180)).astype(np.uint8)
    return np.dstack([r, g, b])


def _convert(src_rgb, p) -> tuple[np.ndarray, np.ndarray]:
    """Full pipeline. `p` is a plain dict of the current settings."""
    theme = THEMES[p["theme_name"]]
    cs = p["charset"] if len(p["charset"]) >= 2 else DEFAULT_CHARSET
    bg = hex_to_rgb(p["bg_hex"])
    fg = hex_to_rgb(p["fg_hex"]) if theme["mode"] == "mono" else None

    adjusted = adjust_tonemap(src_rgb, p["brightness"], p["contrast"], p["gamma"])
    char_grid, color_grid = image_to_ascii(
        adjusted,
        output_width=p["output_width"],
        charset=cs,
        aspect_ratio_correction=p["aspect_corr"],
        invert=p["invert_lum"],
    )
    img = get_renderer(cs, p["font_size"]).render(
        char_grid,
        color_grid if theme["mode"] == "color" else None,
        bg,
        fg,
        mode=theme["mode"],
    )
    arr = np.array(img)
    if p["fx_glow"] > 0:
        arr = apply_glow(arr, strength=p["fx_glow"])
    if p["fx_scanlines"] > 0:
        arr = apply_scanlines(arr, intensity=p["fx_scanlines"])
    if p["fx_noise"] > 0:
        arr = apply_noise(arr, amount=p["fx_noise"])
    return arr, char_grid


# ===========================================================================
# State
# ===========================================================================
class State(rx.State):
    # --- controls ---
    output_width: int = 160
    aspect_corr: float = 0.50
    brightness: float = 0.0
    contrast: float = 1.10
    gamma: float = 1.0
    theme_name: str = "Monochrome Dark"
    bg_hex: str = "#0a0a0a"
    fg_hex: str = "#e6e6e6"
    invert_lum: bool = False
    charset: str = DEFAULT_CHARSET
    fx_scanlines: float = 0.0
    fx_noise: float = 0.0
    fx_glow: float = 0.0
    font_size: int = 12

    # --- ui state ---
    mobile_menu_open: bool = False

    # --- image state ---
    use_sample: bool = True
    source_uri: str = ""
    output_uri: str = ""
    grid_label: str = ""
    font_label: str = ""
    error: str = ""
    sample_note: str = "Showing a generated sample. Upload an image for real results."

    # --- video state ---
    max_seconds: int = 10
    video_status: str = ""
    video_uri: str = ""
    _video_raw: bytes = b""
    _video_name: str = ""

    # backend-only payloads for downloads / re-render
    _uploaded_bytes: bytes = b""
    _txt: str = ""
    _png_bytes: bytes = b""

    @rx.var
    def is_mono(self) -> bool:
        return THEMES[self.theme_name]["mode"] == "mono"

    @rx.var
    def theme_desc(self) -> str:
        return THEMES[self.theme_name].get("description", "")

    @rx.var
    def width_chars_label(self) -> str:
        return f"{self.output_width}"

    @rx.var
    def output_width_pct(self) -> str:
        """Map the char-width slider (60..320) to a visible on-screen width
        (40%..100%) so the control has an obvious effect on the preview."""
        frac = (self.output_width - 60) / (320 - 60)
        frac = max(0.0, min(1.0, frac))
        return f"{60 + frac * 40:.0f}%"

    # --------------------------- core render -----------------------------
    def _params(self) -> dict:
        return dict(
            output_width=self.output_width,
            aspect_corr=self.aspect_corr,
            brightness=self.brightness,
            contrast=self.contrast,
            gamma=self.gamma,
            theme_name=self.theme_name,
            charset=self.charset,
            bg_hex=self.bg_hex,
            fg_hex=self.fg_hex,
            invert_lum=self.invert_lum,
            fx_scanlines=self.fx_scanlines,
            fx_noise=self.fx_noise,
            fx_glow=self.fx_glow,
            font_size=self.font_size,
        )

    def _apply(self):
        self.error = ""
        try:
            if self._uploaded_bytes:
                src = read_image_bytes(self._uploaded_bytes)
            else:
                src = _sample_image()
        except ValueError as e:
            self.error = str(e)
            return
        try:
            arr, char_grid = _convert(src, self._params())
        except Exception as e:  # noqa: BLE001 — surface any pipeline error calmly
            self.error = str(e)
            return
        self.source_uri = _png_data_uri(src)
        self.output_uri = _png_data_uri(arr)
        h, w = char_grid.shape
        self.grid_label = f"grid {w} × {h} · {self.theme_name.lower()}"
        self.font_label = f"font {self.font_size}px"
        self._txt = ascii_to_text(char_grid)
        self._png_bytes = pil_to_bytes(Image.fromarray(arr), "PNG")

    def on_load(self):
        self._apply()

    # --------------------------- setters ---------------------------------
    def set_output_width(self, v):
        self.output_width = int(float(v))
        self._apply()

    def set_aspect(self, v):
        self.aspect_corr = round(float(v), 2)
        self._apply()

    def set_brightness(self, v):
        self.brightness = round(float(v), 2)
        self._apply()

    def set_contrast(self, v):
        self.contrast = round(float(v), 2)
        self._apply()

    def set_gamma(self, v):
        self.gamma = round(float(v), 2)
        self._apply()

    def set_theme(self, name: str):
        self.theme_name = name
        t = THEMES[name]
        self.bg_hex = rgb_to_hex(t["bg"])
        self.fg_hex = rgb_to_hex(t["fg"]) if t.get("fg") else "#000000"
        self.invert_lum = bool(t.get("invert_luminance", False))
        self._apply()

    def set_bg(self, v: str):
        self.bg_hex = v
        self._apply()

    def set_fg(self, v: str):
        self.fg_hex = v
        self._apply()

    def toggle_menu(self):
        self.mobile_menu_open = not self.mobile_menu_open

    def close_menu(self):
        self.mobile_menu_open = False

    def set_charset(self, v: str):
        self.charset = v
        self._apply()

    def set_scan(self, v):
        self.fx_scanlines = round(float(v), 2)
        self._apply()

    def set_noise(self, v):
        self.fx_noise = round(float(v), 3)
        self._apply()

    def set_glow(self, v):
        self.fx_glow = round(float(v), 2)
        self._apply()

    def set_font_size(self, v):
        self.font_size = int(float(v))
        self._apply()

    def set_use_sample(self, v: bool):
        self.use_sample = bool(v)
        if self.use_sample:
            self._uploaded_bytes = b""
            self.sample_note = "Showing a generated sample. Upload an image for real results."
        self._apply()

    def set_max_seconds(self, v):
        self.max_seconds = int(float(v))

    # --------------------------- uploads ---------------------------------
    async def handle_image_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        data = await files[0].read()
        if len(data) > MAX_IMAGE_MB * 1024 * 1024:
            self.error = f"Image too large (>{MAX_IMAGE_MB} MB)."
            return
        self._uploaded_bytes = data
        self.use_sample = False
        self.sample_note = "Live preview from your upload."
        self._apply()

    async def handle_video_upload(self, files: list[rx.UploadFile]):
        if not files:
            return
        data = await files[0].read()
        if len(data) > MAX_VIDEO_MB * 1024 * 1024:
            self.video_status = f"Video too large (keep under {MAX_VIDEO_MB} MB)."
            return
        self._video_raw = data
        self._video_name = files[0].filename or "clip.mp4"
        self.video_uri = ""
        self.video_status = f"Loaded {self._video_name}. Press Process to render."

    # --------------------------- downloads -------------------------------
    def download_txt(self):
        if self._txt:
            return rx.download(data=self._txt, filename="ascii_art.txt")

    def download_png(self):
        if self._png_bytes:
            return rx.download(data=self._png_bytes, filename="ascii_art.png")

    def download_png_2x(self):
        # Re-render at 2× font size for a crisp print export.
        try:
            src = read_image_bytes(self._uploaded_bytes) if self._uploaded_bytes else _sample_image()
            p = self._params()
            p["font_size"] = self.font_size * 2
            arr, _ = _convert(src, p)
            return rx.download(
                data=pil_to_bytes(Image.fromarray(arr), "PNG"),
                filename="ascii_art_2x.png",
            )
        except Exception as e:  # noqa: BLE001
            self.error = str(e)

    # --------------------------- video render ----------------------------
    @rx.event(background=True)
    async def process_video(self):
        async with self:
            if not self._video_raw:
                self.video_status = "Upload a video first."
                return
            self.video_status = "Rendering ASCII frames… this can take a while."
            self.video_uri = ""
            raw, name = self._video_raw, self._video_name
            params = self._params()
            max_seconds = self.max_seconds

        suffix = os.path.splitext(name)[1] or ".mp4"
        in_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        in_tmp.write(raw)
        in_tmp.close()
        out_path = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4").name

        try:
            import cv2

            probe = cv2.VideoCapture(in_tmp.name)
            src_fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
            total = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            probe.release()
            max_frames = (
                min(total, int(max_seconds * src_fps))
                if total
                else int(max_seconds * src_fps)
            )

            def frame_fn(rgb):
                arr, _ = _convert(rgb, params)
                return arr

            stats = process_video(
                input_path=in_tmp.name,
                output_path=out_path,
                frame_fn=frame_fn,
                progress_cb=None,
                max_frames=max_frames,
            )
            with open(out_path, "rb") as f:
                out_bytes = f.read()
        except Exception as e:  # noqa: BLE001
            async with self:
                self.video_status = f"Video processing failed: {e}"
            return
        finally:
            for pth in (in_tmp.name, out_path):
                try:
                    os.unlink(pth)
                except OSError:
                    pass

        uri = "data:video/mp4;base64," + base64.b64encode(out_bytes).decode()
        async with self:
            self.video_uri = uri
            self.video_status = (
                f"Rendered {stats['frames']} frames @ {stats['fps']:.1f} fps → "
                f"{stats['size'][0]}×{stats['size'][1]} px."
            )


# ===========================================================================
# UI components
# ===========================================================================
def slider_row(name, value_label, min_, max_, step, value, on_change, ticks=None):
    children = [
        rx.box(
            rx.el.span(name, class_name="name"),
            rx.el.span(value_label, class_name="val"),
            class_name="as-row",
        ),
        rx.el.input(
            type="range",
            min=min_,
            max=max_,
            step=step,
            default_value=value.to_string(),
            on_change=on_change,
        ),
    ]
    if ticks is not None:
        children.append(
            rx.box(
                rx.el.span(ticks[0]),
                rx.el.span(ticks[1]),
                class_name="as-ticks",
            )
        )
    return rx.box(*children, margin_bottom="1rem")


def section(idx, label):
    return rx.box(
        rx.el.span(idx, class_name="idx"),
        rx.el.span(label),
        class_name="as-section-label",
    )


def sidebar():
    return rx.box(
        rx.box(
            rx.box(
                rx.el.span("Controls", class_name="t"),
                rx.el.span("06 modules", class_name="m"),
                class_name="as-controls-head",
            ),
            rx.el.hr(class_name="as-divider"),
            # 01 CONVERSION
            section("01", "Conversion"),
            slider_row("Output width", State.width_chars_label, 60, 320, 10,
                       State.output_width, State.set_output_width, ("60", "320 chars")),
            slider_row("Aspect-ratio correction", State.aspect_corr.to_string(), 0.30, 0.90, 0.05,
                       State.aspect_corr, State.set_aspect, ("0.30", "0.90")),
            # 02 TUNING
            section("02", "Tuning"),
            slider_row("Brightness", State.brightness.to_string(), -0.50, 0.50, 0.05,
                       State.brightness, State.set_brightness),
            slider_row("Contrast", State.contrast.to_string(), 0.50, 2.00, 0.05,
                       State.contrast, State.set_contrast),
            slider_row("Gamma", State.gamma.to_string(), 0.40, 2.50, 0.05,
                       State.gamma, State.set_gamma),
            # 03 THEME
            section("03", "Theme"),
            rx.box(rx.el.span("Color engine", class_name="name"), class_name="as-row"),
            rx.el.select(
                *[rx.el.option(n, value=n) for n in THEME_NAMES],
                value=State.theme_name,
                on_change=State.set_theme,
                class_name="as-select",
            ),
            rx.el.p(State.theme_desc, class_name="as-help"),
            rx.cond(
                State.is_mono,
                rx.box(
                    rx.box(
                        rx.box(
                            rx.el.span("Background", class_name="name"),
                            rx.el.input(type="color", value=State.bg_hex,
                                        on_change=State.set_bg),
                        ),
                        rx.box(
                            rx.el.span("Foreground", class_name="name"),
                            rx.el.input(type="color", value=State.fg_hex,
                                        on_change=State.set_fg),
                        ),
                        class_name="as-fields",
                    ),
                    margin_top="0.6rem",
                ),
            ),
            # 04 CHARACTER RAMP
            section("04", "Character ramp"),
            rx.box(rx.el.span("Dense → sparse", class_name="name"), class_name="as-row"),
            rx.el.textarea(
                default_value=State.charset,
                on_blur=State.set_charset,
                class_name="as-textarea",
                spell_check=False,
            ),
            # 05 VISUAL FX
            section("05", "Visual FX"),
            slider_row("CRT scanlines", State.fx_scanlines.to_string(), 0.0, 0.60, 0.05,
                       State.fx_scanlines, State.set_scan),
            slider_row("Grain / noise", State.fx_noise.to_string(), 0.0, 0.20, 0.01,
                       State.fx_noise, State.set_noise),
            slider_row("Phosphor glow", State.fx_glow.to_string(), 0.0, 0.80, 0.05,
                       State.fx_glow, State.set_glow),
            # 06 RENDER
            section("06", "Render"),
            slider_row("Font size", State.font_size.to_string() + "px", 8, 22, 1,
                       State.font_size, State.set_font_size),
            class_name="as-card",
        ),
        class_name="as-sidebar",
        custom_attrs={"data-open": State.mobile_menu_open},
    )


def panel_head(idx, title, right):
    return rx.box(
        rx.box(
            rx.el.span(idx, class_name="idx"),
            rx.el.span(title),
            class_name="as-panel-title",
        ),
        right,
        class_name="as-panel-head",
    )


def image_tab():
    return rx.box(
        rx.box(
            # SOURCE
            rx.box(
                panel_head("01", "Source",
                           rx.el.span("JPG · PNG · WEBP · BMP", class_name="as-panel-meta")),
                rx.upload(
                    rx.box(
                        rx.el.div("⬆", class_name="ico"),
                        rx.el.div("Drop an image, or click to browse", class_name="main"),
                        rx.el.div(
                            f"{MAX_IMAGE_MB} MB max · longest side auto-scaled to 4096 px",
                            class_name="sub",
                        ),
                    ),
                    id="img_upload",
                    accept={"image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"]},
                    multiple=False,
                    on_drop=State.handle_image_upload(
                        rx.upload_files(upload_id="img_upload")
                    ),
                    class_name="as-drop",
                ),
                rx.box(
                    rx.el.span("Use sample image", class_name="name"),
                    rx.switch(checked=State.use_sample, on_change=State.set_use_sample),
                    class_name="as-switch",
                    margin_top="0.9rem",
                ),
                rx.el.p(State.sample_note, class_name="as-help"),
                rx.cond(
                    State.source_uri != "",
                    rx.box(
                        rx.el.img(src=State.source_uri, class_name="as-img"),
                        rx.el.div("— source preview —", class_name="as-cap"),
                        margin_top="0.9rem",
                    ),
                ),
                class_name="as-card",
            ),
            # ASCII OUTPUT
            rx.box(
                panel_head("02", "ASCII output",
                           rx.box(rx.el.span(class_name="dot"), rx.el.span("Live"),
                                  class_name="as-live")),
                rx.cond(
                    State.error != "",
                    rx.el.p(State.error, class_name="as-error"),
                    rx.cond(
                        State.output_uri != "",
                        rx.box(
                            rx.box(
                                rx.el.img(
                                    src=State.output_uri,
                                    class_name="as-out-img",
                                    style={"width": State.output_width_pct},
                                ),
                                class_name="as-out-wrap",
                            ),
                            rx.box(
                                rx.el.span(State.grid_label),
                                rx.el.span(State.font_label, class_name="as-panel-meta"),
                                display="flex", justify_content="space-between",
                                margin_top="0.6rem", color="var(--dim)", font_size="0.74rem",
                            ),
                        ),
                    ),
                ),
                rx.el.div("Export", class_name="as-section-label", margin_top="1.1rem"),
                rx.box(
                    rx.el.button(".txt", on_click=State.download_txt, class_name="as-btn"),
                    rx.el.button(".png", on_click=State.download_png, class_name="as-btn"),
                    rx.el.button("2× .png", on_click=State.download_png_2x, class_name="as-btn"),
                    class_name="as-btn-row",
                ),
                class_name="as-card",
            ),
            class_name="as-grid2",
        ),
    )


def video_tab():
    return rx.box(
        rx.box(
            panel_head("01", "Source",
                       rx.el.span("MP4 · MOV · WEBM · MKV · AVI", class_name="as-panel-meta")),
            rx.upload(
                rx.box(
                    rx.el.div("⬆", class_name="ico"),
                    rx.el.div("Drop a video, or click to browse", class_name="main"),
                    rx.el.div(
                        f"{MAX_VIDEO_MB} MB max · short clips (under 15s) render fastest",
                        class_name="sub",
                    ),
                ),
                id="vid_upload",
                accept={"video/*": [".mp4", ".mov", ".webm", ".mkv", ".avi"]},
                multiple=False,
                on_drop=State.handle_video_upload(rx.upload_files(upload_id="vid_upload")),
                class_name="as-drop",
            ),
            slider_row("Max duration (s)", State.max_seconds.to_string(), 2, 60, 1,
                       State.max_seconds, State.set_max_seconds),
            rx.el.button("▶  Process video", on_click=State.process_video,
                         class_name="as-btn as-btn-primary"),
            rx.cond(
                State.video_status != "",
                rx.el.p(State.video_status, class_name="as-help"),
            ),
            rx.cond(
                State.video_uri != "",
                rx.box(
                    rx.el.video(src=State.video_uri, controls=True, class_name="as-img"),
                    rx.el.a(
                        rx.el.button("Download .mp4", class_name="as-btn"),
                        href=State.video_uri,
                        download="ascii_art.mp4",
                        margin_top="0.7rem",
                        display="block",
                    ),
                    margin_top="1rem",
                ),
            ),
            class_name="as-card",
        ),
    )


def about_tab():
    return rx.box(
        rx.box(
            rx.el.div(
                rx.el.h4("How it works"),
                rx.el.p(
                    "Every pixel's luminance is mapped to a character from a "
                    "customizable ramp ordered by visual density. Contrast, gamma, "
                    "and brightness are applied first; then the image is downscaled "
                    "(with aspect-ratio correction so output isn't stretched); finally "
                    "the character grid is rendered by compositing pre-rasterized glyph "
                    "masks. All steps are vectorized NumPy — which is why video works "
                    "at interactive speeds."
                ),
                rx.el.h4("Color engines"),
                rx.el.p(
                    "Monochrome — pick any background/foreground pair (Matrix, Amber, "
                    "Synthwave presets). True Color — each glyph inherits its source "
                    "pixel's RGB."
                ),
                rx.el.h4("Visual FX"),
                rx.el.p("Stack CRT scanlines, grain, and phosphor glow for lo-fi looks."),
                rx.el.h4("Limits"),
                rx.el.p(
                    f"Images: {MAX_IMAGE_MB} MB, longest side auto-scaled to 4096 px. "
                    f"Video: {MAX_VIDEO_MB} MB, user-selectable duration cap."
                ),
                class_name="as-about",
            ),
            class_name="as-card",
        ),
    )


def tabs():
    return rx.tabs.root(
        rx.tabs.list(
            rx.tabs.trigger(rx.el.span("Image"), value="image", class_name="as-tab"),
            rx.tabs.trigger(rx.el.span("Video"), value="video", class_name="as-tab"),
            rx.tabs.trigger(rx.el.span("About"), value="about", class_name="as-tab"),
            class_name="as-tabs",
        ),
        rx.tabs.content(image_tab(), value="image"),
        rx.tabs.content(video_tab(), value="video"),
        rx.tabs.content(about_tab(), value="about"),
        default_value="image",
    )


def topbar():
    return rx.box(
        rx.box(
            rx.el.button("≡", on_click=State.toggle_menu, class_name="as-burger",
                         custom_attrs={"aria-label": "Toggle controls"}),
            rx.el.span("ASCII.STUDIO"),
            rx.el.span(class_name="cursor"),
            class_name="as-wordmark",
        ),
        rx.el.span(
            "$ built by ",
            rx.el.a(f"@{CREATOR_HANDLE}", href=CREATOR_INSTAGRAM_URL, target="_blank"),
            class_name="as-byline",
        ),
        class_name="as-topbar",
    )


def index():
    return rx.box(
        topbar(),
        rx.box(
            class_name="as-backdrop",
            custom_attrs={"data-open": State.mobile_menu_open},
            on_click=State.close_menu,
        ),
        rx.box(
            sidebar(),
            rx.box(
                rx.el.p(
                    "Turn images and video into luminous ASCII — monochrome, "
                    "true-color, or retro-themed. Vectorized, fast, ready to export.",
                    class_name="as-sub",
                ),
                rx.box(
                    rx.el.span(rx.el.span("●", class_name="d"), "Vectorized NumPy", class_name="as-badge"),
                    rx.el.span(rx.el.span("●", class_name="d"), "True Color", class_name="as-badge"),
                    rx.el.span(rx.el.span("●", class_name="d"), "CRT FX", class_name="as-badge"),
                    rx.el.span(rx.el.span("●", class_name="d"), "MP4 Export", class_name="as-badge"),
                    class_name="as-badges",
                ),
                tabs(),
                class_name="as-main",
            ),
            class_name="as-shell",
        ),
        rx.el.div("$ ready  ·  ascii.studio", class_name="as-footer"),
        class_name="as-app",
        on_mount=State.on_load,
    )


app = rx.App(
    stylesheets=["/styles.css"],
    theme=rx.theme(appearance="dark", accent_color="gray", has_background=False),
)
app.add_page(index, route="/", title="ASCII Studio")
