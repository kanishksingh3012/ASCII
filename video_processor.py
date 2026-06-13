"""
Video pipeline.

Streams frames from an input clip through a user-supplied `frame_fn`
(image → image) and writes the result out as MP4 via OpenCV's
VideoWriter. The pipeline is streaming — it never holds the whole clip
in memory — so long clips only cost time, not RAM.
"""

from __future__ import annotations

from typing import Callable, Optional

import cv2
import numpy as np


def process_video(
    input_path: str,
    output_path: str,
    frame_fn: Callable[[np.ndarray], np.ndarray],
    progress_cb: Optional[Callable[[float], None]] = None,
    max_frames: Optional[int] = None,
    fps_override: Optional[float] = None,
) -> dict:
    """
    Read `input_path`, apply `frame_fn` to every RGB frame, write `output_path`.

    Parameters
    ----------
    input_path : source video
    output_path : destination .mp4
    frame_fn : callable (rgb_frame) -> rgb_frame of consistent shape
    progress_cb : optional callable(fraction_complete in [0, 1])
    max_frames : cap the number of processed frames
    fps_override : force a specific output frame rate; otherwise source FPS

    Returns
    -------
    dict with keys: frames (int), fps (float), size (w, h)
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError("Could not open video file (corrupt or unsupported codec).")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames and total_frames:
        total_frames = min(total_frames, max_frames)
    elif max_frames:
        total_frames = max_frames

    fps = float(fps_override or src_fps)

    # Read first frame to discover output dimensions
    ok, frame_bgr = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Video appears to be empty.")

    first_out = frame_fn(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    out_h, out_w = first_out.shape[:2]

    # H.264 / libx264 require even dimensions. Trim by one pixel if needed.
    out_w -= out_w % 2
    out_h -= out_h % 2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError("Could not open output video writer.")

    def _write_rgb(rgb: np.ndarray) -> None:
        if rgb.shape[0] != out_h or rgb.shape[1] != out_w:
            rgb = cv2.resize(rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)
        writer.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

    _write_rgb(first_out)
    n_written = 1

    try:
        while True:
            if max_frames and n_written >= max_frames:
                break
            ok, frame_bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            _write_rgb(frame_fn(rgb))
            n_written += 1
            if progress_cb and total_frames:
                progress_cb(min(1.0, n_written / total_frames))
    finally:
        cap.release()
        writer.release()

    return {"frames": n_written, "fps": fps, "size": (out_w, out_h)}
