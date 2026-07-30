"""
video.py

Minimal video pipeline for DroneFind:
  1) read uploaded video
  2) run detector + render on sampled frames
  3) write annotated mp4 output
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

import detector
import render


def _build_writer(path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    """
    Create a writer with codec fallback.
    Prefer H.264 (`avc1`) for browser compatibility, fallback to `mp4v`.
    """
    for codec in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            path,
            cv2.VideoWriter_fourcc(*codec),
            fps,
            (width, height),
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError("Could not initialize video writer with avc1 or mp4v.")


def process_video(
    input_path: str,
    query: str,
    confidence: float,
    model_choice: str,
    sample_rate: int = 5,
) -> tuple[str | None, str]:
    """
    Process a video and return (output_path, status_message).
    """
    if not input_path:
        return None, "Upload a video first."
    if not query.strip():
        return None, "Type what to look for (e.g. 'person', 'car')."

    sample_rate = max(1, int(sample_rate))
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        return None, f"Could not open video: {input_path}"

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 24.0

    out_path = str(Path(tempfile.mkdtemp(prefix="dronefind_video_")) / "annotated.mp4")
    writer = _build_writer(out_path, fps, width, height)

    frame_idx = 0
    processed_frames = 0
    last_annotated_bgr = None

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break

        if frame_idx % sample_rate == 0:
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            detections = detector.detect(
                pil_img,
                query=query,
                confidence=confidence,
                model_choice=model_choice,
            )
            annotated_rgb = render.draw(frame_rgb, detections, conf_threshold=confidence)
            last_annotated_bgr = cv2.cvtColor(annotated_rgb, cv2.COLOR_RGB2BGR)
            writer.write(last_annotated_bgr)
            processed_frames += 1
        else:
            # Keep output duration equal to input duration.
            if last_annotated_bgr is not None:
                writer.write(last_annotated_bgr)
            else:
                writer.write(frame_bgr)

        frame_idx += 1

    cap.release()
    writer.release()

    if frame_idx == 0:
        return None, "The uploaded video had no readable frames."

    status = (
        f"Processed {frame_idx} frames (inference on every {sample_rate}th frame, "
        f"{processed_frames} inference frames)."
    )
    return out_path, status

