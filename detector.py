"""
detector.py — adapter layer

Purpose:
  - Provide the API expected by `app.py`: `detect(image, query, confidence, model_choice)`
  - Internally call the detection core in `decoder.py`
  - Convert raw detector boxes into `supervision.Detections` so `render.draw()`
    can render ellipses + labels.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import supervision as sv

import decoder


def _boxes_to_supervision_detections(boxes: List[list]) -> sv.Detections:
    """
    Convert boxes from `decoder.detect()` into sv.Detections.

    Expected box format:
      [x1, y1, x2, y2, score, class_name]
    """
    if not boxes:
        return sv.Detections.empty()

    xyxy = np.array([b[:4] for b in boxes], dtype=np.float32)
    conf = np.array([b[4] for b in boxes], dtype=np.float32)
    class_names = [b[5] for b in boxes]

    # Deterministic IDs for clustering in render.py (stable across runs).
    unique_names = sorted(set(class_names))
    name_to_id = {name: i for i, name in enumerate(unique_names)}
    class_id = np.array([name_to_id[n] for n in class_names], dtype=int)

    return sv.Detections(
        xyxy=xyxy,
        confidence=conf,
        class_id=class_id,
        data={"class_name": np.array(class_names)},
    )


def detect(
    image,
    query: str = "person",
    confidence: float = 0.25,
    model_choice: Optional[str] = None,
) -> sv.Detections:
    """
    Main function used by `app.py`.

    Notes:
      - For now we only support the detector core's single pathway (YOLO-World
        with a dynamic text query), plus `decoder.STUB_MODE` for stubs.
      - `model_choice` is kept for future branching (zero-shot vs fine-tuned).
    """
    # Keep the team-friendly stub behavior available, but default to
    # real zero-shot inference.
    if model_choice is not None and "stub" in model_choice.lower():
        decoder.STUB_MODE = True
    else:
        decoder.STUB_MODE = False

    raw_boxes = decoder.detect(image, text_query=query, conf_threshold=confidence)
    return _boxes_to_supervision_detections(raw_boxes)

