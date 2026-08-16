"""
detector.py — adapter layer

Provides a single `detect()` entry point for:
  - YOLO-World zero-shot detection through `decoder.py`
  - fine-tuned YOLO inference from `best.onnx`

Both paths return `supervision.Detections` so the rest of the app can render
them with the same code.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
import supervision as sv
from ultralytics import YOLO

import decoder

FINE_TUNED_MODEL_PATH = Path(__file__).with_name("best.onnx")
_fine_tuned_model = None


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


def _get_fine_tuned_model():
    """Lazy-load the exported fine-tuned ONNX model."""
    global _fine_tuned_model
    if _fine_tuned_model is None:
        if not FINE_TUNED_MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {FINE_TUNED_MODEL_PATH}. "
                "Place your exported ONNX file there and name it 'best.onnx'."
            )
        _fine_tuned_model = YOLO(str(FINE_TUNED_MODEL_PATH))
    return _fine_tuned_model


def get_fine_tuned_class_names() -> List[str]:
    """Return the list of class names the fine-tuned model was trained on.

    Used by the UI to build a dropdown so the user picks a real class name
    instead of typing a word the model may not recognise. Returns an empty
    list if the model file is missing.
    """
    try:
        model = _get_fine_tuned_model()
    except FileNotFoundError:
        return []
    names = model.names or {}
    return [str(names[k]) for k in sorted(names)]


def _ultralytics_results_to_boxes(result, class_filter=None) -> List[list]:
    """Convert an Ultralytics result object to the shared box format.

    `class_filter` may be a single class name (str) or a list of class names.
    An empty/None filter keeps every detected class.
    """
    boxes: List[list] = []

    # Normalise the filter into a set of lowercase class names.
    if isinstance(class_filter, str):
        requested = {class_filter.strip().lower()} if class_filter.strip() else set()
    elif class_filter:
        requested = {str(c).strip().lower() for c in class_filter if str(c).strip()}
    else:
        requested = set()

    if result.boxes is None:
        return boxes

    names = result.names or {}
    for box in result.boxes:
        cls_idx = int(box.cls[0])
        class_name = str(names.get(cls_idx, cls_idx))
        if requested and class_name.lower() not in requested:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        score = float(box.conf[0])
        boxes.append([x1, y1, x2, y2, score, class_name])

    return boxes


def detect(
    image,
    query: str = "person",
    confidence: float = 0.25,
    model_choice: Optional[str] = None,
) -> sv.Detections:
    """
    Main function used by `app.py`.

    Notes:
      - YOLO-World uses `query` as an open-vocabulary prompt.
      - Fine-tuned ONNX uses `query` as an optional class-name filter.
    """
    choice = (model_choice or "").lower()

    if "stub" in choice:
        decoder.STUB_MODE = True
        raw_boxes = decoder.detect(image, text_query=query or "person", conf_threshold=confidence)
        return _boxes_to_supervision_detections(raw_boxes)

    if "fine-tuned" in choice or "best.onnx" in choice or "onnx" in choice:
        decoder.STUB_MODE = False
        model = _get_fine_tuned_model()
        result = model.predict(image, conf=confidence, verbose=False)[0]
        raw_boxes = _ultralytics_results_to_boxes(result, class_filter=query)
        return _boxes_to_supervision_detections(raw_boxes)

    decoder.STUB_MODE = False
    raw_boxes = decoder.detect(image, text_query=query, conf_threshold=confidence)
    return _boxes_to_supervision_detections(raw_boxes)

