import numpy as np
import torch
from PIL import Image
from torchvision.ops import nms
from ultralytics import YOLOWorld  # or YOLO for fine-tuned models


# Global state control - Person D & B build against STUB_MODE = True initially
STUB_MODE = True

# Initialize model lazily.
MODEL_NAME = "yolov8s-world.pt"
_model = None


def _get_model():
    """Lazy-load YOLO to avoid downloading weights on import in stub mode."""
    global _model
    if _model is None:
        _model = YOLOWorld(MODEL_NAME)
    return _model


def slice_image(image, tile_size=640, overlap_ratio=0.20):
    """Slices a high-res image into overlapping tiles[cite: 1]."""
    if isinstance(image, Image.Image):
        image = np.array(image)

    height, width, _ = image.shape
    stride = int(tile_size * (1 - overlap_ratio))  # 512px step size for 20% overlap[cite: 1]

    tiles = []
    offsets = []

    # For small images, fall back to a single tile starting at (0, 0).
    if width <= tile_size:
        x_coords = [0]
    else:
        x_coords = list(range(0, width - tile_size + 1, stride))

    if height <= tile_size:
        y_coords = [0]
    else:
        y_coords = list(range(0, height - tile_size + 1, stride))

    if x_coords[-1] + tile_size < width:
        x_coords.append(width - tile_size)
    if y_coords[-1] + tile_size < height:
        y_coords.append(height - tile_size)

    for y in y_coords:
        for x in x_coords:
            tile = image[y : y + tile_size, x : x + tile_size]
            tiles.append(tile)
            offsets.append((x, y))

    return tiles, offsets


def run_model_on_tile(tile, text_query="person", conf_threshold=0.25):
    """Runs YOLO inference on a single 640x640 tile[cite: 1]."""
    # Set the text prompt class dynamically for open-vocabulary detection
    model = _get_model()
    model.set_classes([text_query])

    # Run prediction
    results = model.predict(tile, conf=conf_threshold, verbose=False)[0]

    local_boxes = []
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            score = float(box.conf[0])
            local_boxes.append([x1, y1, x2, y2, score, text_query])

    return local_boxes


def deduplicate_boxes(boxes_list, iou_threshold=0.50):
    """NMS deduplication across tile seam boundaries[cite: 1]."""
    if not boxes_list:
        return []

    boxes_tensor = torch.tensor([b[:4] for b in boxes_list], dtype=torch.float32)
    scores_tensor = torch.tensor([b[4] for b in boxes_list], dtype=torch.float32)

    keep_indices = nms(boxes_tensor, scores_tensor, iou_threshold)
    return [boxes_list[i] for i in keep_indices.tolist()]


def detect(image, text_query="person", conf_threshold=0.25):
    """Main function called by Person B (rendering) and Person D (Gradio / evaluate.py)[cite: 1]."""
    if STUB_MODE:
        # 3 mock boxes so teammates can test their rendering and UI code[cite: 1]
        return [
            [100, 150, 150, 200, 0.85, text_query],
            [300, 400, 340, 460, 0.91, text_query],
            [1000, 1200, 1050, 1280, 0.78, text_query],
        ]

    # 1. Slice original high-res image into overlapping tiles[cite: 1]
    tiles, offsets = slice_image(image, tile_size=640, overlap_ratio=0.20)

    # 2. Infer on tiles & shift coordinates to full-frame pixel space[cite: 1]
    all_raw_boxes = []
    for tile, (x_off, y_off) in zip(tiles, offsets):
        local_boxes = run_model_on_tile(tile, text_query, conf_threshold)

        for x1, y1, x2, y2, score, label in local_boxes:
            all_raw_boxes.append([x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, score, label])

    # 3. Deduplicate overlapping seam detections[cite: 1]
    final_boxes = deduplicate_boxes(all_raw_boxes, iou_threshold=0.50)
    return final_boxes


# Video processing is not wired to rendering yet in this repo, but keeping
# it here lets Person B/D test the pipeline end-to-end later.
# Imports are inside the function to avoid requiring them for image-only runs.


def process_video(
    input_path: str,
    output_path: str,
    text_query: str = "person",
    conf_threshold: float = 0.25,
    sample_rate: int = 5,
    minimum_consecutive_frames: int = 3,
):
    """
    Processes a video file by running detection, tracking (ByteTrack), and temporal filtering.

    Outputs H.264-encoded mp4 readable by Chrome/Gradio.
    """
    import cv2
    import supervision as sv

    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {input_path}")

    # Read video metadata
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0

    # Calculate effective FPS after frame sampling
    effective_fps = max(1.0, fps / sample_rate)

    # Initialize ByteTrack for persistent ID tracking across frames[cite: 1]
    tracker = sv.ByteTrack(frame_rate=int(effective_fps))

    # Force 'avc1' (H.264) codec so browser/Chrome can play it[cite: 1]
    fourcc = cv2.VideoWriter_fourcc(*"avc1")
    out = cv2.VideoWriter(output_path, fourcc, effective_fps, (width, height))

    frame_count = 0
    track_history = {}  # Tracks frame counts for temporal filtering[cite: 1]

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Step 1: Frame Sampling (Process every Nth frame to save time)[cite: 1]
        if frame_count % sample_rate == 0:
            # OpenCV provides BGR; convert to RGB for detector.py
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Step 2: Run your full Tiling Detection pipeline[cite: 1]
            boxes = detect(rgb_frame, text_query=text_query, conf_threshold=conf_threshold)

            # Step 3: Convert output boxes to Supervision Detections format
            if len(boxes) > 0:
                xyxy = np.array([b[:4] for b in boxes], dtype=np.float32)
                confidence = np.array([b[4] for b in boxes], dtype=np.float32)

                sv_detections = sv.Detections(xyxy=xyxy, confidence=confidence)
            else:
                sv_detections = sv.Detections.empty()

            # Step 4: Update ByteTrack to get persistent tracking IDs[cite: 1]
            tracked_detections = tracker.update_with_detections(sv_detections)

            # Step 5: Temporal Filtering (Discard single-frame noise blips)[cite: 1]
            valid_indices = []
            if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0:
                for idx, track_id in enumerate(tracked_detections.tracker_id):
                    track_history[track_id] = track_history.get(track_id, 0) + 1

                    # Only keep objects visible for at least N frames[cite: 1]
                    if track_history[track_id] >= minimum_consecutive_frames:
                        valid_indices.append(idx)

                # Filter detections to keep only temporally persistent tracks
                if valid_indices:
                    filtered_detections = tracked_detections[np.array(valid_indices)]
                else:
                    filtered_detections = sv.Detections.empty()
            else:
                filtered_detections = sv.Detections.empty()

            # Person B (render.py) will draw on this frame[cite: 1]
            # For now, write raw frame or pass filtered_detections to render.py
            out.write(frame)

        frame_count += 1

    cap.release()
    out.release()
    print(f"Video processing complete. Saved to {output_path}")