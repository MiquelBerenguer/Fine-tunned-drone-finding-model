"""
render.py  —  OWNER: PERSON B (rendering)

Turns a list of detections into a readable annotated image. Every decision
here was locked before writing code — see ROLE_B_REQUIREMENTS.md for the
reasoning behind each one. Summary of what's implemented:

  1. Ellipse shape, inscribed in each box
  2. Color assigned per class, via a stable hash (never Python's hash() —
     that's randomized per process and would reshuffle colors on restart)
  3. Labels show class + reading-order ID + confidence, e.g. "person #3 0.91"
  4. Same-class overlapping boxes (IoU >= threshold) merge into one shape
     with a count label, e.g. "person x12" — bumper-to-bumper cars stay
     separate because touching ≠ overlapping
  5. Line thickness and text scale adapt to image resolution, floored so
     nothing disappears on a small thumbnail
  6. Ellipse opacity scales with confidence — borderline detections fade,
     confident ones stay solid. Label text stays full-opacity always.
  7. A count badge always renders, including "0 found" in a distinct color
     when nothing was detected

COORDINATION NOTE FOR PERSON D:
    draw() needs conf_threshold to match whatever was passed to
    detector.detect(). Opacity is calibrated relative to that threshold, so
    call it as render.draw(image, dets, conf_threshold=conf) — not with the
    default — or opacity will be scaled against the wrong baseline.
"""

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tunables — change these here, not scattered through the functions below.
# ---------------------------------------------------------------------------
DENSE_THRESHOLD = 40          # above this many detections, suppress individual labels
CLUSTER_IOU_THRESHOLD = 0.15  # same-class boxes overlapping this much get merged
MIN_OPACITY = 0.35            # a detection right at conf_threshold still reads at this

PALETTE_BGR = [
    (60, 76, 231),    # coral
    (0, 204, 255),    # amber
    (222, 82, 175),   # purple
    (0, 199, 199),    # teal
    (30, 149, 0),     # green
]


# ---------------------------------------------------------------------------
# 2. Color assignment
# ---------------------------------------------------------------------------
def _stable_hash(text: str) -> int:
    """
    A reproducible string hash. Python's built-in hash() is randomized per
    process (PYTHONHASHSEED) as a security feature — fine for a dict key,
    wrong here, because the same class name would get a different color
    every time the notebook restarts. This is deterministic forever.
    """
    h = 0
    for ch in text:
        h = (h * 31 + ord(ch)) % 1_000_000_007
    return h


def color_for_class(class_name: str) -> tuple:
    """One color per class, stable across runs, no hardcoded class list
    needed — a new class name from a different model just lands wherever
    it hashes to."""
    return PALETTE_BGR[_stable_hash(class_name) % len(PALETTE_BGR)]


# ---------------------------------------------------------------------------
# 6. Confidence -> opacity
# ---------------------------------------------------------------------------
def opacity_for_confidence(conf: float, conf_threshold: float,
                            min_opacity: float = MIN_OPACITY) -> float:
    """
    Linear scale: conf == conf_threshold -> min_opacity, conf == 1.0 -> fully
    opaque. Detections below conf_threshold are already filtered out by
    detector.detect(), so this should never actually need to clamp — the
    clamp is defensive in case the calling order ever changes.
    """
    span = max(1.0 - conf_threshold, 1e-6)
    frac = (conf - conf_threshold) / span
    frac = max(0.0, min(1.0, frac))
    return min_opacity + frac * (1.0 - min_opacity)


# ---------------------------------------------------------------------------
# 4. Overlap-based clustering
# ---------------------------------------------------------------------------
def _iou(box_a, box_b) -> float:
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (xa2 - xa1) * (ya2 - ya1) + (xb2 - xb1) * (yb2 - yb1) - inter
    return inter / union if union > 0 else 0.0


def _cluster_overlapping(detections, iou_threshold: float = CLUSTER_IOU_THRESHOLD) -> list:
    """
    Union-find over same-class boxes that overlap past the threshold.

    Think of it as name tags: everyone starts holding their own tag.
    union(i, j) makes i hold j's tag instead. find(i) follows the chain of
    tags until it reaches whoever holds their own — that's the group's
    representative. Everyone who ends up pointing at the same representative
    is in the same group.

    Grouping triggers on OVERLAP, not proximity — bumper-to-bumper cars
    (touching, not overlapping) stay separate; a genuinely packed crowd
    (boxes overlapping) merges. This is deliberate, not a simplification.
    """
    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i, j):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    for i in range(n):
        for j in range(i + 1, n):
            if detections.class_id[i] != detections.class_id[j]:
                continue
            if _iou(detections.xyxy[i], detections.xyxy[j]) >= iou_threshold:
                union(i, j)

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())


# ---------------------------------------------------------------------------
# 3. Reading-order IDs
# ---------------------------------------------------------------------------
def _reading_order_ids(detections) -> list:
    """1-based IDs in the order a human would naturally count: top-to-bottom,
    left-to-right on ties. Sorting by (y_center, x_center) does this in one
    line — y breaks first, x only resolves ties."""
    centers = [((y1 + y2) / 2, (x1 + x2) / 2) for x1, y1, x2, y2 in detections.xyxy]
    order = sorted(range(len(centers)), key=lambda i: centers[i])
    ids = [0] * len(centers)
    for rank, i in enumerate(order, start=1):
        ids[i] = rank
    return ids


# ---------------------------------------------------------------------------
# 1. Ellipse drawing, with opacity
# ---------------------------------------------------------------------------
def _draw_ellipse(image, box, color_bgr, opacity, thickness):
    """
    OpenCV has no native per-shape alpha. The standard workaround: draw onto
    a COPY of the image, then blend copy against original with addWeighted.
    Everywhere the two images are identical (everywhere nothing was drawn),
    the blend math reduces to the original pixel unchanged — only the
    ellipse pixels actually differ, so only they visibly fade.
    """
    x1, y1, x2, y2 = [int(v) for v in box]
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    ax, ay = max(1, (x2 - x1) // 2), max(1, (y2 - y1) // 2)

    if opacity >= 0.999:
        cv2.ellipse(image, (cx, cy), (ax, ay), 0, 0, 360, color_bgr, thickness, cv2.LINE_AA)
        return image

    overlay = image.copy()
    cv2.ellipse(overlay, (cx, cy), (ax, ay), 0, 0, 360, color_bgr, thickness, cv2.LINE_AA)
    cv2.addWeighted(overlay, opacity, image, 1 - opacity, 0, dst=image)
    return image


# ---------------------------------------------------------------------------
# 3. Labels
# ---------------------------------------------------------------------------
def _build_label(class_name, reading_id, confidence, cluster_size=1) -> str:
    """Singleton: 'person #3 0.91'. Cluster: 'person x12' — no ID or
    confidence shown, because those numbers stop meaning anything once
    several detections have been merged into one shape."""
    if cluster_size > 1:
        return f"{class_name} x{cluster_size}"
    return f"{class_name} #{reading_id} {confidence:.2f}"


def _draw_label(image, box, text, text_scale, thickness):
    """Centered above the box, with a dark backing rect for legibility on
    any background. ty is clamped so a detection at the very top of the
    frame doesn't get a label drawn off-screen with a negative y-coordinate
    — cv2.putText won't error on that, it'll just silently vanish."""
    x1, y1, x2, y2 = [int(v) for v in box]
    cx = (x1 + x2) // 2
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, text_scale, thickness)
    tx = max(0, cx - tw // 2)
    ty = max(th + 4, y1 - 6)

    cv2.rectangle(image, (tx - 2, ty - th - 4), (tx + tw + 2, ty + 2), (20, 20, 20), -1)
    cv2.putText(image, text, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, text_scale,
                (255, 255, 255), thickness, cv2.LINE_AA)
    return image


# ---------------------------------------------------------------------------
# 7. Count badge
# ---------------------------------------------------------------------------
def _draw_count_badge(image, n, text_scale):
    """Always renders, including n == 0 — which is what gives you the
    empty-result warning for free: an amber '0 found' badge instead of a
    silent, unexplained image."""
    text = f"{n} found"
    bg = (0, 140, 255) if n == 0 else (20, 20, 20)
    scale = text_scale * 1.6
    thick = max(1, int(scale * 2))
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    pad = int(th * 0.5)

    cv2.rectangle(image, (10, 10), (10 + tw + 2 * pad, 10 + th + 2 * pad), bg, -1)
    cv2.putText(image, text, (10 + pad, 10 + th + pad), cv2.FONT_HERSHEY_SIMPLEX,
                scale, (255, 255, 255), thick, cv2.LINE_AA)
    return image


# ---------------------------------------------------------------------------
# THE CONTRACT — Person D calls this. Do not change this signature without
# telling D — app.py calls it directly.
# ---------------------------------------------------------------------------
def draw(image: np.ndarray, detections, conf_threshold: float = 0.25,
         iou_cluster_threshold: float = CLUSTER_IOU_THRESHOLD) -> np.ndarray:
    """
    Args:
        image:          HxWx3 RGB numpy array (not modified in place — copied)
        detections:     sv.Detections from detector.detect()
        conf_threshold: MUST match whatever was passed to detector.detect() —
                        opacity is scaled relative to this value
        iou_cluster_threshold: overlap fraction above which same-class boxes merge

    Returns:
        A new annotated HxWx3 RGB array. Never crashes on 0 detections —
        returns the image with just the "0 found" badge.
    """
    out = image.copy()
    h, w = out.shape[:2]
    thickness = max(1, int(round(min(h, w) / 400)))
    text_scale = max(0.35, min(h, w) / 1600)

    n = len(detections)
    if n == 0:
        return _draw_count_badge(out, 0, text_scale)

    groups = _cluster_overlapping(detections, iou_cluster_threshold)
    reading_ids = _reading_order_ids(detections)
    names = detections.data.get("class_name", [str(c) for c in detections.class_id])
    dense = n > DENSE_THRESHOLD

    for group in groups:
        cls = names[group[0]]
        color = color_for_class(cls)

        if len(group) == 1:
            i = group[0]
            conf = float(detections.confidence[i]) if detections.confidence is not None else 1.0
            opacity = opacity_for_confidence(conf, conf_threshold)
            _draw_ellipse(out, detections.xyxy[i], color, opacity, thickness)
            if not dense:
                label = _build_label(cls, reading_ids[i], conf)
                _draw_label(out, detections.xyxy[i], label, text_scale, max(1, thickness // 2))
        else:
            boxes = detections.xyxy[group]
            union_box = (boxes[:, 0].min(), boxes[:, 1].min(),
                        boxes[:, 2].max(), boxes[:, 3].max())
            avg_conf = float(detections.confidence[group].mean()) if detections.confidence is not None else 1.0
            opacity = opacity_for_confidence(avg_conf, conf_threshold)
            _draw_ellipse(out, union_box, color, opacity, thickness)
            label = _build_label(cls, 0, avg_conf, cluster_size=len(group))
            _draw_label(out, union_box, label, text_scale, max(1, thickness // 2))

    return _draw_count_badge(out, n, text_scale)


# ---------------------------------------------------------------------------
# Standalone test — run this file directly to sanity-check output without
# waiting on Person A's real detector. Uses fake sv.Detections, same shape
# detector.STUB_MODE returns.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import supervision as sv

    # Blank test canvas — swap for a real drone photo once you have one.
    img = np.full((720, 1280, 3), (40, 60, 40), dtype=np.uint8)

    dets = sv.Detections(
        xyxy=np.array([
            [100, 100, 160, 220],   # isolated person, high confidence
            [300, 400, 340, 470],   # isolated person, borderline confidence
            [600, 300, 660, 380],   # a car
            [700, 500, 740, 560],   # overlapping cluster start
            [715, 510, 755, 570],   # overlaps the one above -> should merge
            [900, 520, 940, 580],   # overlaps the pair above too -> 3-way merge
        ], dtype=np.float32),
        confidence=np.array([0.95, 0.28, 0.81, 0.6, 0.65, 0.7], dtype=np.float32),
        class_id=np.array([0, 0, 1, 0, 0, 0]),
        data={"class_name": np.array(["person", "person", "car",
                                       "person", "person", "person"])},
    )

    out = draw(img, dets, conf_threshold=0.25)
    cv2.imwrite("render_test_output.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    print("wrote render_test_output.png — open it and check:")
    print("  - two isolated ellipses labeled #1 and #2 (person), different opacity")
    print("  - one isolated ellipse labeled #3 (car)")
    print("  - the last three boxes merged into ONE ellipse labeled 'person x3'")
    print("  - count badge reads '4 found' (3 singles + 1 cluster)")

    # Zero-detection case
    empty = sv.Detections.empty()
    out2 = draw(img, empty, conf_threshold=0.25)
    cv2.imwrite("render_test_empty.png", cv2.cvtColor(out2, cv2.COLOR_RGB2BGR))
    print("wrote render_test_empty.png — check the amber '0 found' badge")