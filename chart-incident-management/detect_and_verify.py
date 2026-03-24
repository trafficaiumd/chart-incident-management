"""
detect_and_verify.py

Perception Layer (MVP) for an Intelligent Traffic Incident Management System.

What this script does:
- Loads `ROBOFLOW_API_KEY` from `roboflow_API.env` via python-dotenv
- Uses Roboflow's official `inference` library to load your fine-tuned model:
  `road-accident-5pzoq-iigit/2`
- Processes a local video file (`highway_feed.mp4`) like a CCTV feed by
  analyzing EXACTLY 1 frame per second
- If an `Accident` is detected with confidence > 0.75:
  - Draws a bounding box + label on the FULL original frame (NO cropping)
  - Saves the annotated full frame as `incident_trigger.jpg`
  - Writes a structured payload to `incident_payload.json`
- Implements a strict 60-second cooldown after a positive detection; during
  cooldown we continue scanning the video but ignore new accident detections

Run:
  python detect_and_verify.py
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
from dotenv import load_dotenv


# ----------------------------
# Configuration
# ----------------------------

MODEL_ID = "road-accident-5pzoq-iigit/2"
TARGET_CLASS_NAME = "Accident"
CONFIDENCE_THRESHOLD = 0.20  # lowered for debugging / sensitivity check

COOLDOWN_SECONDS = 60.0
FRAMES_PER_SECOND_TO_ANALYZE = 3  # debug: analyze 3 frames per second

VIDEO_FILENAME = "highway_feed.mp4"
DOTENV_FILENAME = "roboflow_API.env"

OUTPUT_IMAGE_FILENAME = "incident_trigger.jpg"
OUTPUT_PAYLOAD_FILENAME = "incident_payload.json"
DEMO_VIDEOS_DIRNAME = "demo_videos"
PREFERRED_NIGHT_VIDEO_RELATIVE = "demo_videos/night acc.mp4"


# ----------------------------
# Data structures
# ----------------------------

@dataclass(frozen=True)
class BoundingBox:
    """
    Bounding box in pixel coordinates on the ORIGINAL full frame.
    Coordinates follow the conventional (x_min, y_min) top-left and
    (x_max, y_max) bottom-right corners.
    """

    x_min: int
    y_min: int
    x_max: int
    y_max: int

    def clamp(self, width: int, height: int) -> "BoundingBox":
        """
        Ensure the box stays within the image bounds.
        """
        return BoundingBox(
            x_min=max(0, min(self.x_min, width - 1)),
            y_min=max(0, min(self.y_min, height - 1)),
            x_max=max(0, min(self.x_max, width - 1)),
            y_max=max(0, min(self.y_max, height - 1)),
        )

    def as_dict(self) -> Dict[str, int]:
        return {
            "x_min": int(self.x_min),
            "y_min": int(self.y_min),
            "x_max": int(self.x_max),
            "y_max": int(self.y_max),
        }


@dataclass(frozen=True)
class Detection:
    """
    Normalized detection record (independent of the raw Roboflow response shape).
    """

    class_name: str
    confidence: float
    bbox: BoundingBox


# ----------------------------
# Utilities (timestamps, printing, IO)
# ----------------------------

def format_timestamp_hhmmss(seconds: float) -> str:
    """
    Convert video time in seconds to a friendly HH:MM:SS.mmm string.
    We keep milliseconds for higher fidelity handoff to downstream systems.
    """
    if seconds < 0:
        seconds = 0.0
    td = timedelta(seconds=seconds)

    # timedelta doesn't expose milliseconds cleanly as a separate field; format manually
    total_ms = int(round(seconds * 1000.0))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    """
    Write JSON with stable formatting.
    """
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def status(msg: str) -> None:
    """
    Clean terminal status line.
    """
    print(msg, flush=True)


# ----------------------------
# Roboflow inference integration
# ----------------------------

def load_roboflow_model(model_id: str):
    """
    Load the Roboflow model using the official `inference` library.

    Notes:
    - The `inference` package uses your `ROBOFLOW_API_KEY` to securely fetch
      the fine-tuned model artifacts if required.
    """
    from inference import get_model  # local import to keep startup errors clearer

    return get_model(model_id=model_id)


def _as_dict(obj: Any) -> Optional[Dict[str, Any]]:
    """
    Best-effort conversion to a dictionary. Roboflow response objects may be:
    - Pydantic models with `.model_dump()`
    - Dataclasses / custom classes with `.dict()` or `.to_dict()`
    - Plain dict already
    """
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj
    for method_name in ("model_dump", "dict", "to_dict"):
        method = getattr(obj, method_name, None)
        if callable(method):
            try:
                return method()
            except TypeError:
                # Some .dict() implementations need args; fallback below
                try:
                    return method(exclude_none=True)
                except Exception:
                    pass
            except Exception:
                pass
    return None


def _extract_predictions(raw_result: Any) -> List[Any]:
    """
    `model.infer(...)` can return a list of results or a single result object.
    This normalizes to a single result and then returns its `.predictions` list
    (or equivalent dict field).
    """
    # Some APIs return [result], others return result directly
    result = raw_result
    if isinstance(raw_result, list) and raw_result:
        result = raw_result[0]

    # Common: result.predictions
    preds = getattr(result, "predictions", None)
    if isinstance(preds, list):
        return preds

    # Fallback: dict-like
    result_dict = _as_dict(result)
    if result_dict and isinstance(result_dict.get("predictions"), list):
        return result_dict["predictions"]

    return []


def _prediction_to_detection(
    pred: Any, frame_width: int, frame_height: int
) -> Optional[Detection]:
    """
    Convert a raw prediction into our normalized `Detection`.

    Roboflow object-detection predictions often provide center-based boxes:
      - x, y (center)
      - width, height
    We'll convert to corner coords and clamp to the image bounds.
    """
    # Try attribute-first, then dict fallback
    pred_dict = _as_dict(pred) or {}

    class_name = getattr(pred, "class_name", None) or pred_dict.get("class_name") or pred_dict.get("class")
    confidence = getattr(pred, "confidence", None)
    if confidence is None:
        confidence = pred_dict.get("confidence") or pred_dict.get("confidence_score")

    # Box fields (center-based is typical)
    x = getattr(pred, "x", None) or pred_dict.get("x")
    y = getattr(pred, "y", None) or pred_dict.get("y")
    w = getattr(pred, "width", None) or pred_dict.get("width") or pred_dict.get("w")
    h = getattr(pred, "height", None) or pred_dict.get("height") or pred_dict.get("h")

    # Sometimes corner-based boxes may be available directly
    x_min = pred_dict.get("x_min") or pred_dict.get("xmin")
    y_min = pred_dict.get("y_min") or pred_dict.get("ymin")
    x_max = pred_dict.get("x_max") or pred_dict.get("xmax")
    y_max = pred_dict.get("y_max") or pred_dict.get("ymax")

    if class_name is None or confidence is None:
        return None

    try:
        conf_f = float(confidence)
    except Exception:
        return None

    # Prefer corner coords if present; otherwise convert from center box.
    if all(v is not None for v in (x_min, y_min, x_max, y_max)):
        try:
            box = BoundingBox(int(round(float(x_min))), int(round(float(y_min))),
                              int(round(float(x_max))), int(round(float(y_max))))
        except Exception:
            return None
    elif all(v is not None for v in (x, y, w, h)):
        try:
            cx = float(x)
            cy = float(y)
            bw = float(w)
            bh = float(h)
        except Exception:
            return None

        box = BoundingBox(
            x_min=int(round(cx - bw / 2.0)),
            y_min=int(round(cy - bh / 2.0)),
            x_max=int(round(cx + bw / 2.0)),
            y_max=int(round(cy + bh / 2.0)),
        )
    else:
        return None

    box = box.clamp(frame_width, frame_height)
    return Detection(class_name=str(class_name), confidence=conf_f, bbox=box)


def infer_accident_detections(model: Any, frame_bgr) -> List[Detection]:
    """
    Run model inference on a full frame and return normalized detections.
    """
    raw = model.infer(frame_bgr)
    frame_h, frame_w = frame_bgr.shape[:2]
    detections: List[Detection] = []
    for pred in _extract_predictions(raw):
        det = _prediction_to_detection(pred, frame_w, frame_h)
        if det is not None:
            detections.append(det)
    return detections


# ----------------------------
# Annotation + payload generation
# ----------------------------

def annotate_full_frame(frame_bgr, detection: Detection) -> Any:
    """
    Draw a clear bounding box + label on the FULL original frame.
    CRITICAL: This function never crops; it only draws on the provided frame.
    """
    annotated = frame_bgr.copy()

    # Visual style chosen for visibility on highway footage
    box_color = (0, 0, 255)  # red (BGR)
    thickness = 3

    x1, y1, x2, y2 = detection.bbox.x_min, detection.bbox.y_min, detection.bbox.x_max, detection.bbox.y_max
    cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, thickness)

    label = f"{detection.class_name}: {detection.confidence:.2f}"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    text_thickness = 2

    # Draw label background for readability
    (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, text_thickness)
    label_x = x1
    label_y = max(0, y1 - 10)
    bg_top_left = (label_x, max(0, label_y - text_h - baseline - 6))
    bg_bottom_right = (label_x + text_w + 10, label_y + 6)
    cv2.rectangle(annotated, bg_top_left, bg_bottom_right, box_color, -1)
    cv2.putText(
        annotated,
        label,
        (label_x + 5, label_y),
        font,
        font_scale,
        (255, 255, 255),
        text_thickness,
        lineType=cv2.LINE_AA,
    )

    return annotated


def build_incident_payload(
    *,
    video_timestamp_seconds: float,
    video_timestamp_hhmmss: str,
    detection: Detection,
    model_id: str,
    video_filename: str,
) -> Dict[str, Any]:
    """
    Create the structured JSON handoff payload for downstream systems.
    """
    return {
        "event": "ACCIDENT_DETECTED",
        "model_id": model_id,
        "video": {
            "file": video_filename,
            "timestamp_seconds": round(float(video_timestamp_seconds), 3),
            "timestamp_hhmmss": video_timestamp_hhmmss,
        },
        "detection": {
            "class": detection.class_name,
            "confidence": round(float(detection.confidence), 4),
            "bbox": detection.bbox.as_dict(),
        },
        "artifacts": {
            "annotated_frame": OUTPUT_IMAGE_FILENAME,
        },
    }


# ----------------------------
# Main CCTV-like video loop
# ----------------------------

def process_video(
    *,
    model: Any,
    video_path: Path,
    output_dir: Path,
) -> None:
    """
    Stream through the video and analyze frames at
    `FRAMES_PER_SECOND_TO_ANALYZE` frames per second.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        # Some containers report 0; fallback to a common CCTV-like FPS
        fps = 30.0

    # We analyze frames at a fixed time step by seeking to the
    # appropriate timestamp (in ms) for each sample.
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_seconds = (total_frames / fps) if total_frames > 0 else None

    status(
        f"Video: {video_path.name} | FPS: {fps:.2f}"
        + (f" | Duration: ~{duration_seconds:.1f}s" if duration_seconds else "")
    )
    status(f"Scanning... ({FRAMES_PER_SECOND_TO_ANALYZE} frames per second)")

    cooldown_until_video_time = -1.0

    # Iterate over uniformly spaced samples in video time.
    # If duration is unknown, we loop until a read fails.
    step = 1.0 / float(FRAMES_PER_SECOND_TO_ANALYZE)
    frame_index = 0
    while True:
        target_time_s = frame_index * step
        cap.set(cv2.CAP_PROP_POS_MSEC, target_time_s * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        timestamp_str = format_timestamp_hhmmss(target_time_s)

        # Cooldown logic is based on VIDEO time, not wall-clock time.
        if target_time_s < cooldown_until_video_time:
            remaining = cooldown_until_video_time - target_time_s
            status(f"[{timestamp_str}] Cooldown active ({remaining:.0f}s remaining) - scanning continues")
            frame_index += 1
            continue

        # Run inference on the full frame
        detections = infer_accident_detections(model, frame)

        # Debug visibility: show what the model is thinking on every frame.
        if detections:
            debug_lines = [
                f"[{timestamp_str}] DET {d.class_name} conf={d.confidence:.3f}"
                for d in detections
            ]
            for line in debug_lines:
                status(line)
        else:
            status(f"[{timestamp_str}] No detections")

        # Find the best qualifying Accident detection (highest confidence above threshold).
        # Match class name case-insensitively so 'accident' or 'ACCIDENT' still count.
        qualifying = [
            d for d in detections
            if d.class_name.lower() == TARGET_CLASS_NAME.lower()
            and d.confidence > CONFIDENCE_THRESHOLD
        ]
        qualifying.sort(key=lambda d: d.confidence, reverse=True)

        if not qualifying:
            status(f"[{timestamp_str}] Scanning... (no qualifying accident)")
            frame_index += 1
            continue

        best = qualifying[0]

        # Trigger!
        status(f"[{timestamp_str}] ACCIDENT DETECTED (confidence={best.confidence:.2f}) - Cooldown active")

        annotated = annotate_full_frame(frame, best)
        image_out = output_dir / OUTPUT_IMAGE_FILENAME
        payload_out = output_dir / OUTPUT_PAYLOAD_FILENAME

        # Save full annotated frame (NO cropping)
        cv2.imwrite(str(image_out), annotated)

        payload = build_incident_payload(
            video_timestamp_seconds=target_time_s,
            video_timestamp_hhmmss=timestamp_str,
            detection=best,
            model_id=MODEL_ID,
            video_filename=video_path.name,
        )
        write_json(payload_out, payload)

        # Activate cooldown
        cooldown_until_video_time = target_time_s + COOLDOWN_SECONDS

        frame_index += 1

    cap.release()
    status("Done. Video processing complete.")


def main(argv: Optional[List[str]] = None) -> None:
    """
    Entrypoint. Loads env, loads model, and processes `highway_feed.mp4`.
    """
    import sys

    if argv is None:
        argv = sys.argv[1:]

    here = Path(__file__).resolve().parent

    # Load ROBOFLOW_API_KEY from roboflow_API.env
    dotenv_path = here / DOTENV_FILENAME
    if not dotenv_path.exists():
        raise FileNotFoundError(
            f"Missing {DOTENV_FILENAME} next to this script. Expected: {dotenv_path}"
        )

    load_dotenv(dotenv_path=dotenv_path)
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY not found. Ensure it's set in roboflow_API.env."
        )

    # Load the Roboflow model securely via the official inference SDK
    status(f"Loading Roboflow model: {MODEL_ID}")
    model = load_roboflow_model(MODEL_ID)
    status("Model loaded.")

    # If the user passes one or more video paths on the command line,
    # process each of them sequentially (absolute or relative paths allowed).
    if argv:
        for raw in argv:
            video_path = Path(raw).expanduser()
            if not video_path.is_absolute():
                # Treat relative to current working directory for ergonomics.
                video_path = Path.cwd() / video_path
            if not video_path.exists():
                status(f"Skipping missing video: {video_path}")
                continue
            status(f"=== Processing video: {video_path} ===")
            process_video(model=model, video_path=video_path, output_dir=here)
        return

    # No CLI args: fall back to configured defaults.
    # 1) Prefer the explicitly requested night video next to this script.
    preferred_night_video = here / PREFERRED_NIGHT_VIDEO_RELATIVE
    if preferred_night_video.exists():
        video_path = preferred_night_video
        status(f"Using preferred night video: {video_path}")
    else:
        # 2) Fallback to legacy default name in this folder.
        video_path = here / VIDEO_FILENAME
        if not video_path.exists():
            # 3) Fall back to any demo video in the repository's `demo_videos/` folder.
            demo_dir = (here.parent / DEMO_VIDEOS_DIRNAME).resolve()
            candidates = []
            if demo_dir.exists() and demo_dir.is_dir():
                candidates = sorted([p for p in demo_dir.glob("*.mp4") if p.is_file()])

            if not candidates:
                raise FileNotFoundError(
                    f"Missing video file: {preferred_night_video}\n"
                    f"Missing default video file: {video_path}\n"
                    f"Also no .mp4 files found in: {demo_dir}"
                )

            video_path = candidates[0]
            status(f"Using demo video: {video_path}")

    process_video(model=model, video_path=video_path, output_dir=here)


if __name__ == "__main__":
    main()

