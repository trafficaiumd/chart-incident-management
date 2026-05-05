from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "temp"
DEFAULT_WEIGHTS = PROJECT_ROOT / "yolo_ai_layer" / "epoch14.pt"

CONF_THRESHOLD = 0.40
VID_STRIDE = 8
CLIP_DURATION_SEC = 10.0
ACCIDENT_CLASS_NAME = "accident"


def _is_video(path: Path) -> bool:
    return path.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"}


def _is_image(path: Path) -> bool:
    return path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def _best_accident_conf(result_obj: Any, class_names: Any) -> tuple[float, bool]:
    boxes = getattr(result_obj, "boxes", None)
    if boxes is None or not hasattr(boxes, "cls") or not hasattr(boxes, "conf"):
        return 0.0, False
    max_conf = 0.0
    found = False
    for cls, conf in zip(boxes.cls.tolist(), boxes.conf.tolist()):
        name = str(class_names[int(cls)]).lower() if isinstance(class_names, dict) else str(int(cls))
        c = float(conf)
        if ACCIDENT_CLASS_NAME in name:
            found = True
            if c > max_conf:
                max_conf = c
    return max_conf, found


def _write_clip(video_path: Path, out_path: Path, center_frame_idx: int, fps: float, width: int, height: int) -> tuple[int, int]:
    half_frames = int(round((CLIP_DURATION_SEC * fps) / 2.0))
    start_frame = max(0, center_frame_idx - half_frames)
    end_frame = center_frame_idx + half_frames

    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    fourcc = cv2.VideoWriter_fourcc(*"VP80")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))

    idx = start_frame
    while idx <= end_frame:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        writer.write(frame)
        idx += 1
    cap.release()
    writer.release()
    return start_frame, max(start_frame, idx - 1)


def run_detection(media_path: str, weights_path: Optional[str] = None, output_dir: Optional[str] = None) -> Dict[str, Any]:
    media = Path(media_path).expanduser().resolve()
    if not media.exists():
        raise FileNotFoundError(f"Input media not found: {media}")

    out_dir = Path(output_dir).expanduser().resolve() if output_dir else DEFAULT_OUTPUT_DIR.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_out = out_dir / "best_accident_frame_raw.jpg"
    ann_out = out_dir / "best_accident_frame_annotated.jpg"
    ann_alt_out = out_dir / "annotated.jpg"
    clip_out = out_dir / "accident_context_clip_raw.webm"

    model = YOLO(str(Path(weights_path).expanduser().resolve() if weights_path else DEFAULT_WEIGHTS))
    class_names = getattr(model, "names", {})

    if _is_image(media):
        frame = cv2.imread(str(media))
        if frame is None:
            raise RuntimeError(f"Could not read image: {media}")
        result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        annotated = result.plot()
        conf, found = _best_accident_conf(result, class_names)
        cv2.imwrite(str(raw_out), frame)
        cv2.imwrite(str(ann_out), annotated)
        cv2.imwrite(str(ann_alt_out), annotated)
        return {
            "input_media": str(media),
            "media_type": "image",
            "best_frame_index": 0,
            "best_accident_confidence": float(conf if found else 0.0),
            "best_accident_frame_raw": str(raw_out.resolve()),
            "best_accident_frame_annotated": str(ann_out.resolve()),
            "annotated_image": str(ann_alt_out.resolve()),
            "accident_context_clip_raw": "",
            "clip_start_s": 0.0,
            "clip_end_s": 0.0,
            "fps": 0.0,
        }

    if not _is_video(media):
        raise ValueError(f"Unsupported media type: {media.suffix}")

    cap = cv2.VideoCapture(str(media))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {media}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

    best_conf = -1.0
    best_idx = 0
    best_raw = None
    best_ann = None
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            break
        if idx % VID_STRIDE != 0:
            idx += 1
            continue
        result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        annotated = result.plot()
        conf, found = _best_accident_conf(result, class_names)
        if found and conf >= best_conf:
            best_conf = conf
            best_idx = idx
            best_raw = frame.copy()
            best_ann = annotated.copy()
        idx += 1
    cap.release()

    if best_raw is None:
        # deterministic fallback to first frame so downstream can proceed
        cap = cv2.VideoCapture(str(media))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError("Failed to extract any frame from video.")
        result = model(frame, conf=CONF_THRESHOLD, verbose=False)[0]
        best_raw = frame
        best_ann = result.plot()
        best_conf = 0.0
        best_idx = 0

    cv2.imwrite(str(raw_out), best_raw)
    cv2.imwrite(str(ann_out), best_ann)
    cv2.imwrite(str(ann_alt_out), best_ann)
    start_f, end_f = _write_clip(media, clip_out, best_idx, float(fps), width, height)

    return {
        "input_media": str(media),
        "media_type": "video",
        "best_frame_index": int(best_idx),
        "best_accident_confidence": float(max(best_conf, 0.0)),
        "best_accident_frame_raw": str(raw_out.resolve()),
        "best_accident_frame_annotated": str(ann_out.resolve()),
        "annotated_image": str(ann_alt_out.resolve()),
        "accident_context_clip_raw": str(clip_out.resolve()),
        "clip_start_s": float(start_f / max(fps, 1.0)),
        "clip_end_s": float(end_f / max(fps, 1.0)),
        "fps": float(fps),
    }


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="YOLO detection stage for Traffic.AI.")
    p.add_argument("--media_path", required=True, help="Path to input image or video")
    p.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to YOLO weights")
    p.add_argument("--output_dir", default=str(DEFAULT_OUTPUT_DIR), help="Output folder for detection artifacts")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    report = run_detection(args.media_path, weights_path=args.weights, output_dir=args.output_dir)
    print(report)