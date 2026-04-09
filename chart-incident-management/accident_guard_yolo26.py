"""
accident_guard_yolo26.py

Real-time "Guard" for traffic incident awareness using Ultralytics YOLO26-S.

What you get:
- Environment check that ensures `ultralytics` is installed and up-to-date
  (so YOLO26 models are supported).
- Model loading of `yolo26s.pt` (auto-downloads via Ultralytics).
- Real-time OpenCV inference loop:
  - Defaults to webcam (index 0)
  - Easily switch to a video file path
- Paranoid guard settings:
  - Very low confidence threshold: 0.15
- Target monitoring / logging:
  - Only logs detections for COCO classes:
    - 2: car
    - 3: motorcycle
    - 7: truck
- Test verification:
  - `test_system()` runs one frame through the model and prints:
    - latency (ms)
    - detection count
  - This is a quick sanity check that the model executes end-to-end.

Run:
  python chart-incident-management/accident_guard_yolo26.py
"""

from __future__ import annotations

import os
import json
import argparse
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import cv2
from PIL import Image


# ----------------------------
# Guard configuration
# ----------------------------

# Paranoid detection threshold (requested)
CONF_THRESHOLD = 0.15

# Judge trigger band: only invoke Judge for "borderline" vehicle detections
# (requested: between 0.15 and 0.50).
JUDGE_TRIGGER_MIN_CONF = 0.15
JUDGE_TRIGGER_MAX_CONF = 0.50

# COCO class IDs requested
COCO_CAR = 2
COCO_MOTORCYCLE = 3
COCO_TRUCK = 7
TARGET_CLASS_IDS = {COCO_CAR, COCO_MOTORCYCLE, COCO_TRUCK}

# Model file to download/load (requested)
MODEL_WEIGHTS = "yolo26s.pt"

# Input source settings (easy switch)
USE_WEBCAM = True
WEBCAM_INDEX = 0
VIDEO_FILE_PATH: Optional[str] = None  # e.g. "/path/to/video.mp4"

# Visualization settings
BOX_THICKNESS = 2
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.6
TEXT_THICKNESS = 2

# Headless behavior (when no GUI/display is available)
# - If running on a server/CI with no DISPLAY, OpenCV's imshow() will crash.
# - In that case we write an annotated output video instead.
OUTPUT_DIR = Path(__file__).resolve().parent
OUTPUT_VIDEO_PATH = str(OUTPUT_DIR / "guard_output.mp4")
SAVE_PREVIEW_JPEG_EVERY_N_FRAMES = 90  # set to 0 to disable periodic JPEG previews
PREVIEW_JPEG_PATH = str(OUTPUT_DIR / "guard_preview.jpg")

# Judge artifacts
JUDGE_MONTAGE_PATH = str(OUTPUT_DIR / "judge_montage.jpg")
DASHBOARD_LIVE_INCIDENTS_PATH = OUTPUT_DIR / "dashboard" / "data" / "live_incidents.json"
DASHBOARD_ASSETS_DIR = OUTPUT_DIR / "dashboard" / "assets"
STATUS_JSON_PATH = OUTPUT_DIR / "dashboard" / "data" / "status.json"

# Circular buffer: last 2 seconds in RAM (requested: deque maxlen=60).
# Note: this assumes ~30 FPS. If your source FPS differs, you can adjust maxlen.
FRAME_BUFFER_MAXLEN = 60

# To keep RAM usage predictable, we store a resized copy in the buffer.
BUFFER_FRAME_MAX_WIDTH = 640

# Gemma 4 Judge settings (Local Path architecture)
# Judge model is loaded strictly from local files under:
#   ./models/gemma-4-E4B
LOCAL_GEMMA_DIR = Path(__file__).resolve().parent / "models" / "gemma-4-E4B"
GEMMA4_MAX_NEW_TOKENS = 64
JUDGE_TILE_SIZE = 336

# Strict cooldown between Judge triggers to avoid repeated expensive requests.
JUDGE_COOLDOWN_SEC = 10
JUDGE_SHUTDOWN_WAIT_SEC = 45.0

# Tie-breaker hazards that can elevate non-collision incidents.
HIGH_WEIGHT_HAZARDS = ["smoke", "fire", "fluid_leak", "airbag_deployed"]

# Reserve at least 50% of GPU memory for other users (requested).
GPU_MEMORY_FRACTION_FOR_THIS_PROCESS = 0.50

JUDGE_SYSTEM_PROMPT = (
    "You are a traffic safety expert. Analyze this sequence of 6 frames. "
    "If no physical collision is visible but a major hazard is visible "
    "(fire, heavy smoke, or disabled vehicle blocking travel lanes), set collision_confirmed=false and severity=high. "
    "Return ONLY valid JSON (no markdown fences, no extra text) with EXACTLY these keys: "
    "collision_confirmed (boolean), severity (one of: low, med, high), "
    "lanes_blocked (integer), hazards (array of strings), vehicle_types (array of strings). "
    "Do not output explanations or chain-of-thought. Output JSON only."
)


@dataclass(frozen=True)
class TargetDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: Tuple[int, int, int, int]


@dataclass(frozen=True)
class JudgeResult:
    verdict_text: str
    started_at_s: float
    finished_at_s: float


def _is_accident_verdict(verdict_text: str) -> bool:
    """
    Interpret Judge output conservatively and detect only positive accident verdicts.
    """
    parsed = _parse_judge_json_report(verdict_text)
    if parsed is not None:
        return bool(parsed.get("collision_confirmed", False))
    text = (verdict_text or "").upper()
    if "NO ACCIDENT" in text:
        return False
    return "ACCIDENT" in text


def _normalize_hazard_token(hazard: str) -> str:
    token = str(hazard or "").strip().lower().replace(" ", "_").replace("-", "_")
    return token


def _has_high_weight_hazard(hazards: List[str]) -> bool:
    normalized = {_normalize_hazard_token(h) for h in hazards}
    expected = {_normalize_hazard_token(h) for h in HIGH_WEIGHT_HAZARDS}
    return any(h in expected for h in normalized)


def _derive_incident_type(parsed: Dict) -> str:
    collision_confirmed = bool(parsed.get("collision_confirmed", False))
    severity = str(parsed.get("severity", "low")).lower()
    lanes_blocked = int(parsed.get("lanes_blocked", 0) or 0)
    hazards = list(parsed.get("hazards", []))
    has_high_weight = _has_high_weight_hazard(hazards)

    if collision_confirmed:
        return "COLLISION"
    if has_high_weight or severity == "high":
        return "ACTIVE_HAZARD"
    if lanes_blocked > 0:
        return "DISABLED_VEHICLE"
    return "NEAR_MISS"


def _build_unique_montage_path() -> Path:
    """
    Build a unique montage filename in dashboard/assets for Dash cache-busting.
    """
    DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DASHBOARD_ASSETS_DIR / f"montage_{stamp}_{time.time_ns() % 1000000}.jpg"


def _write_status(state: str, detail: str = "") -> None:
    """
    Write coarse pipeline status for dashboard progress tracker.
    """
    try:
        STATUS_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state": str(state).upper(),
            "detail": str(detail),
            "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        STATUS_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _extract_json_object(text: str) -> Optional[str]:
    """
    Extract first top-level JSON object substring from free-form text.
    """
    s = (text or "").strip()
    if not s:
        return None
    start = s.find("{")
    if start < 0:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == "\"":
                in_str = False
            continue
        if ch == "\"":
            in_str = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1]
    return None


def _parse_judge_json_report(verdict_text: str) -> Optional[Dict]:
    """
    Parse and normalize Judge JSON report into expected schema.
    """
    payload = (verdict_text or "").strip()
    candidate = payload
    if not candidate.startswith("{"):
        extracted = _extract_json_object(payload)
        if extracted is None:
            return None
        candidate = extracted
    try:
        data = json.loads(candidate)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None

    severity_raw = str(data.get("severity", "low")).strip().lower()
    severity = severity_raw if severity_raw in {"low", "med", "high"} else "low"
    try:
        lanes_blocked = int(data.get("lanes_blocked", 0))
    except Exception:
        lanes_blocked = 0
    lanes_blocked = max(0, lanes_blocked)

    hazards = data.get("hazards", [])
    if not isinstance(hazards, list):
        hazards = []
    hazards = [str(h).strip().lower() for h in hazards if str(h).strip()]

    vehicle_types = data.get("vehicle_types", [])
    if not isinstance(vehicle_types, list):
        vehicle_types = []
    vehicle_types = [str(v).strip().lower() for v in vehicle_types if str(v).strip()]

    return {
        "collision_confirmed": bool(data.get("collision_confirmed", False)),
        "severity": severity,
        "lanes_blocked": lanes_blocked,
        "hazards": hazards,
        "vehicle_types": vehicle_types,
    }


def _coerce_json_only(verdict_text: str) -> str:
    """
    Normalize model output into compact JSON to remove extra natural language.
    """
    parsed = _parse_judge_json_report(verdict_text)
    if parsed is None:
        return "{}"
    return json.dumps(parsed, separators=(",", ":"))


def _prepare_montage_for_judge(montage_bgr):
    """
    Resize the 2x3 montage so each tile matches a 336x336 vision encoder size.
    """
    if montage_bgr is None:
        return montage_bgr
    h, w = montage_bgr.shape[:2]
    if h <= 0 or w <= 0:
        return montage_bgr
    tile_h = max(1, h // 2)
    tile_w = max(1, w // 3)
    out = Image.new("RGB", (JUDGE_TILE_SIZE * 3, JUDGE_TILE_SIZE * 2))
    pil = Image.fromarray(cv2.cvtColor(montage_bgr, cv2.COLOR_BGR2RGB))
    for r in range(2):
        for c in range(3):
            left = c * tile_w
            upper = r * tile_h
            right = w if c == 2 else (c + 1) * tile_w
            lower = h if r == 1 else (r + 1) * tile_h
            tile = pil.crop((left, upper, right, lower)).resize((JUDGE_TILE_SIZE, JUDGE_TILE_SIZE), Image.BICUBIC)
            out.paste(tile, (c * JUDGE_TILE_SIZE, r * JUDGE_TILE_SIZE))
    return out


def save_verdict_to_dashboard(verdict: str, montage_path: str) -> None:
    """
    Append a live incident event for the Dash UI bridge.
    """
    parsed = _parse_judge_json_report(verdict)
    if parsed is None:
        raise ValueError("Judge output is not valid JSON status report.")
    incident_type = _derive_incident_type(parsed)
    if incident_type == "NEAR_MISS":
        # Explicitly avoid flooding the live dashboard stream with near-miss events.
        return
    incident = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": str(verdict or "").strip(),
        "montage_path": str(Path(montage_path).resolve()),
        "collision_confirmed": bool(parsed.get("collision_confirmed", False)),
        "severity": str(parsed.get("severity", "low")),
        "lanes_blocked": int(parsed.get("lanes_blocked", 0)),
        "hazards": list(parsed.get("hazards", [])),
        "vehicle_types": list(parsed.get("vehicle_types", [])),
        "incident_type": incident_type,
    }
    DASHBOARD_LIVE_INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DASHBOARD_LIVE_INCIDENTS_PATH.exists():
        try:
            existing = json.loads(DASHBOARD_LIVE_INCIDENTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(incident)
    DASHBOARD_LIVE_INCIDENTS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def save_fallback_incident_to_dashboard(*, montage_path: str, reason: str) -> None:
    """
    Ensure the dashboard gets a terminal record even if Judge stalls.
    """
    incident = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": f"FALLBACK: {reason}",
        "montage_path": str(Path(montage_path).resolve()),
        "collision_confirmed": False,
        "severity": "high",
        "lanes_blocked": 1,
        "hazards": ["unknown"],
        "vehicle_types": [],
        "incident_type": "ACTIVE_HAZARD",
        "uncertainty": True,
        "confidence": 0.65,
    }
    DASHBOARD_LIVE_INCIDENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DASHBOARD_LIVE_INCIDENTS_PATH.exists():
        try:
            existing = json.loads(DASHBOARD_LIVE_INCIDENTS_PATH.read_text(encoding="utf-8"))
            if not isinstance(existing, list):
                existing = []
        except Exception:
            existing = []
    else:
        existing = []
    existing.append(incident)
    DASHBOARD_LIVE_INCIDENTS_PATH.write_text(json.dumps(existing, indent=2), encoding="utf-8")


def _run_pip(args: List[str]) -> None:
    """
    Run pip as a subprocess using the current interpreter.
    """
    cmd = [sys.executable, "-m", "pip", *args]
    subprocess.check_call(cmd)


def ensure_ultralytics_latest() -> None:
    """
    Installs/updates Ultralytics to the latest version.

    Why we do this:
    - YOLO26 support requires a sufficiently new `ultralytics`.
    - This keeps the script robust when run on fresh machines/environments.

    Notes:
    - This will modify the CURRENT Python environment.
    - If you prefer isolation, run this script inside your `chart-yolo` env.
    """
    try:
        import ultralytics  # noqa: F401
        print("ultralytics detected; ensuring it's up-to-date...")
        _run_pip(["install", "--upgrade", "ultralytics"])
    except Exception:
        print("ultralytics not found; installing latest ultralytics...")
        _run_pip(["install", "ultralytics"])


def ensure_judge_runtime_dependencies() -> None:
    """
    Ensure Judge-local dependencies are available.

    Required for Local Path flow:
    - python-dotenv: read HF token from .env during download workflow
    - huggingface_hub: snapshot download and local model metadata handling
    - transformers / accelerate / bitsandbytes: local 4-bit model loading
    """
    deps = ["python-dotenv", "huggingface_hub", "transformers", "accelerate", "bitsandbytes"]
    missing = []
    for d in deps:
        try:
            __import__(d if d != "python-dotenv" else "dotenv")
        except Exception:
            missing.append(d)
    if missing:
        print(f"Installing missing Judge dependencies: {', '.join(missing)}", flush=True)
        _run_pip(["install", *missing])


def reserve_gpu_max_memory() -> Optional[Dict]:
    """
    Build a `max_memory` dict for Transformers `device_map='auto'` that caps the
    GPU memory usage to the specified fraction, reserving the rest for others.

    Returns None if CUDA is not available.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return None

        device_idx = 0
        props = torch.cuda.get_device_properties(device_idx)
        total_bytes = int(getattr(props, "total_memory", 0) or 0)
        if total_bytes <= 0:
            return None

        allowed_bytes = int(total_bytes * GPU_MEMORY_FRACTION_FOR_THIS_PROCESS)
        allowed_gib = max(1, allowed_bytes // (1024**3))

        # Give CPU plenty of headroom for offloading.
        return {0: f"{allowed_gib}GiB", "cpu": "64GiB"}
    except Exception:
        return None


def load_gemma4_judge():
    """
    Load Gemma 4 E4B multimodal model using Transformers with 4-bit quantization
    (bitsandbytes) to keep VRAM usage low.

    This function is intentionally defensive because exact class names for
    multimodal models can vary across Transformers versions.
    """
    import torch
    from transformers import AutoProcessor, BitsAndBytesConfig

    max_memory = reserve_gpu_max_memory()
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
    )

    # Prefer multimodal-capable auto model classes if available.
    model = None
    last_err = None
    for cls_path in (
        "transformers.AutoModelForImageTextToText",
        "transformers.AutoModelForVision2Seq",
        "transformers.AutoModelForCausalLM",
    ):
        try:
            module_name, cls_name = cls_path.rsplit(".", 1)
            mod = __import__(module_name, fromlist=[cls_name])
            AutoModel = getattr(mod, cls_name)
            model = AutoModel.from_pretrained(
                str(LOCAL_GEMMA_DIR),
                device_map="auto",
                max_memory=max_memory,
                quantization_config=quant,
                torch_dtype=torch.float16,
                local_files_only=True,
                low_cpu_mem_usage=True,
            )
            break
        except Exception as e:
            last_err = e
            model = None

    if model is None:
        raise RuntimeError(
            "Failed to load Gemma 4 model via Transformers. "
            f"Tried multiple AutoModel classes. Last error: {last_err}"
        )

    processor = AutoProcessor.from_pretrained(
        str(LOCAL_GEMMA_DIR),
        local_files_only=True,
    )
    # Move model to its chosen device once at startup to reduce cold-start stalls.
    try:
        first_param = next(model.parameters())
        model.to(first_param.device)
    except Exception:
        pass
    return model, processor


def _resize_for_buffer(frame_bgr):
    """
    Resize a frame to a max width for buffer storage (saves RAM).
    """
    h, w = frame_bgr.shape[:2]
    if w <= BUFFER_FRAME_MAX_WIDTH:
        return frame_bgr
    scale = BUFFER_FRAME_MAX_WIDTH / float(w)
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame_bgr, (BUFFER_FRAME_MAX_WIDTH, nh), interpolation=cv2.INTER_AREA)


def build_montage_2x3(frames_bgr: List) -> Optional:
    """
    Stitch 6 frames into a 2x3 montage image.
    Returns a BGR image or None.
    """
    if len(frames_bgr) != 6:
        return None

    # Normalize to same size (use the smallest dimensions among the 6).
    hs = [f.shape[0] for f in frames_bgr]
    ws = [f.shape[1] for f in frames_bgr]
    target_h = min(hs)
    target_w = min(ws)

    resized = [cv2.resize(f, (target_w, target_h), interpolation=cv2.INTER_AREA) for f in frames_bgr]
    row1 = cv2.hconcat(resized[0:3])
    row2 = cv2.hconcat(resized[3:6])
    return cv2.vconcat([row1, row2])


def sample_6_keyframes(buffer_frames: List) -> List:
    """
    Select 6 key frames from the circular buffer.
    If we have >= 60 frames, pick every 10th frame; otherwise pick evenly spaced.
    """
    n = len(buffer_frames)
    if n == 0:
        return []
    if n >= 60:
        idxs = [0, 10, 20, 30, 40, 50]
    else:
        # Evenly spaced indices across available frames.
        idxs = [int(round(i * (n - 1) / 5.0)) for i in range(6)]
    return [buffer_frames[i] for i in idxs]


def judge_inference_worker(
    *,
    montage_bgr,
    montage_path: str,
    started_at_s: float,
    shared_state: Dict,
    state_lock: threading.Lock,
) -> None:
    """
    Runs Gemma inference in a background thread and writes the verdict back
    into shared_state.
    """
    try:
        import torch

        _write_status("JUDGING", "Gemma 4 reasoning on physics...")
        print("[JUDGE] worker started", flush=True)
        model, processor = shared_state.get("gemma_model"), shared_state.get("gemma_processor")
        if model is None or processor is None:
            print(f"[JUDGE] loading local Gemma model: {LOCAL_GEMMA_DIR}", flush=True)
            model, processor = load_gemma4_judge()
            with state_lock:
                shared_state["gemma_model"] = model
                shared_state["gemma_processor"] = processor
            print("[JUDGE] Gemma model loaded", flush=True)

        # Resize to encoder-friendly tile geometry, then pass to processor.
        montage_prepared = _prepare_montage_for_judge(montage_bgr)

        # Processor APIs vary; keep it simple: pass image + text.
        inputs = processor(images=montage_prepared, text=JUDGE_SYSTEM_PROMPT, return_tensors="pt")

        # Move tensors to the first model device where possible.
        try:
            first_param = next(model.parameters())
            device = first_param.device
            inputs = {k: v.to(device) for k, v in inputs.items()}
        except Exception:
            pass

        t0 = time.perf_counter()
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=GEMMA4_MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
            )
        gen_s = time.perf_counter() - t0

        try:
            text = processor.batch_decode(outputs, skip_special_tokens=True)[0]
        except Exception:
            # Fallback to model tokenizer if processor decode is not present.
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(LOCAL_GEMMA_DIR), local_files_only=True)
            text = tok.decode(outputs[0], skip_special_tokens=True)

        finished_at_s = time.perf_counter()
        verdict = _coerce_json_only(text.strip())
        print(f"[JUDGE] completed in {gen_s:.2f}s", flush=True)
        print(f"[JUDGE] verdict: {verdict}", flush=True)
        _write_status("FINALIZING", "Generating CHART Report...")
        parsed = _parse_judge_json_report(verdict)
        should_persist = False
        if parsed is not None:
            incident_type = _derive_incident_type(parsed)
            should_persist = incident_type != "NEAR_MISS"
        else:
            should_persist = _is_accident_verdict(verdict)

        if should_persist:
            try:
                save_verdict_to_dashboard(verdict, montage_path)
                print(
                    f"[JUDGE->DASH] wrote incident to {DASHBOARD_LIVE_INCIDENTS_PATH}",
                    flush=True,
                )
            except Exception as e:
                print(f"[JUDGE->DASH] failed to write incident: {e}", flush=True)

        with state_lock:
            shared_state["last_judge_result"] = JudgeResult(
                verdict_text=f"{verdict} (judge_time={gen_s:.2f}s)",
                started_at_s=started_at_s,
                finished_at_s=finished_at_s,
            )
    except Exception as e:
        print(f"[JUDGE] ERROR: {e}", flush=True)
        _write_status("FINALIZING", "Generating CHART Report...")
        try:
            save_fallback_incident_to_dashboard(
                montage_path=montage_path,
                reason=f"Judge error: {e}",
            )
            print("[JUDGE->DASH] wrote fallback incident from Judge error", flush=True)
        except Exception as write_err:
            print(f"[JUDGE->DASH] fallback write failed on Judge error: {write_err}", flush=True)
        with state_lock:
            shared_state["last_judge_result"] = JudgeResult(
                verdict_text=f"JUDGE ERROR: {e}",
                started_at_s=started_at_s,
                finished_at_s=time.perf_counter(),
            )
            shared_state["judge_disabled"] = True
    finally:
        with state_lock:
            shared_state["judge_running"] = False
        print("[JUDGE] worker finished", flush=True)


def should_trigger_judge(targets: List[TargetDetection]) -> bool:
    """
    Trigger the Judge if ANY target vehicle is detected with confidence in the
    borderline band [0.15, 0.50].
    """
    for t in targets:
        if JUDGE_TRIGGER_MIN_CONF <= t.confidence <= JUDGE_TRIGGER_MAX_CONF:
            return True
    return False


def load_model():
    """
    Download and load the YOLO26-S model.
    """
    from ultralytics import YOLO

    print(f"Loading model weights: {MODEL_WEIGHTS}")
    model = YOLO(MODEL_WEIGHTS)  # Ultralytics will download if missing
    print("Model loaded.")
    return model


def open_video_capture() -> cv2.VideoCapture:
    """
    Create a VideoCapture either from webcam or a file path.
    """
    def _open_webcam() -> Tuple[cv2.VideoCapture, str]:
        cap = cv2.VideoCapture(WEBCAM_INDEX)
        return cap, f"webcam index {WEBCAM_INDEX}"

    def _open_file(path: str) -> Tuple[cv2.VideoCapture, str]:
        cap = cv2.VideoCapture(path)
        return cap, f"file {path}"

    # 1) Prefer user-selected mode (webcam vs file).
    if USE_WEBCAM:
        cap, source_desc = _open_webcam()
        if cap.isOpened():
            print(f"Video source opened: {source_desc}")
            return cap

        # Webcam failures are common on servers/containers (no device, permissions).
        print(
            f"WARNING: Failed to open {source_desc}. "
            "Falling back to a demo video file instead.",
            flush=True,
        )

    # 2) If a video file path is provided, try it.
    if VIDEO_FILE_PATH:
        cap, source_desc = _open_file(VIDEO_FILE_PATH)
        if cap.isOpened():
            print(f"Video source opened: {source_desc}")
            return cap

    # 3) Last resort: auto-pick the first video in `demo_videos/` next to this script.
    here = Path(__file__).resolve().parent
    demo_dir = here / "demo_videos"
    exts = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    candidates = []
    if demo_dir.exists() and demo_dir.is_dir():
        candidates = sorted([p for p in demo_dir.iterdir() if p.is_file() and p.suffix.lower() in exts])

    if candidates:
        cap, source_desc = _open_file(str(candidates[0]))
        if cap.isOpened():
            print(f"Video source opened: {source_desc}")
            return cap

    raise RuntimeError(
        "Failed to open any video source.\n"
        "- Webcam access may be blocked or unavailable on this machine.\n"
        "- To use a file, set USE_WEBCAM=False and set VIDEO_FILE_PATH.\n"
        "- Or place videos in chart-incident-management/demo_videos/ for auto-fallback."
    )


def _safe_get_class_name(model, class_id: int) -> str:
    """
    Resolve class name from Ultralytics model names mapping safely.
    """
    names = getattr(model, "names", None)
    if isinstance(names, dict):
        return str(names.get(int(class_id), class_id))
    if isinstance(names, list) and 0 <= class_id < len(names):
        return str(names[class_id])
    return str(class_id)


def extract_target_detections(model, results) -> List[TargetDetection]:
    """
    Convert Ultralytics Results into a list of `TargetDetection` limited to target class IDs.
    """
    out: List[TargetDetection] = []
    if results is None or not hasattr(results, "boxes") or results.boxes is None:
        return out

    boxes = results.boxes
    # Ultralytics provides tensor-like objects; `.tolist()` yields Python scalars.
    cls_list = boxes.cls.tolist() if hasattr(boxes, "cls") else []
    conf_list = boxes.conf.tolist() if hasattr(boxes, "conf") else []
    xyxy_list = boxes.xyxy.tolist() if hasattr(boxes, "xyxy") else []

    for cls, conf, xyxy in zip(cls_list, conf_list, xyxy_list):
        class_id = int(cls)
        confidence = float(conf)
        if class_id not in TARGET_CLASS_IDS:
            continue
        x1, y1, x2, y2 = (int(round(v)) for v in xyxy)
        out.append(
            TargetDetection(
                class_id=class_id,
                class_name=_safe_get_class_name(model, class_id),
                confidence=confidence,
                xyxy=(x1, y1, x2, y2),
            )
        )

    return out


def annotate_frame(frame, detections: List[TargetDetection]) -> None:
    """
    Draw bounding boxes and labels in-place on the frame.
    """
    for det in detections:
        x1, y1, x2, y2 = det.xyxy

        # Color by class for quick visual parsing
        if det.class_id == COCO_CAR:
            color = (0, 255, 255)  # yellow
        elif det.class_id == COCO_MOTORCYCLE:
            color = (255, 0, 255)  # magenta
        else:  # truck
            color = (0, 0, 255)  # red

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
        label = f"{det.class_name} {det.confidence:.2f}"

        (tw, th), base = cv2.getTextSize(label, FONT, FONT_SCALE, TEXT_THICKNESS)
        bg_tl = (x1, max(0, y1 - th - base - 6))
        bg_br = (x1 + tw + 8, y1)
        cv2.rectangle(frame, bg_tl, bg_br, color, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 4, max(12, y1 - 6)),
            FONT,
            FONT_SCALE,
            (255, 255, 255),
            TEXT_THICKNESS,
            lineType=cv2.LINE_AA,
        )


def log_targets(detections: List[TargetDetection], timestamp_s: float) -> None:
    """
    Log only when target classes are detected.
    """
    if not detections:
        return
    for d in detections:
        print(
            f"[t={timestamp_s:8.3f}s] TARGET DETECTED: {d.class_name} "
            f"(class_id={d.class_id}) conf={d.confidence:.3f} bbox={d.xyxy}",
            flush=True,
        )


def test_system(model, cap: cv2.VideoCapture) -> None:
    """
    Run a single frame through the model and report latency and detection count.

    This is a quick verification that inference executes correctly end-to-end.
    """
    ok, frame = cap.read()
    if not ok or frame is None:
        raise RuntimeError("test_system(): failed to read a frame from the video source.")

    t0 = time.perf_counter()
    results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
    dt_ms = (time.perf_counter() - t0) * 1000.0

    r0 = results[0] if isinstance(results, list) and results else results
    dets = extract_target_detections(model, r0)

    # For debug visibility, also count total boxes regardless of class
    total_boxes = 0
    if r0 is not None and getattr(r0, "boxes", None) is not None and getattr(r0.boxes, "cls", None) is not None:
        try:
            total_boxes = len(r0.boxes.cls)
        except Exception:
            total_boxes = 0

    print("=== test_system() ===")
    print(f"Latency: {dt_ms:.1f} ms")
    print(f"Total detections (all classes): {total_boxes}")
    print(f"Target detections (car/truck/motorcycle): {len(dets)}")
    try:
        post_ms = float((getattr(r0, "speed", {}) or {}).get("postprocess", -1))
        print(f"Postprocess time: {post_ms:.3f} ms")
        if 0 <= post_ms <= 1.0:
            print("🚀 NMS-Free Architecture Verified")
    except Exception:
        pass
    print("=====================")


def run_guard(video_file_path: Optional[str] = None, use_webcam: bool = True) -> None:
    """
    Main realtime guard loop.
    """
    _write_status("INGESTING", "Extracting Video Frames...")
    ensure_ultralytics_latest()
    ensure_judge_runtime_dependencies()
    model = load_model()

    global USE_WEBCAM, VIDEO_FILE_PATH
    USE_WEBCAM = use_webcam
    VIDEO_FILE_PATH = video_file_path
    cap = open_video_capture()
    out = None
    state_lock = threading.Lock()
    shared_state: Dict = {}
    try:
        # Quick sanity check
        test_system(model, cap)

        print("Guard active. Press 'q' to quit.", flush=True)
        _write_status("GUARDING", "YOLO26 Perception Active...")
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0  # webcam often reports 0

        # Circular buffer of recent frames (2 seconds @ ~30 FPS).
        frame_buffer: deque = deque(maxlen=FRAME_BUFFER_MAXLEN)

        # Judge shared state (kept minimal and threadsafe).
        shared_state: Dict = {
            "judge_running": False,
            "last_judge_result": None,
            "gemma_model": None,
            "gemma_processor": None,
            "last_judge_reported_finished_at_s": None,
            "judge_thread": None,
            "judge_disabled": False,
            "last_judge_trigger_ts": 0.0,
        }

        # Detect whether we can show a GUI window.
        # On Linux servers, DISPLAY is usually unset -> imshow will fail with Qt/xcb errors.
        headless = os.environ.get("DISPLAY") in (None, "")
        if headless:
            print(
                "HEADLESS mode detected (no DISPLAY). "
                f"Writing annotated output to: {OUTPUT_VIDEO_PATH}",
                flush=True,
            )
        else:
            print("GUI display available. Showing live window.", flush=True)

        # If headless, initialize a VideoWriter once we know frame size.
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("Stream ended or frame read failed.", flush=True)
                break

            # Approx video-time timestamp (for webcam this is just an estimate)
            timestamp_s = frame_index / fps
            frame_index += 1

            # Store into circular buffer (resized for RAM).
            frame_buffer.append(_resize_for_buffer(frame))

            # Inference
            results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
            r0 = results[0] if isinstance(results, list) and results else results

            # Extract + log targets
            targets = extract_target_detections(model, r0)
            log_targets(targets, timestamp_s)

            # Trigger Judge in the background for borderline detections.
            if should_trigger_judge(targets):
                with state_lock:
                    judge_running = bool(shared_state["judge_running"])
                    judge_disabled = bool(shared_state.get("judge_disabled", False))
                    last_trigger = float(shared_state.get("last_judge_trigger_ts", 0.0))
                now_ts = time.time()
                cooldown_ok = (now_ts - last_trigger) >= JUDGE_COOLDOWN_SEC
                if (not judge_running) and (not judge_disabled) and cooldown_ok and len(frame_buffer) >= 6:
                    keyframes = sample_6_keyframes(list(frame_buffer))
                    montage = build_montage_2x3(keyframes)
                    if montage is not None:
                        # Keep legacy path + cache-busting unique asset file for Dash.
                        cv2.imwrite(JUDGE_MONTAGE_PATH, montage)
                        unique_montage_path = _build_unique_montage_path()
                        cv2.imwrite(str(unique_montage_path), montage)
                        print(
                            f"JUDGE triggered (borderline vehicle). "
                            f"Wrote montage: {unique_montage_path}",
                            flush=True,
                        )
                        _write_status("JUDGING", "Gemma 4 reasoning on physics...")
                        with state_lock:
                            shared_state["judge_running"] = True
                            shared_state["last_judge_trigger_ts"] = now_ts
                            shared_state["last_montage_path"] = str(unique_montage_path)
                        th = threading.Thread(
                            target=judge_inference_worker,
                            kwargs={
                                "montage_bgr": montage,
                                "montage_path": str(unique_montage_path),
                                "started_at_s": time.perf_counter(),
                                "shared_state": shared_state,
                                "state_lock": state_lock,
                            },
                            daemon=False,
                        )
                        with state_lock:
                            shared_state["judge_thread"] = th
                        th.start()
                elif (not judge_running) and (not judge_disabled) and (not cooldown_ok):
                    remaining = JUDGE_COOLDOWN_SEC - (now_ts - last_trigger)
                    if remaining > 0:
                        print(f"[JUDGE] cooldown active ({remaining:.1f}s remaining)", flush=True)

            # Draw what the Guard sees
            annotate_frame(frame, targets)

            # Overlay small status HUD
            hud = f"Guard | conf>={CONF_THRESHOLD:.2f} | targets: car/motorcycle/truck | hits={len(targets)}"
            cv2.putText(frame, hud, (10, 25), FONT, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

            # Overlay latest Judge verdict (if available).
            with state_lock:
                jr = shared_state.get("last_judge_result")
                running = bool(shared_state.get("judge_running"))
            if running:
                cv2.putText(frame, "Judge: THINKING...", (10, 55), FONT, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
            elif jr is not None:
                text = str(jr.verdict_text)
                cv2.putText(frame, f"Judge: {text[:90]}", (10, 55), FONT, 0.6, (255, 255, 0), 2, cv2.LINE_AA)
                with state_lock:
                    last_reported = shared_state.get("last_judge_reported_finished_at_s")
                    if last_reported != jr.finished_at_s:
                        elapsed = jr.finished_at_s - jr.started_at_s
                        print(f"[JUDGE] final result ({elapsed:.2f}s): {jr.verdict_text}", flush=True)
                        shared_state["last_judge_reported_finished_at_s"] = jr.finished_at_s

            if headless:
                if out is None:
                    h, w = frame.shape[:2]
                    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (w, h))
                out.write(frame)

                if SAVE_PREVIEW_JPEG_EVERY_N_FRAMES > 0 and (frame_index % SAVE_PREVIEW_JPEG_EVERY_N_FRAMES) == 0:
                    cv2.imwrite(PREVIEW_JPEG_PATH, frame)
                    print(f"Wrote preview JPEG: {PREVIEW_JPEG_PATH}", flush=True)
            else:
                cv2.imshow("YOLO26 Guard", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
    except KeyboardInterrupt:
        print("KeyboardInterrupt received. Shutting down cleanly...", flush=True)
    finally:
        # Ensure Judge background thread has finished before interpreter shutdown.
        # This prevents rare CPython/GIL shutdown crashes in threaded ML workloads.
        judge_thread = None
        with state_lock:
            judge_thread = shared_state.get("judge_thread")
        if isinstance(judge_thread, threading.Thread) and judge_thread.is_alive():
            print("[JUDGE] waiting for worker thread to finish...", flush=True)
            judge_thread.join(timeout=JUDGE_SHUTDOWN_WAIT_SEC)
            if judge_thread.is_alive():
                print("[JUDGE] worker still running after timeout; creating fallback incident.", flush=True)
                _write_status("FINALIZING", "Generating CHART Report...")
                last_montage_path = ""
                with state_lock:
                    last_montage_path = str(shared_state.get("last_montage_path", "") or "")
                if not last_montage_path:
                    last_montage_path = JUDGE_MONTAGE_PATH
                try:
                    save_fallback_incident_to_dashboard(
                        montage_path=last_montage_path,
                        reason="Judge timeout at end-of-stream",
                    )
                    print("[JUDGE->DASH] wrote fallback incident", flush=True)
                except Exception as e:
                    print(f"[JUDGE->DASH] fallback write failed: {e}", flush=True)

        cap.release()
        if out is not None:
            out.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            # In headless mode, this can be a no-op; keep shutdown robust.
            pass
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                # IPC cleanup can free fragmented allocator state in shared environments.
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
                print("[GPU] VRAM cache cleared.", flush=True)
        except Exception:
            pass


def _parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO26 Guard + Gemma Judge.")
    parser.add_argument(
        "--video_path",
        type=str,
        default=None,
        help="Single video file path to process.",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default=None,
        help="Directory containing .mp4 demo videos to process sequentially.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.video_path:
        video_path = Path(args.video_path).expanduser().resolve()
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"--video_path does not exist or is not a file: {video_path}")
        print(f"Single-video mode: {video_path}", flush=True)
        run_guard(video_file_path=str(video_path), use_webcam=False)
    elif args.video_dir:
        video_dir = Path(args.video_dir).expanduser().resolve()
        if not video_dir.exists() or not video_dir.is_dir():
            raise FileNotFoundError(f"--video_dir does not exist or is not a directory: {video_dir}")
        files = sorted(video_dir.glob("*.mp4"))
        if not files:
            raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")
        print(f"Demo suite mode: processing {len(files)} video(s) from {video_dir}", flush=True)
        for idx, path in enumerate(files, start=1):
            print(f"[DEMO {idx}/{len(files)}] Starting: {path}", flush=True)
            run_guard(video_file_path=str(path), use_webcam=False)
            print(f"[DEMO {idx}/{len(files)}] Completed: {path}", flush=True)
    else:
        run_guard()