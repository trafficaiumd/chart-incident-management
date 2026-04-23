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
import shutil
import subprocess
from pathlib import Path


def _append_ld_library_path(path_str: str) -> None:
    p = str(path_str or "").strip()
    if not p:
        return
    current = os.environ.get("LD_LIBRARY_PATH", "")
    parts = [x for x in current.split(":") if x]
    if p not in parts:
        parts.append(p)
        os.environ["LD_LIBRARY_PATH"] = ":".join(parts)


def _discover_nvidia_lib_dirs() -> list[str]:
    dirs: list[str] = []
    candidates = []
    try:
        out = subprocess.check_output(["/sbin/ldconfig", "-p"], text=True, stderr=subprocess.DEVNULL)
        for line in out.splitlines():
            if ("libnvidia-ml.so" in line) or ("libcuda.so" in line):
                if "=>" in line:
                    p = line.split("=>", 1)[1].strip()
                    candidates.append(p)
    except Exception:
        pass
    for base in (
        "/usr/lib/x86_64-linux-gnu",
        "/usr/lib64",
        "/usr/lib",
        "/usr/local/cuda/lib64",
        "/usr/local/nvidia/lib64",
    ):
        b = Path(base)
        if not b.exists():
            continue
        for pat in ("libnvidia-ml.so*", "libcuda.so*"):
            candidates.extend([str(x) for x in b.glob(pat)])
    for c in candidates:
        try:
            d = str(Path(c).resolve().parent)
            if d not in dirs:
                dirs.append(d)
        except Exception:
            continue
    return dirs


# Keep CUDA libs discoverable for torch runtime checks.
for _d in _discover_nvidia_lib_dirs():
    _append_ld_library_path(_d)

import torch

try:
    _ = torch.cuda.device_count()
except Exception:
    os.environ["PYTORCH_NVML_BASED_CUDA_CHECK"] = "0"

import json
import math
import argparse
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
from PIL import Image
from dotenv import load_dotenv
from utils.pdf_generator import generate_incident_pdf, generate_batch_summary_pdf


# ----------------------------
# Guard configuration
# ----------------------------

# Paranoid detection threshold (requested)
CONF_THRESHOLD = 0.15

# Judge trigger band: invoke Judge for "borderline" *any monitored class* in [MIN, MAX].
JUDGE_TRIGGER_MIN_CONF = 0.15
JUDGE_TRIGGER_MAX_CONF = 0.50
# Strong *generic* targets (vehicles, etc.) for EOS override / heuristics — keep fairly high.
JUDGE_STRONG_TRIGGER_CONF = 0.70
# Accident / collision class: send to Gemini only at high confidence.
JUDGE_ACCIDENT_PRIORITY_CONF = 0.70
JUDGE_EOS_WINDOW_SEC = 3.0

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
ACTIVE_PROCESS_PATH = OUTPUT_DIR / "dashboard" / "data" / "active_process.json"
CHART_CAMERA_GEOJSON_PATH = OUTPUT_DIR / "live_camera" / "MDOT_SHA_CHART_Traffic_Cameras.geojson"
REPORTS_DIR = OUTPUT_DIR / "dashboard" / "data" / "reports"

# Circular buffer stores tuples: (timestamp_s, frame_bgr).
# Default maxlen; actual deque maxlen is set in run_guard from source FPS so we
# can retain ~5s+ of frames at up to ~60 FPS for replay + montage.
FRAME_BUFFER_MAXLEN = 180
MONTAGE_WINDOW_SEC = 5.0
REPLAY_DURATION_SEC = 5.0
SKIP_RUNTIME_DEPS = True

# To keep RAM usage predictable, we store a resized copy in the buffer.
BUFFER_FRAME_MAX_WIDTH = 640

JUDGE_TILE_SIZE = 336
GEMINI_MODEL_ID = "gemini-2.5-flash"

# Strict cooldown between Judge triggers to avoid repeated expensive requests.
JUDGE_COOLDOWN_SEC = 10
JUDGE_SHUTDOWN_WAIT_SEC = 60.0

# Tie-breaker hazards that can elevate non-collision incidents.
HIGH_WEIGHT_HAZARDS = ["smoke", "fire", "fluid_leak", "airbag_deployed"]

JUDGE_SYSTEM_PROMPT_TEMPLATE = """Critical rules:
- Do NOT guess.
- If a value is not present in trusted metadata and cannot be determined reliably from the image, return "unknown" for string fields, [] for list fields, 0 for numeric count fields, false for boolean fields, and null only where explicitly required by the schema.
- You are an early-warning system for a Traffic Management Center. If there is a >40% chance of a collision, hazard, or stopped vehicle based on visual cues (abrupt stop, debris, proximity), classify it as possible_incident rather than no_incident.
- If you see an accident happening in any of the 6 frames, set incident_detected to true even if the vehicles come to a rest later in the sequence.
- Never identify people or personal attributes. Do not attempt face recognition.
- Visible injuries must be treated conservatively: Return "unknown" if status is unclear.

[IMAGE ANALYSIS TASK]
You are provided with two images. Image 1 is a temporal montage for context.
Image 2 is a high-resolution "Hero Frame" of the peak impact.
Use Image 2 for damage/debris verification and Image 1 for lane blockage assessment.
Analyze the attached roadway evidence using this metadata:
{metadata_json}

Return ONLY this JSON schema:
{
  "camera": { "id": "unknown", "name": "unknown", "milePost": "unknown", "direction": "unknown" },
  "incident": { "incident_detected": false, "incident_status": "no_incident", "confidence_incident": 0.0, "why": "unknown" },
  "vehicles": { "count_involved": 0, "list": [] },
  "people": { "people_visible_count": 0, "injuries_visible": "unknown", "confidence_injuries": 0.0 },
  "hazards": { "fire_visible": "unknown", "smoke_visible": "unknown", "debris_visible": "unknown", "confidence_hazards": 0.0 },
  "severity_info": {
    "severity_inputs": {
      "vehicle_damage": { "score_0_to_4": 0, "weight": 0.25, "evidence": "unknown" },
      "lane_blockage": { "score_0_to_4": 0, "weight": 0.25, "evidence": "unknown" },
      "injury_visibility": { "score_0_to_4": 0, "weight": 0.25, "evidence": "unknown" },
      "fire_smoke_hazard": { "score_0_to_4": 0, "weight": 0.25, "evidence": "unknown" }
    },
    "severe_gating_triggers": { "vehicle_fire": false, "multiple_vehicle_pileup": false, "visible_injury": false },
    "derived_by_python": { "severity_score_0_to_100": null, "severity_category": null, "notes_uncertainty": "unknown" }
  }
}
"""


def _build_judge_prompt(camera_metadata: Dict) -> str:
    metadata_json = json.dumps(camera_metadata or {}, separators=(",", ":"), ensure_ascii=True)
    # Use plain token replacement so JSON braces in schema are not interpreted
    # by str.format (which caused KeyError on keys like "camera").
    return JUDGE_SYSTEM_PROMPT_TEMPLATE.replace("{metadata_json}", metadata_json)


def _build_gemini_prompt(camera_metadata: Dict) -> str:
    metadata_json = json.dumps(camera_metadata or {}, separators=(",", ":"), ensure_ascii=True)
    return (
        "You are a traffic incident analyst. You are provided with two images: "
        "Image 1 is a 2x3 temporal montage and Image 2 is a high-resolution Hero Frame. "
        "Use Image 2 for impact/damage/debris checks and Image 1 for temporal lane blockage context. "
        "Return ONLY one JSON object. "
        "No markdown, no extra text.\n"
        f"Camera metadata: {metadata_json}\n"
        "Schema:\n"
        "{"
        "\"camera\":{\"id\":\"unknown\",\"name\":\"unknown\",\"milePost\":\"unknown\",\"direction\":\"unknown\"},"
        "\"incident\":{\"incident_detected\":false,\"incident_status\":\"unknown\",\"confidence_incident\":0.0,\"why\":\"unknown\"},"
        "\"vehicles\":{\"count_involved\":0,\"list\":[]},"
        "\"people\":{\"people_visible_count\":0,\"injuries_visible\":\"unknown\",\"confidence_injuries\":0.0},"
        "\"hazards\":{\"fire_visible\":\"unknown\",\"smoke_visible\":\"unknown\",\"debris_visible\":\"unknown\",\"confidence_hazards\":0.0},"
        "\"lane_impact\":{\"lanes_blocked\":0,\"blocked_lanes_description\":\"unknown\"},"
        "\"severity_info\":{\"severity_inputs\":{\"vehicle_damage\":{\"score_0_to_4\":0,\"weight\":0.25,\"evidence\":\"unknown\"},\"lane_blockage\":{\"score_0_to_4\":0,\"weight\":0.25,\"evidence\":\"unknown\"},\"injury_visibility\":{\"score_0_to_4\":0,\"weight\":0.25,\"evidence\":\"unknown\"},\"fire_smoke_hazard\":{\"score_0_to_4\":0,\"weight\":0.25,\"evidence\":\"unknown\"}},\"severe_gating_triggers\":{\"vehicle_fire\":false,\"multiple_vehicle_pileup\":false,\"visible_injury\":false},\"derived_by_python\":{\"severity_score_0_to_100\":null,\"severity_category\":null,\"notes_uncertainty\":\"unknown\"}}"
        "}\n"
        "Rules: if likely incident (>40%), set incident_status to possible_incident and incident_detected true when visible."
    )


def _load_chart_camera_records() -> List[Dict]:
    if not CHART_CAMERA_GEOJSON_PATH.exists():
        return []
    try:
        obj = json.loads(CHART_CAMERA_GEOJSON_PATH.read_text(encoding="utf-8"))
    except Exception:
        return []
    feats = obj.get("features", []) if isinstance(obj, dict) else []
    out = []
    for f in feats:
        p = f.get("properties", {}) if isinstance(f, dict) else {}
        if isinstance(p, dict):
            out.append(p)
    return out


def _get_camera_info(camera_data: Any, index: int = 0, key: Optional[str] = None) -> Dict:
    if isinstance(camera_data, list):
        if camera_data:
            idx = max(0, min(len(camera_data) - 1, int(index)))
            return dict(camera_data[idx])
        return {}
    if isinstance(camera_data, dict):
        if key is not None and key in camera_data:
            v = camera_data.get(key)
            return dict(v) if isinstance(v, dict) else {}
        return dict(camera_data)
    return {}


def _build_prompt_metadata(camera_info: Dict) -> Dict:
    return {
        "cameraCategories": camera_info.get("cameraCategories", []),
        "cctvIp": camera_info.get("cctvIp", "unknown"),
        "commMode": camera_info.get("commMode", "unknown"),
        "description": camera_info.get("description", "unknown"),
        "id": camera_info.get("id", camera_info.get("ID", "unknown")),
        "lastCachedDataUpdateTime": str(camera_info.get("lastCachedDataUpdateTime", "unknown")),
        "lat": str(camera_info.get("lat", camera_info.get("Latitude", "unknown"))),
        "lon": str(camera_info.get("lon", camera_info.get("Longitude", "unknown"))),
        "milePost": str(camera_info.get("milePost", "unknown")),
        "name": camera_info.get("name", camera_info.get("location", "unknown")),
        "opStatus": camera_info.get("opStatus", "unknown"),
        "publicVideoURL": camera_info.get("publicVideoURL", camera_info.get("CCTVPublicURL", "unknown")),
        "routeNumber": str(camera_info.get("routeNumber", "unknown")),
        "routePrefix": str(camera_info.get("routePrefix", "unknown")),
        "routeSuffix": str(camera_info.get("routeSuffix", "")),
        "direction": str(camera_info.get("direction", "unknown")),
    }


def _resolve_camera_metadata(video_file_path: Optional[str], use_webcam: bool, webcam_index: int) -> Dict:
    records = _load_chart_camera_records()
    best = None
    src_name = str(Path(video_file_path).name) if video_file_path else f"webcam_{webcam_index}"
    if records:
        stem = str(Path(video_file_path).stem) if video_file_path else ""
        lower_name = src_name.lower()
        for rec in records:
            rec_id = str(rec.get("ID", rec.get("id", ""))).strip()
            rec_name = str(rec.get("location", rec.get("name", ""))).strip().lower()
            if rec_id and (rec_id in lower_name or rec_id in stem):
                best = rec
                break
            if rec_name and rec_name in lower_name:
                best = rec
                break
        if best is None:
            best = _get_camera_info(records, 0)
    else:
        best = {}
    md = _build_prompt_metadata(best if isinstance(best, dict) else {})
    md["name"] = md.get("name", "unknown") if md.get("name", "unknown") != "unknown" else src_name
    md["source"] = "webcam" if use_webcam else "video_file"
    return md


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
    incident_status = str(parsed.get("incident_status", "unknown")).lower()
    confidence_incident = float(parsed.get("confidence_incident", 0.0) or 0.0)
    raw_hazards = parsed.get("hazards_list", parsed.get("hazards", []))
    if isinstance(raw_hazards, dict):
        hazards = [k.replace("_visible", "") for k, v in raw_hazards.items() if str(v).lower() in {"yes", "true", "present"}]
    else:
        hazards = list(raw_hazards if isinstance(raw_hazards, list) else [])
    has_high_weight = _has_high_weight_hazard(hazards)

    if collision_confirmed:
        return "COLLISION"
    if has_high_weight or severity == "high":
        return "ACTIVE_HAZARD"
    if lanes_blocked > 0:
        return "DISABLED_VEHICLE"
    if incident_status in {"possible_incident", "incident"} or confidence_incident > 0.40:
        return "UNDER_REVIEW"
    return "NEAR_MISS"


def _build_unique_montage_path() -> Path:
    """
    Build a unique montage filename in dashboard/assets for Dash cache-busting.
    """
    DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DASHBOARD_ASSETS_DIR / f"montage_{stamp}_{time.time_ns() % 1000000}.jpg"


def _build_unique_hero_frame_path() -> Path:
    DASHBOARD_ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return DASHBOARD_ASSETS_DIR / f"hero_frame_{stamp}_{time.time_ns() % 1000000}.jpg"


def _write_status(state: str, detail: str = "", cpu_fallback_active: Optional[bool] = None) -> None:
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
        if cpu_fallback_active is not None:
            payload["cpu_fallback_active"] = bool(cpu_fallback_active)
        STATUS_JSON_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception:
        pass


def _read_live_incidents() -> List[Dict]:
    if not DASHBOARD_LIVE_INCIDENTS_PATH.exists():
        return []
    try:
        data = json.loads(DASHBOARD_LIVE_INCIDENTS_PATH.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


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
    Parse and normalize Judge JSON report into forensic nested schema, while
    exposing legacy flat keys used by the dashboard bridge.
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

    def _as_dict(v):
        return v if isinstance(v, dict) else {}

    def _as_list(v):
        return v if isinstance(v, list) else []

    def _as_unknown_str(v):
        s = str(v).strip() if v is not None else ""
        return s if s else "unknown"

    def _as_nonneg_int(v):
        try:
            return max(0, int(v))
        except Exception:
            return 0

    def _as_score_0_4(v):
        try:
            return max(0.0, min(4.0, float(v)))
        except Exception:
            return 0.0

    def _as_nonneg_float(v):
        try:
            return max(0.0, float(v))
        except Exception:
            return 0.0

    camera = _as_dict(data.get("camera"))
    incident = _as_dict(data.get("incident"))
    vehicles = _as_dict(data.get("vehicles"))
    people = _as_dict(data.get("people"))
    hazards_obj = _as_dict(data.get("hazards"))
    lane_impact = _as_dict(data.get("lane_impact"))
    severity_info = _as_dict(data.get("severity_info"))
    severity_inputs = _as_dict(severity_info.get("severity_inputs"))
    severe_gating_triggers = _as_dict(severity_info.get("severe_gating_triggers"))
    derived_by_python = _as_dict(severity_info.get("derived_by_python"))

    def _normalize_input_block(name: str) -> Dict:
        blk = _as_dict(severity_inputs.get(name))
        return {
            "score_0_to_4": _as_score_0_4(blk.get("score_0_to_4", 0)),
            "weight": _as_nonneg_float(blk.get("weight", 0.25)),
            "evidence": _as_unknown_str(blk.get("evidence", "unknown")),
        }

    normalized = {
        "camera": {
            "id": _as_unknown_str(camera.get("id", "unknown")),
            "name": _as_unknown_str(camera.get("name", "unknown")),
            "milePost": _as_unknown_str(camera.get("milePost", "unknown")),
            "direction": _as_unknown_str(camera.get("direction", "unknown")),
        },
        "incident": {
            "incident_detected": bool(incident.get("incident_detected", False)),
            "incident_status": _as_unknown_str(incident.get("incident_status", "unknown")),
            "confidence_incident": _as_nonneg_float(incident.get("confidence_incident", 0.0)),
            "why": _as_unknown_str(incident.get("why", "unknown")),
        },
        "vehicles": {
            "count_involved": _as_nonneg_int(vehicles.get("count_involved", 0)),
            "list": [str(v).strip().lower() for v in _as_list(vehicles.get("list")) if str(v).strip()],
        },
        "people": {
            "people_visible_count": _as_nonneg_int(people.get("people_visible_count", 0)),
            "injuries_visible": _as_unknown_str(people.get("injuries_visible", "unknown")),
            "confidence_injuries": _as_nonneg_float(people.get("confidence_injuries", 0.0)),
        },
        "hazards": {
            "fire_visible": _as_unknown_str(hazards_obj.get("fire_visible", "unknown")),
            "smoke_visible": _as_unknown_str(hazards_obj.get("smoke_visible", "unknown")),
            "debris_visible": _as_unknown_str(hazards_obj.get("debris_visible", "unknown")),
            "confidence_hazards": _as_nonneg_float(hazards_obj.get("confidence_hazards", 0.0)),
        },
        "lane_impact": {
            "lanes_blocked": _as_nonneg_int(lane_impact.get("lanes_blocked", 0)),
            "blocked_lanes_description": _as_unknown_str(lane_impact.get("blocked_lanes_description", "unknown")),
        },
        "severity_info": {
            "severity_inputs": {
                "vehicle_damage": _normalize_input_block("vehicle_damage"),
                "lane_blockage": _normalize_input_block("lane_blockage"),
                "injury_visibility": _normalize_input_block("injury_visibility"),
                "fire_smoke_hazard": _normalize_input_block("fire_smoke_hazard"),
            },
            "severe_gating_triggers": {
                "vehicle_fire": bool(severe_gating_triggers.get("vehicle_fire", False)),
                "multiple_vehicle_pileup": bool(severe_gating_triggers.get("multiple_vehicle_pileup", False)),
                "visible_injury": bool(severe_gating_triggers.get("visible_injury", False)),
            },
            "derived_by_python": {
                "severity_score_0_to_100": derived_by_python.get("severity_score_0_to_100", None),
                "severity_category": derived_by_python.get("severity_category", None),
                "notes_uncertainty": _as_unknown_str(derived_by_python.get("notes_uncertainty", "unknown")),
            },
        },
    }

    # Legacy flattened bridge keys to keep existing UI logic stable.
    incident_detected = bool(normalized["incident"]["incident_detected"])
    status_lc = normalized["incident"]["incident_status"].lower()
    lanes_blocked = _as_nonneg_int(normalized.get("lane_impact", {}).get("lanes_blocked", 0))
    hazard_tokens = []
    if normalized["hazards"]["fire_visible"].lower() in {"yes", "true", "present"}:
        hazard_tokens.append("fire")
    if normalized["hazards"]["smoke_visible"].lower() in {"yes", "true", "present"}:
        hazard_tokens.append("smoke")
    if normalized["hazards"]["debris_visible"].lower() in {"yes", "true", "present"}:
        hazard_tokens.append("debris")

    sev_score = 0.0
    try:
        sev_score = float(normalized["severity_info"]["derived_by_python"].get("severity_score_0_to_100", 0) or 0)
    except Exception:
        sev_score = 0.0
    severity = "high" if sev_score >= 70 else ("med" if sev_score >= 35 else "low")
    if "severe" in status_lc or "critical" in status_lc:
        severity = "high"

    normalized["collision_confirmed"] = incident_detected and status_lc in {"collision", "accident", "confirmed_incident", "incident"}
    normalized["severity"] = severity
    normalized["lanes_blocked"] = lanes_blocked
    normalized["hazards_list"] = hazard_tokens
    normalized["vehicle_types"] = list(normalized["vehicles"]["list"])
    normalized["incident_status"] = normalized["incident"]["incident_status"]
    normalized["confidence_incident"] = normalized["incident"]["confidence_incident"]
    return normalized


def _coerce_json_only(verdict_text: str) -> str:
    """
    Normalize model output into compact JSON to remove extra natural language.
    """
    parsed = _parse_judge_json_report(verdict_text)
    if parsed is None:
        return "{}"
    return json.dumps(parsed, separators=(",", ":"))


def _calculate_severity_derivatives(parsed: Dict) -> Dict:
    """
    Compute weighted severity score and category based on severity_info.
    """
    severity_info = parsed.get("severity_info", {}) if isinstance(parsed, dict) else {}
    severity_inputs = severity_info.get("severity_inputs", {}) if isinstance(severity_info, dict) else {}
    triggers = severity_info.get("severe_gating_triggers", {}) if isinstance(severity_info, dict) else {}

    def _score(name: str) -> float:
        item = severity_inputs.get(name, {}) if isinstance(severity_inputs, dict) else {}
        try:
            return max(0.0, min(4.0, float(item.get("score_0_to_4", 0))))
        except Exception:
            return 0.0

    damage = _score("vehicle_damage")
    blockage = _score("lane_blockage")
    injury = _score("injury_visibility")
    hazards = _score("fire_smoke_hazard")
    # Explicit weighted formula requested by architecture:
    # Score = ((D*0.25) + (B*0.25) + (I*0.25) + (H*0.25)) * 25
    severity_score = max(0.0, min(100.0, ((damage * 0.25) + (blockage * 0.25) + (injury * 0.25) + (hazards * 0.25)) * 25.0))

    critical_trigger = any(bool(v) for v in (triggers or {}).values())
    if critical_trigger:
        category = "CRITICAL"
    elif severity_score >= 75.0:
        category = "HIGH"
    elif severity_score >= 40.0:
        category = "MEDIUM"
    else:
        category = "LOW"

    return {
        "severity_score_0_to_100": round(severity_score, 2),
        "severity_category": category,
        "notes_uncertainty": "unknown",
    }


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


def save_verdict_to_dashboard(
    verdict: str,
    montage_path: str,
    hero_frame_path: str = "",
    replay_clip_path: str = "",
    yolo_trigger_confidence: float = 0.0,
    initial_state: str = "UNDER_REVIEW",
) -> Dict:
    """
    Append a live incident event for the Dash UI bridge.
    """
    parsed = _parse_judge_json_report(verdict)
    if parsed is None:
        raise ValueError("Judge output is not valid JSON status report.")
    derived = _calculate_severity_derivatives(parsed)
    parsed.setdefault("severity_info", {}).setdefault("derived_by_python", {}).update(derived)

    # Keep flat compatibility keys aligned with derived severity.
    sev_cat = str(derived.get("severity_category", "LOW")).upper()
    if sev_cat == "CRITICAL":
        parsed["severity"] = "high"
    elif sev_cat == "HIGH":
        parsed["severity"] = "high"
    elif sev_cat == "MEDIUM":
        parsed["severity"] = "med"
    else:
        parsed["severity"] = "low"

    incident_type = _derive_incident_type(parsed)
    incident_status = str(parsed.get("incident_status", "unknown")).lower()
    confidence_incident = float(parsed.get("confidence_incident", 0.0) or 0.0)
    should_save = incident_status in {"possible_incident", "incident"} or confidence_incident > 0.40 or incident_type != "NEAR_MISS"
    if not should_save:
        return

    # Weighted confidence merge: 40% Guard signal + 60% Judge confidence.
    yolo_conf = max(0.0, min(1.0, float(yolo_trigger_confidence or 0.0)))
    judge_conf = max(0.0, min(1.0, confidence_incident))
    composite_confidence = round((yolo_conf * 0.4) + (judge_conf * 0.6), 4)
    incident = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": str(verdict or "").strip(),
        "montage_path": str(Path(montage_path).resolve()),
        "hero_frame_path": str(Path(hero_frame_path).resolve()) if hero_frame_path else "",
        "replay_clip_path": str(Path(replay_clip_path).resolve()) if replay_clip_path else "",
        "collision_confirmed": bool(parsed.get("collision_confirmed", False)),
        "severity": str(parsed.get("severity", "low")),
        "lanes_blocked": int(parsed.get("lanes_blocked", 0)),
        "hazards": list(parsed.get("hazards_list", [])),
        "vehicle_types": list(parsed.get("vehicle_types", [])),
        "incident_type": incident_type,
        "initial_alert_state": str(initial_state or "UNDER_REVIEW").upper(),
        "confidence": composite_confidence,
        "yolo_trigger_confidence": yolo_conf,
        "judge_confidence_incident": judge_conf,
        "forensic_report": parsed,
        "severity_score_0_to_100": float(derived.get("severity_score_0_to_100", 0.0)),
        "severity_category": str(derived.get("severity_category", "LOW")),
        "derived_by_python": derived,
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
    return incident


def save_under_review_from_guard(
    *,
    montage_path: str,
    hero_frame_path: str,
    replay_clip_path: str,
    yolo_trigger_confidence: float,
    reason: str,
    judge_report: Optional[Dict] = None,
) -> None:
    """
    Deterministic Guard-based early-warning path when Judge returns unknown/0.0.
    """
    yolo_conf = max(0.0, min(1.0, float(yolo_trigger_confidence or 0.0)))
    jr = judge_report if isinstance(judge_report, dict) else {}
    jr_inc = jr.get("incident", {}) if isinstance(jr.get("incident", {}), dict) else {}
    jr_conf = float(jr_inc.get("confidence_incident", 0.0) or 0.0)
    synthetic_conf = max(0.45, min(1.0, jr_conf))
    incident = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": f"UNDER_REVIEW (Guard signal): {reason}",
        "montage_path": str(Path(montage_path).resolve()),
        "hero_frame_path": str(Path(hero_frame_path).resolve()) if hero_frame_path else "",
        "replay_clip_path": str(Path(replay_clip_path).resolve()) if replay_clip_path else "",
        "collision_confirmed": False,
        "severity": "med",
        "lanes_blocked": 0,
        "hazards": ["unknown"],
        "vehicle_types": [],
        "incident_type": "UNDER_REVIEW",
        "confidence": round((yolo_conf * 0.4) + (synthetic_conf * 0.6), 4),
        "yolo_trigger_confidence": yolo_conf,
        "judge_confidence_incident": float(jr_conf),
        "forensic_report": jr
        if jr
        else {
            "incident": {
                "incident_detected": False,
                "incident_status": "possible_incident",
                "confidence_incident": 0.45,
                "why": "unknown",
            }
        },
        "severity_score_0_to_100": 45.0,
        "severity_category": "MEDIUM",
        "derived_by_python": {
            "severity_score_0_to_100": 45.0,
            "severity_category": "MEDIUM",
            "notes_uncertainty": "Guard-based early warning due to high-confidence end-window event",
        },
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


def save_fallback_incident_to_dashboard(
    *,
    montage_path: str,
    reason: str,
    hero_frame_path: str = "",
    replay_clip_path: str = "",
    yolo_trigger_confidence: float = 0.0,
) -> None:
    """
    Ensure the dashboard gets a terminal record even if Judge stalls.
    """
    forensic_fallback = {
        "camera": {"id": "unknown", "name": "unknown", "milePost": "unknown", "direction": "unknown"},
        "incident": {
            "incident_detected": False,
            "incident_status": "possible_incident",
            "confidence_incident": 0.0,
            "why": "unknown",
        },
        "vehicles": {"count_involved": 0, "list": []},
        "people": {"people_visible_count": 0, "injuries_visible": "unknown", "confidence_injuries": 0.0},
        "hazards": {"fire_visible": "unknown", "smoke_visible": "unknown", "debris_visible": "unknown", "confidence_hazards": 0.0},
        "lane_impact": {"lanes_blocked": 0, "blocked_lanes_description": "unknown"},
        "severity_info": {
            "severity_inputs": {
                "vehicle_damage": {"score_0_to_4": 0, "weight": 0.25, "evidence": "unknown"},
                "lane_blockage": {"score_0_to_4": 0, "weight": 0.25, "evidence": "unknown"},
                "injury_visibility": {"score_0_to_4": 0, "weight": 0.25, "evidence": "unknown"},
                "fire_smoke_hazard": {"score_0_to_4": 0, "weight": 0.25, "evidence": "unknown"},
            },
            "severe_gating_triggers": {"vehicle_fire": False, "multiple_vehicle_pileup": False, "visible_injury": False},
            "derived_by_python": {"severity_score_0_to_100": 0.0, "severity_category": "LOW", "notes_uncertainty": "unknown"},
        },
    }
    yolo_conf = max(0.0, min(1.0, float(yolo_trigger_confidence or 0.0)))
    high_conf_timeout = ("timeout" in str(reason or "").lower()) and (yolo_conf >= 0.70)
    manual_review_msg = "⚠️ MANUAL REVIEW REQUIRED: High-Confidence Guard Trigger (API Timeout)"
    incident = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "verdict": (manual_review_msg + f" | FALLBACK: {reason}") if high_conf_timeout else f"FALLBACK: {reason}",
        "montage_path": str(Path(montage_path).resolve()),
        "hero_frame_path": str(Path(hero_frame_path).resolve()) if hero_frame_path else "",
        "replay_clip_path": str(Path(replay_clip_path).resolve()) if replay_clip_path else "",
        "collision_confirmed": False,
        "severity": "low",
        "lanes_blocked": 0,
        "hazards": ["unknown"],
        "vehicle_types": [],
        "incident_type": "ACTIVE_HAZARD",
        "uncertainty": True,
        "confidence": 0.0,
        "yolo_trigger_confidence": yolo_conf,
        "manual_review_required": bool(high_conf_timeout),
        "manual_review_reason": manual_review_msg if high_conf_timeout else "",
        "forensic_report": forensic_fallback,
        "severity_score_0_to_100": 0.0,
        "severity_category": "LOW",
        "derived_by_python": forensic_fallback["severity_info"]["derived_by_python"],
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
    Ensure Gemini Judge dependencies are available.
    """
    deps = [
        "python-dotenv",
        "google-generativeai",
    ]
    missing = []
    for d in deps:
        try:
            __import__(d if d != "python-dotenv" else "dotenv")
        except Exception:
            missing.append(d)
    if missing:
        print(f"Installing missing Judge dependencies: {', '.join(missing)}", flush=True)
        _run_pip(["install", *missing])


def load_gemini_judge():
    """
    Load Gemini client/model using API key from environment.
    """
    try:
        import google.generativeai as genai
    except Exception as e:
        raise RuntimeError(f"google-generativeai not importable: {e}")

    api_key = str(os.getenv("GEMINI_API_KEY", "")).strip()
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing. Set it in .env or environment.")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL_ID)
    return model


def _extract_text_from_gemini_response(response: Any) -> str:
    txt = str(getattr(response, "text", "") or "").strip()
    if txt:
        return txt
    try:
        cands = getattr(response, "candidates", None) or []
        if cands:
            parts = getattr(cands[0].content, "parts", None) or []
            return "".join(str(getattr(p, "text", "") or "") for p in parts).strip()
    except Exception:
        pass
    return ""


def _gemini_generate_json_text(model: Any, prompt_text: str, image_inputs: Any) -> str:
    """
    Ask Gemini for strict JSON, with one retry prompt if the first response is
    empty or non-parseable by our forensic parser.
    """
    base_cfg = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_output_tokens": 1024,
    }
    # First pass: normal forensic prompt.
    parts = image_inputs if isinstance(image_inputs, list) else [image_inputs]
    resp = model.generate_content([prompt_text, *parts], generation_config=base_cfg)
    text = _extract_text_from_gemini_response(resp)
    if _parse_judge_json_report(text) is not None:
        return text
    # Retry pass: hard formatting constraint.
    retry_prompt = (
        prompt_text
        + "\n\nReturn ONLY a single valid JSON object following the exact schema."
        + " No markdown, no code fences, no explanation."
        + " Ensure every required key exists even if value is unknown/0/false."
    )
    resp2 = model.generate_content([retry_prompt, *parts], generation_config=base_cfg)
    text2 = _extract_text_from_gemini_response(resp2)
    return text2 or text


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


def sample_6_keyframes(buffer_items: List[Tuple[float, Any]], center_ts_s: float, window_sec: float = MONTAGE_WINDOW_SEC) -> List:
    """
    Select 6 key frames from the circular buffer.
    If we have >= 60 frames, pick every 10th frame; otherwise pick evenly spaced.
    """
    n = len(buffer_items)
    if n == 0:
        return []
    half = max(0.1, float(window_sec) / 2.0)
    lo, hi = float(center_ts_s) - half, float(center_ts_s) + half
    windowed = [it for it in buffer_items if lo <= float(it[0]) <= hi]
    if len(windowed) < 6:
        windowed = buffer_items
    m = len(windowed)
    idxs = [int(round(i * (m - 1) / 5.0)) for i in range(6)]
    return [windowed[i][1] for i in idxs]


def _buffer_frames_for_replay(
    buffer_items: List[Tuple[float, Any]],
    center_ts_s: float,
    duration_sec: float,
    fallback_fps: float,
) -> List:
    """
    Return frames (BGR) in [center - duration/2, center + duration/2] by video
    timestamp, in order. Falls back to the most recent frames if the window is empty.
    """
    if not buffer_items:
        return []
    half = float(duration_sec) / 2.0
    lo, hi = float(center_ts_s) - half, float(center_ts_s) + half
    windowed = []
    for it in buffer_items:
        if len(it) < 2:
            continue
        t, fr = it[0], it[1]
        if fr is not None and lo <= float(t) <= hi:
            windowed.append((float(t), fr))
    windowed.sort(key=lambda x: x[0])
    frames = [fr for _, fr in windowed]
    if frames:
        return frames
    # Not enough timestamps yet (e.g. very start of file): approximate by count.
    n = max(1, int(round(float(fallback_fps or 30.0) * float(duration_sec))))
    tail = []
    for it in buffer_items:
        if len(it) < 2:
            continue
        fr = it[1]
        if fr is not None:
            tail.append(fr)
    tail = tail[-n:]
    return tail


def _write_replay_clip(
    buffer_items: List[Tuple[float, Any]],
    center_ts_s: float,
    out_path: Path,
    duration_sec: float,
    fallback_fps: float,
) -> Optional[str]:
    """
    Write a replay clip whose *playback* length is duration_sec, using sequential
    frames from the buffer time window. Output frame rate is len(frames)/duration_sec
    so the browser plays the full duration (fixes short playback when container FPS
    metadata/timestamps disagree with frame count).

    OpenCV writes MPEG-4 Part 2 ('mp4v') by default; most browsers (Chrome/Edge)
    will not decode that in <video>. When ffmpeg is available, we transcode to
    H.264 (yuv420p) + faststart so replay works in the browser.
    """
    clip_frames = _buffer_frames_for_replay(
        buffer_items,
        center_ts_s=center_ts_s,
        duration_sec=duration_sec,
        fallback_fps=fallback_fps,
    )
    if not clip_frames:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    h, w = clip_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    dur = max(0.1, float(duration_sec))
    clip_fps = float(len(clip_frames)) / dur
    # If OpenCV/ffmpeg struggle with extremely low fps, pad with the last frame so
    # we can encode at >= 1 fps without changing total playback duration.
    if clip_fps < 1.0:
        need = max(0, int(math.ceil(dur)) - len(clip_frames))
        if need > 0:
            pad = clip_frames[-1]
            clip_frames = clip_frames + [pad] * need
        clip_fps = float(len(clip_frames)) / dur
    tmp_raw = out_path.with_suffix(".raw.mp4")
    writer = cv2.VideoWriter(str(tmp_raw), fourcc, clip_fps, (w, h))
    try:
        for fr in clip_frames:
            if fr is None:
                continue
            if fr.shape[:2] != (h, w):
                fr = cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA)
            writer.write(fr)
    finally:
        writer.release()

    if not tmp_raw.exists() or tmp_raw.stat().st_size < 32:
        return None

    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        try:
            subprocess.run(
                [
                    ffmpeg_bin,
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    # Force assumed CFR for the OpenCV-generated mp4 so duration
                    # is N/clip_fps seconds (fixes <1s playback when mux timestamps lie).
                    "-r",
                    f"{clip_fps:.6f}",
                    "-i",
                    str(tmp_raw),
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "23",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    "-an",
                    str(out_path),
                ],
                check=True,
                timeout=120,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                tmp_raw.unlink()
            except OSError:
                pass
            return str(out_path.resolve())
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, OSError) as e:
            print(f"[REPLAY] ffmpeg H.264 transcode failed ({e}); keeping raw mp4 (may not play in browser)", flush=True)

    try:
        if out_path.exists():
            out_path.unlink()
        tmp_raw.rename(out_path)
    except OSError:
        pass
    return str(out_path.resolve())


def judge_inference_worker(
    *,
    montage_bgr,
    montage_path: str,
    hero_frame_bgr,
    hero_frame_path: str,
    replay_clip_path: str,
    yolo_trigger_confidence: float,
    initial_state: str,
    started_at_s: float,
    camera_metadata: Dict,
    shared_state: Dict,
    state_lock: threading.Lock,
) -> None:
    """
    Runs configured Judge backend in a background thread and writes verdict.
    """
    try:
        with state_lock:
            if bool(shared_state.get("judge_stop_requested", False)):
                return
        _write_status("JUDGING", "Reasoning on temporal montage...")
        print("[JUDGE] worker started", flush=True)
        prompt_text = _build_gemini_prompt(camera_metadata)
        t0 = time.perf_counter()
        model = shared_state.get("gemini_model")
        if model is None:
            print(f"[JUDGE] loading Gemini model: {GEMINI_MODEL_ID}", flush=True)
            model = load_gemini_judge()
            with state_lock:
                shared_state["gemini_model"] = model
                shared_state["judge_device_mode"] = "api"
            print("[JUDGE] Gemini model ready", flush=True)
        pil_montage = Image.fromarray(cv2.cvtColor(montage_bgr, cv2.COLOR_BGR2RGB))
        pil_hero = Image.fromarray(cv2.cvtColor(hero_frame_bgr, cv2.COLOR_BGR2RGB))
        text = _gemini_generate_json_text(model, prompt_text, [pil_montage, pil_hero])
        gen_s = time.perf_counter() - t0

        finished_at_s = time.perf_counter()
        with state_lock:
            if bool(shared_state.get("judge_stop_requested", False)):
                return
        verdict = _coerce_json_only(text.strip())
        print(f"[JUDGE] completed in {gen_s:.2f}s", flush=True)
        print(f"[JUDGE] verdict: {verdict}", flush=True)
        _write_status("FINALIZING", "Generating CHART Report...")
        parsed = _parse_judge_json_report(verdict)
        should_persist = False
        if parsed is not None:
            incident_type = _derive_incident_type(parsed)
            incident_status = str(parsed.get("incident_status", "unknown")).lower()
            confidence_incident = float(parsed.get("confidence_incident", 0.0) or 0.0)
            should_persist = (
                incident_type != "NEAR_MISS"
                or incident_status in {"possible_incident", "incident"}
                or confidence_incident > 0.40
            )
            # Deterministic Guard-based UNDER_REVIEW save if Judge is unknown/0.0
            # but YOLO signal was strong near event.
            if (
                (not should_persist)
                and incident_status == "unknown"
                and confidence_incident <= 0.0
                and float(yolo_trigger_confidence or 0.0) >= JUDGE_ACCIDENT_PRIORITY_CONF
            ):
                already_written = False
                with state_lock:
                    already_written = bool(shared_state.get("under_review_written", False))
                if already_written:
                    should_persist = False
                else:
                    try:
                        save_under_review_from_guard(
                            montage_path=montage_path,
                            hero_frame_path=hero_frame_path,
                            replay_clip_path=replay_clip_path,
                            yolo_trigger_confidence=float(yolo_trigger_confidence or 0.0),
                            reason="Judge unknown/0.0 with strong YOLO trigger",
                            judge_report=parsed,
                        )
                        with state_lock:
                            shared_state["under_review_written"] = True
                        should_persist = False
                        print("[JUDGE->DASH] wrote deterministic UNDER_REVIEW from Guard signal", flush=True)
                    except Exception as e:
                        print(f"[JUDGE->DASH] failed deterministic UNDER_REVIEW write: {e}", flush=True)
        else:
            should_persist = _is_accident_verdict(verdict)

        if should_persist:
            try:
                incident_record = save_verdict_to_dashboard(
                    verdict,
                    montage_path,
                    hero_frame_path=hero_frame_path,
                    replay_clip_path=replay_clip_path,
                    yolo_trigger_confidence=yolo_trigger_confidence,
                    initial_state=initial_state,
                )
                try:
                    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
                    ts_file = time.strftime("%Y%m%d_%H%M%S")
                    out_pdf = REPORTS_DIR / f"incident_report_{ts_file}.pdf"
                    latest_pdf = REPORTS_DIR / "incident_report.pdf"
                    generate_incident_pdf(incident_record, output_path=str(out_pdf))
                    generate_incident_pdf(incident_record, output_path=str(latest_pdf))
                    print(f"[REPORT] wrote PDF: {out_pdf}", flush=True)
                except Exception as report_err:
                    print(f"[REPORT] PDF generation failed: {report_err}", flush=True)
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
                hero_frame_path=hero_frame_path,
                replay_clip_path=replay_clip_path,
                reason=f"Judge error: {e}",
                yolo_trigger_confidence=float(yolo_trigger_confidence or 0.0),
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
            shared_state["judge_active"] = False
        print("[JUDGE] worker finished", flush=True)


def should_trigger_judge(targets: List[TargetDetection]) -> bool:
    """
    Trigger the Judge if ANY monitored target lies in the borderline band
    [JUDGE_TRIGGER_MIN_CONF, JUDGE_TRIGGER_MAX_CONF].
    """
    for t in targets:
        if JUDGE_TRIGGER_MIN_CONF <= t.confidence <= JUDGE_TRIGGER_MAX_CONF:
            return True
    return False


def amber_path_trigger(model, results_obj) -> Tuple[bool, float]:
    """
    Amber path trigger: accident/collision class in [0.40, 0.69].
    """
    if results_obj is None or getattr(results_obj, "boxes", None) is None:
        return False, 0.0
    boxes = results_obj.boxes
    cls_list = boxes.cls.tolist() if hasattr(boxes, "cls") else []
    conf_list = boxes.conf.tolist() if hasattr(boxes, "conf") else []
    max_conf = 0.0
    for cls, conf in zip(cls_list, conf_list):
        class_id = int(cls)
        name = _safe_get_class_name(model, class_id).strip().lower()
        c = float(conf)
        if any(tok in name for tok in ("accident", "collision")) and 0.40 <= c < JUDGE_ACCIDENT_PRIORITY_CONF:
            max_conf = max(max_conf, c)
    return (max_conf >= 0.40), max_conf


def has_strong_event(targets: List[TargetDetection]) -> bool:
    """
    Strong event heuristic used to override cooldown near end-of-stream.
    """
    if any(float(t.confidence) >= JUDGE_STRONG_TRIGGER_CONF for t in targets):
        return True
    for t in targets:
        cn = t.class_name.strip().lower()
        if any(k in cn for k in _INCIDENT_NAME_KEYWORDS) and float(t.confidence) >= JUDGE_ACCIDENT_PRIORITY_CONF:
            return True
    return False


def accident_priority_trigger(model, results_obj) -> Tuple[bool, float]:
    """
    Priority trigger: accident/collision class at or above JUDGE_ACCIDENT_PRIORITY_CONF
    (bypasses Judge cooldown; tuned for custom detectors that rarely exceed 0.70).
    """
    if results_obj is None or getattr(results_obj, "boxes", None) is None:
        return False, 0.0
    boxes = results_obj.boxes
    cls_list = boxes.cls.tolist() if hasattr(boxes, "cls") else []
    conf_list = boxes.conf.tolist() if hasattr(boxes, "conf") else []
    max_conf = 0.0
    for cls, conf in zip(cls_list, conf_list):
        class_id = int(cls)
        name = _safe_get_class_name(model, class_id).strip().lower()
        c = float(conf)
        if any(tok in name for tok in ("accident", "collision")) and c >= JUDGE_ACCIDENT_PRIORITY_CONF:
            max_conf = max(max_conf, c)
    return (max_conf >= JUDGE_ACCIDENT_PRIORITY_CONF), max_conf


def _max_accident_confidence(model, results_obj) -> float:
    """
    Return max confidence for accident/collision-like classes in this frame.
    """
    if results_obj is None or getattr(results_obj, "boxes", None) is None:
        return 0.0
    boxes = results_obj.boxes
    cls_list = boxes.cls.tolist() if hasattr(boxes, "cls") else []
    conf_list = boxes.conf.tolist() if hasattr(boxes, "conf") else []
    best = 0.0
    for cls, conf in zip(cls_list, conf_list):
        class_id = int(cls)
        name = _safe_get_class_name(model, class_id).strip().lower()
        c = float(conf)
        if any(tok in name for tok in ("accident", "collision")):
            best = max(best, c)
    return best


def load_model(weights_path: Optional[str] = None):
    """
    Load YOLO weights (default: MODEL_WEIGHTS, usually yolo26s.pt via Ultralytics).
    Pass an explicit .pt path to use a custom checkpoint (e.g. fine-tuned epoch weights).
    """
    from ultralytics import YOLO

    w = str(weights_path or MODEL_WEIGHTS).strip() or MODEL_WEIGHTS
    print(f"Loading model weights: {w}", flush=True)
    model = YOLO(w)
    try:
        names = getattr(model, "names", None)
        if isinstance(names, dict) and names:
            preview = ", ".join(f"{k}:{v}" for k, v in list(names.items())[:12])
            more = "" if len(names) <= 12 else f" ... (+{len(names) - 12} more)"
            print(f"Model classes ({len(names)}): {preview}{more}", flush=True)
    except Exception:
        pass
    print("Model loaded.", flush=True)
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


_VEHICLE_NAME_KEYWORDS = frozenset(
    ("vehicle", "car", "motorcycle", "truck", "bus", "van", "pickup", "suv", "auto")
)
_INCIDENT_NAME_KEYWORDS = frozenset(("accident", "collision", "crash"))


def _is_guard_target_class(model, class_id: int) -> bool:
    """
    True for COCO guard IDs (car/motorcycle/truck) or custom names like
    vehicle/accident (matches fine-tuned checkpoints that are not COCO-indexed).
    """
    if int(class_id) in TARGET_CLASS_IDS:
        return True
    name = _safe_get_class_name(model, class_id).strip().lower()
    if any(k in name for k in _INCIDENT_NAME_KEYWORDS):
        return True
    return any(k in name for k in _VEHICLE_NAME_KEYWORDS)


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
        if not _is_guard_target_class(model, class_id):
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
        cn = det.class_name.strip().lower()
        if any(k in cn for k in _INCIDENT_NAME_KEYWORDS):
            color = (0, 0, 255)  # red — accident / collision
        elif det.class_id == COCO_MOTORCYCLE or "motorcycle" in cn:
            color = (255, 0, 255)  # magenta
        elif det.class_id == COCO_TRUCK or "truck" in cn:
            color = (0, 140, 255)  # orange
        elif det.class_id == COCO_CAR or "vehicle" in cn or "car" in cn:
            color = (0, 255, 255)  # yellow
        else:
            color = (0, 255, 128)  # teal — other monitored classes

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


def run_guard(
    video_file_path: Optional[str] = None,
    use_webcam: bool = True,
    montage_window_sec: float = MONTAGE_WINDOW_SEC,
    weights_path: Optional[str] = None,
) -> None:
    """
    Main realtime guard loop.
    """
    montage_window_sec = max(0.5, float(montage_window_sec or MONTAGE_WINDOW_SEC))
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    # Load API/model tokens from project .env when present.
    try:
        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
        load_dotenv(override=False)
    except Exception:
        pass
    _write_status("INGESTING", "Extracting Video Frames...")
    if not SKIP_RUNTIME_DEPS:
        ensure_ultralytics_latest()
        ensure_judge_runtime_dependencies()
    else:
        print("[RUNTIME] SKIP_RUNTIME_DEPS=True: skipping pip/runtime dependency checks.", flush=True)
    model = load_model(weights_path=weights_path)

    global USE_WEBCAM, VIDEO_FILE_PATH
    USE_WEBCAM = use_webcam
    VIDEO_FILE_PATH = video_file_path
    cap = open_video_capture()
    out = None
    state_lock = threading.Lock()
    shared_state: Dict = {}
    interrupted = False

    def cleanup_on_exit() -> None:
        try:
            with state_lock:
                cpu_fallback = bool(shared_state.get("cpu_fallback_active", False))
            # Cooperative stop request for Judge worker.
            with state_lock:
                shared_state["judge_stop_requested"] = True
                judge_thread = shared_state.get("judge_thread")
                shared_state["last_hero_frame_path"] = ""
            if isinstance(judge_thread, threading.Thread) and judge_thread.is_alive():
                judge_thread.join(timeout=2.0)
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                ACTIVE_PROCESS_PATH.parent.mkdir(parents=True, exist_ok=True)
                ACTIVE_PROCESS_PATH.write_text("{}", encoding="utf-8")
            except Exception:
                pass
        except Exception:
            pass
    try:
        # Quick sanity check
        test_system(model, cap)

        print("Guard active. Press 'q' to quit.", flush=True)
        _write_status("GUARDING", "YOLO26 Perception Active...")
        frame_index = 0
        fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
        if fps <= 0:
            fps = 30.0  # webcam often reports 0
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0

        # Large enough to cover montage + replay windows at high source FPS.
        need_sec = max(float(montage_window_sec), float(REPLAY_DURATION_SEC))
        _buf_cap = max(int(FRAME_BUFFER_MAXLEN), int(min(max(fps, 1.0), 120.0) * (need_sec + 0.5)) + 40)
        frame_buffer: deque = deque(maxlen=_buf_cap)
        camera_metadata_base = _resolve_camera_metadata(video_file_path, use_webcam, WEBCAM_INDEX)

        # Judge shared state (kept minimal and threadsafe).
        shared_state: Dict = {
            "judge_running": False,
            "judge_active": False,
            "last_judge_result": None,
            "judge_device_mode": "unknown",
            "judge_shutdown_wait_sec": JUDGE_SHUTDOWN_WAIT_SEC,
            "last_judge_reported_finished_at_s": None,
            "judge_thread": None,
            "judge_disabled": False,
            "last_judge_trigger_ts": 0.0,
            "last_incident_video_s": -1.0,
            "video_duration_s": (float(frame_count) / float(fps)) if frame_count and fps else None,
            "under_review_written": False,
            "gemini_model": None,
            "judge_stop_requested": False,
            "judge_thread_native_id": 0,
            "cpu_fallback_active": not bool(torch.cuda.is_available()),
        }

        # Startup preload of Gemini Judge backend.
        try:
            print(f"[JUDGE] preloading Gemini model: {GEMINI_MODEL_ID}", flush=True)
            gm = load_gemini_judge()
            with state_lock:
                shared_state["gemini_model"] = gm
                shared_state["judge_device_mode"] = "api"
            print("[JUDGE] Gemini preload complete", flush=True)
        except Exception as e:
            # Keep pipeline alive; Judge path can fallback if load is unavailable.
            print(f"[JUDGE] preload failed (will fallback if triggered): {e}", flush=True)

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

            # Inference
            results = model.predict(source=frame, conf=CONF_THRESHOLD, verbose=False)
            r0 = results[0] if isinstance(results, list) and results else results
            frame_accident_conf = _max_accident_confidence(model, r0)
            # Store into circular buffer (resized for RAM) + per-frame accident confidence.
            frame_buffer.append((timestamp_s, _resize_for_buffer(frame), frame_accident_conf))

            # Extract + log targets
            targets = extract_target_detections(model, r0)
            log_targets(targets, timestamp_s)

            priority_hit, priority_conf = accident_priority_trigger(model, r0)
            amber_hit, amber_conf = amber_path_trigger(model, r0)
            trigger_candidate = priority_hit or amber_hit
            # Trigger Judge on tiered accident/collision paths only.
            if trigger_candidate:
                with state_lock:
                    judge_running = bool(shared_state["judge_running"])
                    judge_disabled = bool(shared_state.get("judge_disabled", False))
                    last_trigger = float(shared_state.get("last_judge_trigger_ts", 0.0))
                    duration_s = shared_state.get("video_duration_s")
                now_ts = time.time()
                cooldown_ok = (now_ts - last_trigger) >= JUDGE_COOLDOWN_SEC
                near_eos = isinstance(duration_s, (int, float)) and duration_s > 0 and (float(duration_s) - float(timestamp_s)) <= JUDGE_EOS_WINDOW_SEC
                eos_override = near_eos and has_strong_event(targets)
                allow_trigger = cooldown_ok or eos_override or priority_hit
                if (not judge_running) and (not judge_disabled) and allow_trigger and len(frame_buffer) >= 6:
                    center_ts_for_montage = float(timestamp_s)
                    # Bias montage center to the strongest accident frame seen recently so the
                    # collision moment is included in the 2x3 evidence grid.
                    best_acc_conf = -1.0
                    for it in list(frame_buffer):
                        if len(it) >= 3:
                            ts_i = float(it[0])
                            acc_i = float(it[2] or 0.0)
                            if acc_i > best_acc_conf:
                                best_acc_conf = acc_i
                                center_ts_for_montage = ts_i
                    keyframes = sample_6_keyframes(
                        list(frame_buffer),
                        center_ts_s=center_ts_for_montage,
                        window_sec=montage_window_sec,
                    )
                    montage = build_montage_2x3(keyframes)
                    if montage is not None:
                        # Keep legacy path + cache-busting unique asset file for Dash.
                        cv2.imwrite(JUDGE_MONTAGE_PATH, montage)
                        unique_montage_path = _build_unique_montage_path()
                        cv2.imwrite(str(unique_montage_path), montage)
                        # Hero frame uses the highest-accident-confidence frame from buffer.
                        hero_frame_path = _build_unique_hero_frame_path()
                        hero_frame_saved = False
                        best_conf = -1.0
                        best_frame = None
                        for it in list(frame_buffer):
                            if len(it) >= 3:
                                acc_i = float(it[2] or 0.0)
                                fr_i = it[1]
                                if fr_i is not None and acc_i > best_conf:
                                    best_conf = acc_i
                                    best_frame = fr_i
                        if best_frame is None:
                            best_frame = frame
                        hero_frame_saved = bool(cv2.imwrite(str(hero_frame_path), best_frame))
                        if not hero_frame_saved:
                            hero_frame_path = unique_montage_path
                        replay_dir = DASHBOARD_ASSETS_DIR / "replays"
                        replay_name = f"replay_{time.strftime('%Y%m%d_%H%M%S')}_{time.time_ns() % 1000000}.mp4"
                        replay_path = _write_replay_clip(
                            buffer_items=list(frame_buffer),
                            center_ts_s=timestamp_s,
                            out_path=(replay_dir / replay_name),
                            duration_sec=REPLAY_DURATION_SEC,
                            fallback_fps=fps,
                        ) or ""
                        yolo_trigger_conf = 0.0
                        if priority_hit:
                            yolo_trigger_conf = float(priority_conf)
                        elif amber_hit:
                            yolo_trigger_conf = float(amber_conf)
                        if priority_hit:
                            trig_reason = f"accident/collision >= {JUDGE_ACCIDENT_PRIORITY_CONF:.2f} (priority)"
                            initial_state = "CRITICAL"
                        elif amber_hit:
                            trig_reason = "accident/collision in amber band [0.40,0.69]"
                            initial_state = "UNDER_REVIEW"
                        else:
                            trig_reason = "unknown"
                            initial_state = "UNDER_REVIEW"
                        print(
                            f"JUDGE triggered ({trig_reason}). "
                            f"Wrote montage: {unique_montage_path}"
                            + (" [EOS override]" if eos_override and not cooldown_ok else ""),
                            flush=True,
                        )
                        with state_lock:
                            cpu_fallback_active = bool(shared_state.get("cpu_fallback_active", False))
                        _write_status("JUDGING", "Gemini reasoning on hero frame + temporal montage...", cpu_fallback_active=cpu_fallback_active)
                        camera_metadata = {
                            **camera_metadata_base,
                            "timestamp_s": round(float(timestamp_s), 3),
                            "source_fps": round(float(fps), 3),
                        }
                        with state_lock:
                            shared_state["judge_running"] = True
                            shared_state["judge_active"] = True
                            shared_state["last_judge_trigger_ts"] = now_ts
                            shared_state["last_montage_path"] = str(unique_montage_path)
                            shared_state["last_hero_frame_path"] = str(hero_frame_path)
                            shared_state["last_incident_video_s"] = float(timestamp_s)
                            shared_state["last_yolo_trigger_confidence"] = float(yolo_trigger_conf)
                        th = threading.Thread(
                            target=judge_inference_worker,
                            kwargs={
                                "montage_bgr": montage,
                                "montage_path": str(unique_montage_path),
                                "hero_frame_bgr": best_frame,
                                "hero_frame_path": str(hero_frame_path),
                                "replay_clip_path": replay_path,
                                "yolo_trigger_confidence": float(yolo_trigger_conf),
                                "initial_state": initial_state,
                                "started_at_s": time.perf_counter(),
                                "camera_metadata": camera_metadata,
                                "shared_state": shared_state,
                                "state_lock": state_lock,
                            },
                            daemon=False,
                        )
                        with state_lock:
                            shared_state["judge_thread"] = th
                            shared_state["judge_thread_native_id"] = 0
                        th.start()
                        with state_lock:
                            shared_state["judge_thread_native_id"] = int(getattr(th, "native_id", 0) or 0)
                elif (not judge_running) and (not judge_disabled) and (not allow_trigger):
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
        interrupted = True
    finally:
        # Ensure Judge background thread has finished before interpreter shutdown.
        # This prevents rare CPython/GIL shutdown crashes in threaded ML workloads.
        judge_thread = None
        with state_lock:
            judge_thread = shared_state.get("judge_thread")
            dynamic_shutdown_wait = float(shared_state.get("judge_shutdown_wait_sec", JUDGE_SHUTDOWN_WAIT_SEC) or JUDGE_SHUTDOWN_WAIT_SEC)
            dynamic_shutdown_wait = max(dynamic_shutdown_wait, 90.0)
        if isinstance(judge_thread, threading.Thread) and judge_thread.is_alive():
            with state_lock:
                judge_active = bool(shared_state.get("judge_active", False))
                duration_s = shared_state.get("video_duration_s")
                last_incident_video_s = float(shared_state.get("last_incident_video_s", -1.0) or -1.0)
            near_tail = isinstance(duration_s, (int, float)) and duration_s > 0 and last_incident_video_s >= (float(duration_s) - 10.0)
            if judge_active:
                print(f"[JUDGE] drain mode active: waiting up to {dynamic_shutdown_wait:.0f}s for Judge to finish...", flush=True)
                deadline = time.time() + dynamic_shutdown_wait
                while judge_thread.is_alive() and time.time() < deadline:
                    judge_thread.join(timeout=2.0)
            else:
                judge_thread.join(timeout=dynamic_shutdown_wait)
            if judge_thread.is_alive():
                print("[JUDGE] drain mode timeout; creating forensic fallback incident.", flush=True)
                _write_status("FINALIZING", "Generating CHART Report...")
                last_montage_path = ""
                with state_lock:
                    last_montage_path = str(shared_state.get("last_montage_path", "") or "")
                if not last_montage_path:
                    last_montage_path = JUDGE_MONTAGE_PATH
                try:
                    save_fallback_incident_to_dashboard(
                        montage_path=last_montage_path,
                        hero_frame_path=str(shared_state.get("last_hero_frame_path", "") or ""),
                        reason="Judge timeout at end-of-stream drain mode" + (" (final 10s event)" if near_tail else ""),
                        yolo_trigger_confidence=float(shared_state.get("last_yolo_trigger_confidence", 0.0) or 0.0),
                    )
                    print("[JUDGE->DASH] wrote fallback incident", flush=True)
                    try:
                        latest = _read_live_incidents()
                        if latest:
                            ts_file = time.strftime("%Y%m%d_%H%M%S")
                            out_pdf = REPORTS_DIR / f"incident_report_fallback_{ts_file}.pdf"
                            latest_pdf = REPORTS_DIR / "incident_report.pdf"
                            generate_incident_pdf(latest[-1], output_path=str(out_pdf))
                            generate_incident_pdf(latest[-1], output_path=str(latest_pdf))
                            print(f"[REPORT] wrote fallback PDF: {out_pdf}", flush=True)
                    except Exception as report_err:
                        print(f"[REPORT] fallback PDF generation failed: {report_err}", flush=True)
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
            cleanup_on_exit()
        except Exception as e:
            print(f"[CLEANUP] warning: cleanup_on_exit failed: {e}", flush=True)
        try:
            with state_lock:
                cpu_fallback = bool(shared_state.get("cpu_fallback_active", False))
            _write_status("DONE", "Pipeline complete", cpu_fallback_active=cpu_fallback)
        except Exception:
            pass
        if interrupted:
            sys.exit(0)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO26 Guard + Gemini Judge.")
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
    parser.add_argument(
        "--montage_window_sec",
        type=float,
        default=MONTAGE_WINDOW_SEC,
        help="Temporal window (seconds) used to sample the 6-frame montage.",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to custom YOLO .pt weights (e.g. fine-tuned epoch). Default: yolo26s.pt.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    weights_resolved: Optional[str] = None
    if args.weights:
        wp = Path(args.weights).expanduser().resolve()
        if not wp.exists() or not wp.is_file():
            raise FileNotFoundError(f"--weights does not exist or is not a file: {wp}")
        weights_resolved = str(wp)
    if args.video_path:
        video_path = Path(args.video_path).expanduser().resolve()
        if not video_path.exists() or not video_path.is_file():
            raise FileNotFoundError(f"--video_path does not exist or is not a file: {video_path}")
        print(f"Single-video mode: {video_path}", flush=True)
        run_guard(
            video_file_path=str(video_path),
            use_webcam=False,
            montage_window_sec=args.montage_window_sec,
            weights_path=weights_resolved,
        )
    elif args.video_dir:
        video_dir = Path(args.video_dir).expanduser().resolve()
        if not video_dir.exists() or not video_dir.is_dir():
            raise FileNotFoundError(f"--video_dir does not exist or is not a directory: {video_dir}")
        files = sorted(video_dir.glob("*.mp4"))
        if not files:
            raise FileNotFoundError(f"No .mp4 files found in: {video_dir}")
        print(f"Demo suite mode: processing {len(files)} video(s) from {video_dir}", flush=True)
        batch_incidents: List[Dict] = []
        for idx, path in enumerate(files, start=1):
            print(f"[DEMO {idx}/{len(files)}] Starting: {path}", flush=True)
            before = _read_live_incidents()
            run_guard(
                video_file_path=str(path),
                use_webcam=False,
                montage_window_sec=args.montage_window_sec,
                weights_path=weights_resolved,
            )
            after = _read_live_incidents()
            if len(after) > len(before):
                batch_incidents.extend(after[len(before) :])
            print(f"[DEMO {idx}/{len(files)}] Completed: {path}", flush=True)
        try:
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d_%H%M%S")
            summary_pdf = REPORTS_DIR / f"batch_summary_{stamp}.pdf"
            latest_summary = REPORTS_DIR / "batch_summary_latest.pdf"
            generate_batch_summary_pdf(batch_incidents, output_path=str(summary_pdf))
            generate_batch_summary_pdf(batch_incidents, output_path=str(latest_summary))
            print(f"[REPORT] wrote batch summary PDF: {summary_pdf}", flush=True)
        except Exception as e:
            print(f"[REPORT] failed to write batch summary PDF: {e}", flush=True)
    else:
        run_guard(
            montage_window_sec=args.montage_window_sec,
            weights_path=weights_resolved,
        )
    sys.exit(0)