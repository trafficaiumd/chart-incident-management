import os
os.environ["OPENCV_LOG_LEVEL"] = "FATAL"        # <--- Add this
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"    # <--- Add this
import json
import time
import tempfile
from datetime import datetime
from collections import deque
import requests

import cv2
from dotenv import load_dotenv
from PIL import Image as PILImage
from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from ultralytics import YOLO

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Table,
    TableStyle,
    Paragraph,
    Spacer,
    Image,
    Flowable,
)

# =========================================================
# GLOBAL CONFIG
# =========================================================

load_dotenv()

MODEL_NAME = "gemini-2.5-flash"
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- CHANGE THESE PATHS ON THE SERVER ----
YOLO_MODEL_PATH = os.path.join(SCRIPT_DIR, "epoch14.pt")
LOGO_PATH = os.path.join(SCRIPT_DIR, "final logo png.png")
PROMPT_TXT_PATH = os.path.join(SCRIPT_DIR, "prompt3.txt")
GEOJSON_PATH = os.path.join(SCRIPT_DIR, "MDOT_SHA_CHART_Traffic_Cameras2.geojson")
GETCAMERAS_PATH = os.path.join(SCRIPT_DIR, "GetCameras2.json")

# Final report outputs
FINAL_JSON_PATH = os.path.join(SCRIPT_DIR, "incident_report.json")
FINAL_PDF_PATH = os.path.join(SCRIPT_DIR, "incident_report.pdf")

# =========================================================
# SHARED STYLES
# =========================================================

styles = getSampleStyleSheet()

field_style = ParagraphStyle(
    name="FieldStyle",
    fontSize=9,
    leading=12,
)

value_style = ParagraphStyle(
    name="ValueStyle",
    fontSize=9,
    leading=12,
)

# =========================================================
# BASIC HELPERS
# =========================================================

def generate_content_with_retry(model, contents, config=None, max_retries=5, delay=5):
    for attempt in range(max_retries):
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config,
            )
        except genai_errors.ServerError as e:
            if attempt == max_retries - 1:
                raise
            print(f"Gemini temporary server error ({e}). Retrying in {delay} seconds...")
            time.sleep(delay)

def load_text(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, json_path):
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def fmt(value):
    if value is None or value == "":
        return ""
    if isinstance(value, list):
        return ", ".join(str(v) for v in value) if value else ""
    if isinstance(value, dict):
        return json.dumps(value) if value else ""
    return str(value)

def to_str(value):
    return "" if value is None else str(value)

def is_unknown_value(value):
    return isinstance(value, str) and value.strip().upper() == "UNKNOWN"

def is_yes_value(value):
    if isinstance(value, bool):
        return value
    return isinstance(value, str) and value.strip().upper() == "YES"

def normalize_url(url):
    if not url:
        return ""
    url = str(url).strip().replace("http://", "https://")
    return url.rstrip("/")

def file_exists_or_raise(path, label):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")

def clean_response(text):
    text = text.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    if text.lower().startswith("json"):
        text = text[4:].strip()

    return text

def format_ts(ts):
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def get_weather_condition(lat, lon):
    lat = safe_float(lat)
    lon = safe_float(lon)

    if lat is None or lon is None:
        return ""

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "weather_code,is_day",
        "timezone": "auto",
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        current = data.get("current", {})
        code = current.get("weather_code")
        is_day = current.get("is_day", 1)

        weather_map = {
            0: "sunny" if is_day else "clear",
            1: "mainly clear",
            2: "partly cloudy",
            3: "cloudy",
            45: "foggy",
            48: "foggy",
            51: "light drizzle",
            53: "drizzle",
            55: "heavy drizzle",
            56: "freezing drizzle",
            57: "freezing drizzle",
            61: "light rain",
            63: "rainy",
            65: "heavy rain",
            66: "freezing rain",
            67: "freezing rain",
            71: "light snow",
            73: "snowy",
            75: "heavy snow",
            77: "snow grains",
            80: "light rain showers",
            81: "rain showers",
            82: "heavy rain showers",
            85: "light snow showers",
            86: "snow showers",
            95: "thunderstorm",
            96: "thunderstorm with hail",
            99: "thunderstorm with hail",
        }

        return weather_map.get(code, "unknown")
    except Exception:
        return ""


def get_city_from_lat_lon(lat, lon):
    lat = safe_float(lat)
    lon = safe_float(lon)

    if lat is None or lon is None:
        return ""

    url = "https://nominatim.openstreetmap.org/reverse"
    params = {
        "lat": lat,
        "lon": lon,
        "format": "jsonv2",
        "addressdetails": 1,
    }
    headers = {
        "User-Agent": "chart-incident-management/1.0"
    }

    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        address = data.get("address", {})

        return (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("municipality")
            or address.get("county")
            or ""
        )
    except Exception:
        return ""

# =========================================================
# LOGO HELPERS
# =========================================================

def load_logo_rgba(path):
    logo = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if logo is None:
        raise RuntimeError(f"Could not load logo: {path}")

    if len(logo.shape) == 2:
        logo = cv2.cvtColor(logo, cv2.COLOR_GRAY2BGRA)
    elif logo.shape[2] == 3:
        b, g, r = cv2.split(logo)
        alpha = 255 * (b > -1).astype("uint8")
        logo = cv2.merge((b, g, r, alpha))

    return logo

def resize_logo_for_frame(logo_rgba, frame_width, scale=0.1):
    target_w = max(1, int(frame_width * scale))
    h, w = logo_rgba.shape[:2]
    aspect = h / w
    target_h = max(1, int(target_w * aspect))
    return cv2.resize(logo_rgba, (target_w, target_h), interpolation=cv2.INTER_AREA)

def overlay_logo_top_right(frame_bgr, logo_rgba, margin=20):
    fh, fw = frame_bgr.shape[:2]
    lh, lw = logo_rgba.shape[:2]

    if lw >= fw or lh >= fh:
        return frame_bgr

    x = fw - lw - margin
    y = margin

    if x < 0 or y < 0:
        return frame_bgr

    output = frame_bgr.copy()

    logo_bgr = logo_rgba[:, :, :3]
    alpha = logo_rgba[:, :, 3] / 255.0
    alpha = alpha[:, :, None]

    roi = output[y:y + lh, x:x + lw]
    blended = (alpha * logo_bgr + (1 - alpha) * roi).astype("uint8")
    output[y:y + lh, x:x + lw] = blended

    return output

# =========================================================
# GEOJSON LOOKUP
# =========================================================

def extract_camera_record_from_feature(feature):
    props = feature.get("properties", {}) if isinstance(feature, dict) else {}
    geometry = feature.get("geometry", {}) if isinstance(feature, dict) else {}
    coords = geometry.get("coordinates", []) if isinstance(geometry, dict) else []

    lon = ""
    lat = ""
    if isinstance(coords, list) and len(coords) >= 2:
        lon = coords[0]
        lat = coords[1]

    return {
        "id": props.get("id", props.get("ID", "")),
        "name": props.get("name", props.get("location", "")),
        "description": props.get("description", props.get("location", "")),
        "publicVideoURL": props.get("publicVideoURL", props.get("CCTVPublicURL", "")),
        "lat": props.get("lat", props.get("Latitude", lat)),
        "lon": props.get("lon", props.get("Longitude", lon)),
        "routePrefix": props.get("routePrefix", ""),
        "routeNumber": props.get("routeNumber", ""),
        "routeSuffix": props.get("routeSuffix", ""),
        "milePost": props.get("milePost", ""),
        "direction": props.get("direction", ""),
        "opStatus": props.get("opStatus", ""),
        "commMode": props.get("commMode", ""),
        "cctvIp": props.get("cctvIp", ""),
        "cameraCategories": props.get("cameraCategories", []),
        "lastCachedDataUpdateTime": props.get("lastCachedDataUpdateTime", ""),
        "url": props.get("url", ""),
        "hlsurl": props.get("hlsurl", ""),
        "location": props.get("location", ""),
    }

def extract_camera_id_from_m3u8_url(stream_url):
    if not stream_url:
        return ""

    marker = "/rtplive/"
    if marker not in stream_url:
        return ""

    tail = stream_url.split(marker, 1)[1]
    return tail.split("/", 1)[0].strip()


def find_camera_in_geojson_by_id(geojson_data, camera_id):
    blank = {
        "id": "",
        "name": "",
        "description": "",
        "publicVideoURL": "",
        "lat": "",
        "lon": "",
        "routePrefix": "",
        "routeNumber": "",
        "routeSuffix": "",
        "milePost": "",
        "direction": "",
        "opStatus": "",
        "commMode": "",
        "cctvIp": "",
        "cameraCategories": [],
        "lastCachedDataUpdateTime": "",
        "url": "",
        "hlsurl": "",
        "location": "",
    }

    if not camera_id:
        return blank

    features = geojson_data.get("features", []) if isinstance(geojson_data, dict) else []

    for feature in features:
        record = extract_camera_record_from_feature(feature)
        if str(record.get("id", "")).strip() == str(camera_id).strip():
            return record

    return blank

def find_camera_in_geojson_by_url(geojson_data, source_url):
    blank = {
        "id": "",
        "name": "",
        "description": "",
        "publicVideoURL": "",
        "lat": "",
        "lon": "",
        "routePrefix": "",
        "routeNumber": "",
        "routeSuffix": "",
        "milePost": "",
        "direction": "",
        "opStatus": "",
        "commMode": "",
        "cctvIp": "",
        "cameraCategories": [],
        "lastCachedDataUpdateTime": "",
        "url": "",
        "hlsurl": "",
        "location": "",
    }

    if not source_url:
        return blank

    target = normalize_url(source_url)
    features = geojson_data.get("features", []) if isinstance(geojson_data, dict) else []

    for feature in features:
        record = extract_camera_record_from_feature(feature)
        candidates = [
            record.get("publicVideoURL", ""),
            record.get("url", ""),
            record.get("hlsurl", ""),
        ]
        normalized_candidates = [normalize_url(x) for x in candidates if x]

        if target in normalized_candidates:
            return record

        for candidate in normalized_candidates:
            if target and candidate and (target in candidate or candidate in target):
                return record

    return blank

def find_camera_in_getcameras_by_id(getcameras_data, camera_id):
    blank = {
        "routePrefix": "",
        "routeNumber": "",
        "routeSuffix": "",
        "milePost": "",
        "opStatus": "",
        "cctvIp": "",
        "cameraCategories": [],
        "known_lane_count": "",
        "direction_of_travel_interpreted": "",
        "confidence_direction": "",
    }

    if not camera_id:
        return blank

    records = []
    if isinstance(getcameras_data, list):
        records = getcameras_data
    elif isinstance(getcameras_data, dict):
        for key in ["data", "cameras", "features", "results"]:
            if isinstance(getcameras_data.get(key), list):
                records = getcameras_data[key]
                break

    for record in records:
        if not isinstance(record, dict):
            continue

        record_id = str(record.get("id", record.get("ID", ""))).strip()
        if record_id == str(camera_id).strip():
            return {
                "routePrefix": record.get("routePrefix", ""),
                "routeNumber": record.get("routeNumber", ""),
                "routeSuffix": record.get("routeSuffix", ""),
                "milePost": record.get("milePost", ""),
                "opStatus": record.get("opStatus", ""),
                "cctvIp": record.get("cctvIp", ""),
                "cameraCategories": record.get("cameraCategories", []),
                "known_lane_count": record.get("known_lane_count", ""),
                "direction_of_travel_interpreted": record.get("direction_of_travel_interpreted", ""),
                "confidence_direction": record.get("confidence_direction", ""),
            }

    return blank

# =========================================================
# PROMPTS
# =========================================================

def build_main_prompt(prompt_template, camera_info, source_video_url, raw_image_path, annotated_image_path, clip_video_path):
    metadata = {
        "cameraCategories": camera_info.get("cameraCategories", []),
        "cctvIp": camera_info.get("cctvIp", ""),
        "commMode": camera_info.get("commMode", ""),
        "description": camera_info.get("description", ""),
        "id": camera_info.get("id", ""),
        "lastCachedDataUpdateTime": to_str(camera_info.get("lastCachedDataUpdateTime")),
        "lat": camera_info.get("lat", ""),
        "lon": camera_info.get("lon", ""),
        "milePost": camera_info.get("milePost", ""),
        "name": camera_info.get("name", ""),
        "opStatus": camera_info.get("opStatus", ""),
        "publicVideoURL": camera_info.get("publicVideoURL", ""),
        "routeNumber": camera_info.get("routeNumber", ""),
        "routePrefix": camera_info.get("routePrefix", ""),
        "routeSuffix": camera_info.get("routeSuffix", ""),
    }

    extra_context = f"""

Additional context:
- Source video URL used for GeoJSON lookup: {source_video_url}
- Unannotated image path: {raw_image_path}
- Annotated image path: {annotated_image_path}
- Context clip path: {clip_video_path}

Important additional instruction:
- Use the unannotated image and the video only for incident understanding.
- Geographic/camera metadata must come only from the provided trusted metadata.
- If a trusted metadata value is unavailable, leave it blank.
- Do not invent camera/location data.
- Return ONLY valid JSON.
"""

    return (
        prompt_template.replace("__CAMERA_METADATA__", json.dumps(metadata, indent=2))
        + "\n"
        + extra_context
    )

def image_verification_prompt():
    return """
You are verifying whether a roadway camera image shows an actual traffic accident.

Rules:
- Use only the unannotated image.
- Be conservative.
- Return TRUE only if the image clearly shows an actual crash/accident, such as visible collision damage, collision position, disabled vehicles in lanes with obvious crash context, debris field, smoke/fire from crash, rollover, or other strong crash evidence.
- Return FALSE if the image does not show a clear accident.
- Return UNKNOWN if the image is too unclear, obstructed, distant, or ambiguous to verify.

Return ONLY valid JSON with this schema:
{
  "confirmed_accident": true,
  "note": ""
}

confirmed_accident must be one of:
- true
- false
- "UNKNOWN"
""".strip()

def video_verification_prompt():
    return """
You are verifying whether a roadway context video shows an actual traffic accident.

Rules:
- Use only the video.
- Be conservative.
- Return TRUE only if the video clearly shows an actual crash/accident, such as visible collision damage, disabled vehicles in lanes with obvious crash context, debris field, smoke/fire from crash, rollover, or other strong crash evidence.
- Return FALSE if the video does not show a clear accident.
- Return UNKNOWN if the video is too unclear, too short, obstructed, distant, or ambiguous to verify.

Return ONLY valid JSON with this schema:
{
  "confirmed_accident": true,
  "note": ""
}

confirmed_accident must be one of:
- true
- false
- "UNKNOWN"
""".strip()

# =========================================================
# DEFAULT OUTPUTS
# =========================================================

SEVERITY_WEIGHTS = {
    "lane_blockage": 0.20,
    "vehicle_count": 0.20,
    "vehicle_type": 0.15,
    "hazards": 0.15,
    "vehicle_orientation": 0.10,
    "damage_deformation": 0.10,
    "debris_extent": 0.10,
}

CRITICAL_UNKNOWN_FIELDS = [
    "lane_blockage",
    "vehicle_count",
    "vehicle_type",
    "hazards",
]

SEVERE_TRIGGER_FIELDS = [
    "fire_visible",
    "rollover_confirmed",
    "major_structural_collapse",
    "full_roadway_blocked",
    "large_debris_field_across_multiple_lanes",
]

def default_severity_info():
    return {
        "severity_inputs": {
            "lane_blockage": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
            "vehicle_count": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
            "vehicle_type": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
            "hazards": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
            "vehicle_orientation": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
            "damage_deformation": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
            "debris_extent": {"score_0_to_4": "UNKNOWN", "evidence": "", "confidence_0_to_1": "UNKNOWN"},
        },
        "severe_gating_triggers": {
            "fire_visible": "UNKNOWN",
            "rollover_confirmed": "UNKNOWN",
            "major_structural_collapse": "UNKNOWN",
            "full_roadway_blocked": "UNKNOWN",
            "large_debris_field_across_multiple_lanes": "UNKNOWN",
        },
        "unknown_flags": {
            "critical_unknown_count": 0,
            "should_return_unknown": "NO",
        },
        "derived_by_python": {
            "severity_score_0_to_100": None,
            "severity_category": None,
            "notes_uncertainty": "",
        },
    }

def default_output(camera_info=None):
    camera_info = camera_info or {}

    return {
        "camera": {
            "id": camera_info.get("id", ""),
            "name": camera_info.get("name", ""),
            "description": camera_info.get("description", ""),
            "publicVideoURL": camera_info.get("publicVideoURL", ""),
            "lat": camera_info.get("lat", ""),
            "lon": camera_info.get("lon", ""),
            "routePrefix": camera_info.get("routePrefix", ""),
            "routeNumber": camera_info.get("routeNumber", ""),
            "routeSuffix": camera_info.get("routeSuffix", ""),
            "milePost": camera_info.get("milePost", ""),
            "direction_of_travel_interpreted": camera_info.get("direction_of_travel_interpreted", ""),
            "confidence_direction": camera_info.get("confidence_direction", ""),
            "known_lane_count": camera_info.get("known_lane_count", ""),
            "weather_notes": "",
            "weather_live": camera_info.get("weather_live", ""),
            "city": camera_info.get("city", ""),
            "opStatus": camera_info.get("opStatus", ""),
            "commMode": camera_info.get("commMode", ""),
            "cctvIp": camera_info.get("cctvIp", ""),
            "cameraCategories": camera_info.get("cameraCategories", []),
            "lastCachedDataUpdateTime": camera_info.get("lastCachedDataUpdateTime", ""),
        },
        "incident": {
            "incident_detected": False,
            "incident_status": "no_incident",
            "incident_types": [],
            "confidence_incident": 0.0,
            "why": "",
            "actual_accident_verified_by_gemini": False,
            "actual_accident_verification_status": "",
            "actual_accident_verification_note": "",
        },
        "vehicles": {
            "count_involved": 0,
            "confidence_vehicle_count": 0.0,
            "confidence_vehicle_types": 0.0,
            "list": [],
        },
        "people": {
            "people_visible_count": 0,
            "injuries_visible": "",
            "injury_signs": [],
            "confidence_injuries": 0.0,
        },
        "hazards": {
            "fire_visible": "",
            "smoke_visible": "",
            "debris_visible": "",
            "fluid_spill_possible": "",
            "notes": "",
            "confidence_hazards": 0.0,
        },
        "lane_impact": {
            "lanes_affected": [],
            "shoulder_affected": "",
            "traffic_flow": "",
            "confidence_lane_impact": 0.0,
            "lane_impact_notes": "",
        },
        "location_in_view": {
            "in_frame_region": "",
            "distance_bucket": "",
            "relative_road_position": {
                "lane_id": "",
                "shoulder": "",
                "median": "",
            },
            "location_statement": "",
        },
        "severity_info": default_severity_info(),
        "media_inputs": {
            "unannotated_image_path": "",
            "annotated_image_path": "",
            "video_path": "",
            "source_video_url": "",
        },
        "verification_details": {
            "image_verification": {"confirmed_accident": "UNKNOWN", "note": ""},
            "video_verification": {"confirmed_accident": "UNKNOWN", "note": ""},
        },
    }

def normalize_output(model_output, camera_info, raw_image_path, annotated_image_path, clip_video_path, source_video_url):
    output = default_output(camera_info)

    for top_key, top_value in model_output.items():
        if isinstance(top_value, dict) and top_key in output and isinstance(output[top_key], dict):
            output[top_key].update(top_value)
        else:
            output[top_key] = top_value

    output["camera"]["id"] = camera_info.get("id", "")
    output["camera"]["name"] = camera_info.get("name", "")
    output["camera"]["description"] = camera_info.get("description", "")
    output["camera"]["publicVideoURL"] = camera_info.get("publicVideoURL", "")
    output["camera"]["lat"] = camera_info.get("lat", "")
    output["camera"]["lon"] = camera_info.get("lon", "")
    output["camera"]["routePrefix"] = camera_info.get("routePrefix", "")
    output["camera"]["routeNumber"] = camera_info.get("routeNumber", "")
    output["camera"]["routeSuffix"] = camera_info.get("routeSuffix", "")
    output["camera"]["milePost"] = camera_info.get("milePost", "")
    output["camera"]["direction_of_travel_interpreted"] = camera_info.get("direction_of_travel_interpreted", "")
    output["camera"]["confidence_direction"] = camera_info.get("confidence_direction", "")
    output["camera"]["known_lane_count"] = camera_info.get("known_lane_count", "")
    output["camera"]["opStatus"] = camera_info.get("opStatus", "")
    output["camera"]["commMode"] = camera_info.get("commMode", "")
    output["camera"]["cctvIp"] = camera_info.get("cctvIp", "")
    output["camera"]["cameraCategories"] = camera_info.get("cameraCategories", [])
    output["camera"]["lastCachedDataUpdateTime"] = to_str(camera_info.get("lastCachedDataUpdateTime", ""))
    output["camera"]["city"] = camera_info.get("city", "")
    output["camera"]["weather_live"] = camera_info.get("weather_live", "")

    output["incident"].setdefault("actual_accident_verified_by_gemini", False)
    output["incident"].setdefault("actual_accident_verification_status", "")
    output["incident"].setdefault("actual_accident_verification_note", "")

    output["media_inputs"] = {
        "unannotated_image_path": raw_image_path,
        "annotated_image_path": annotated_image_path,
        "video_path": clip_video_path,
        "source_video_url": source_video_url,
    }

    sev = output.setdefault("severity_info", default_severity_info())
    sev.setdefault("severity_inputs", default_severity_info()["severity_inputs"])
    sev.setdefault("severe_gating_triggers", default_severity_info()["severe_gating_triggers"])
    sev.setdefault("unknown_flags", default_severity_info()["unknown_flags"])
    sev.setdefault("derived_by_python", default_severity_info()["derived_by_python"])

    output.setdefault("verification_details", default_output()["verification_details"])

    return output

# =========================================================
# GEMINI ANALYSIS
# =========================================================

def analyze_main_incident(raw_image_path, camera_info, prompt_template, source_video_url, annotated_image_path, clip_video_path):
    prompt = build_main_prompt(
        prompt_template=prompt_template,
        camera_info=camera_info,
        source_video_url=source_video_url,
        raw_image_path=raw_image_path,
        annotated_image_path=annotated_image_path,
        clip_video_path=clip_video_path,
    )

    raw_image = PILImage.open(raw_image_path)

    response = generate_content_with_retry(
        model=MODEL_NAME,
        contents=[prompt, raw_image],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    text = clean_response(response.text)
    output = json.loads(text)

    return normalize_output(
        model_output=output,
        camera_info=camera_info,
        raw_image_path=raw_image_path,
        annotated_image_path=annotated_image_path,
        clip_video_path=clip_video_path,
        source_video_url=source_video_url,
    )

def analyze_image_accident_verification(raw_image_path):
    image = PILImage.open(raw_image_path)

    response = generate_content_with_retry(
        model=MODEL_NAME,
        contents=[image_verification_prompt(), image],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    text = clean_response(response.text)
    data = json.loads(text)

    return {
        "confirmed_accident": data.get("confirmed_accident", "UNKNOWN"),
        "note": data.get("note", ""),
    }

def upload_video_and_wait(video_path):
    video_file = client.files.upload(file=video_path)

    while getattr(video_file, "state", None) and getattr(video_file.state, "name", "") == "PROCESSING":
        time.sleep(2)
        video_file = client.files.get(name=video_file.name)

    state_name = getattr(getattr(video_file, "state", None), "name", "")
    if state_name and state_name != "ACTIVE":
        raise RuntimeError(f"Video upload failed or not active. State: {state_name}")

    return video_file

def analyze_video_accident_verification(video_path):
    video_file = upload_video_and_wait(video_path)

    response = generate_content_with_retry(
        model=MODEL_NAME,
        contents=[video_file, video_verification_prompt()],
        config=types.GenerateContentConfig(response_mime_type="application/json"),
    )

    text = clean_response(response.text)
    data = json.loads(text)

    return {
        "confirmed_accident": data.get("confirmed_accident", "UNKNOWN"),
        "note": data.get("note", ""),
    }

def normalize_verification_value(value):
    if value is True:
        return True
    if value is False:
        return False
    if isinstance(value, str) and value.strip().upper() == "UNKNOWN":
        return "UNKNOWN"
    return "UNKNOWN"

def combine_accident_verification(image_result, video_result):
    img = normalize_verification_value(image_result.get("confirmed_accident", "UNKNOWN"))
    vid = normalize_verification_value(video_result.get("confirmed_accident", "UNKNOWN"))

    if img is True and vid is True:
        return {
            "actual_accident_verified_by_gemini": True,
            "actual_accident_verification_status": "confirmed_by_both",
            "actual_accident_verification_note": (
                f"Both the unannotated image and video clip confirm a true accident. "
                f"Image note: {image_result.get('note', '')} "
                f"Video note: {video_result.get('note', '')}"
            ).strip(),
        }

    if img is False and vid is False:
        return {
            "actual_accident_verified_by_gemini": False,
            "actual_accident_verification_status": "rejected_by_both",
            "actual_accident_verification_note": (
                f"Both the unannotated image and video clip do not confirm an actual accident. "
                f"Image note: {image_result.get('note', '')} "
                f"Video note: {video_result.get('note', '')}"
            ).strip(),
        }

    issues = []
    if img is not True:
        issues.append(f"image={img}")
    if vid is not True:
        issues.append(f"video={vid}")

    return {
        "actual_accident_verified_by_gemini": False,
        "actual_accident_verification_status": "needs_review",
        "actual_accident_verification_note": (
            f"Verification mismatch or uncertainty. Manual verification is needed because "
            f"{' and '.join(issues)}. "
            f"Image note: {image_result.get('note', '')} "
            f"Video note: {video_result.get('note', '')}"
        ).strip(),
    }

# =========================================================
# SEVERITY / PDF
# =========================================================

def calculate_severity(severity_info):
    if not severity_info:
        return "UNKNOWN", "UNKNOWN", "Missing severity_info"

    severity_inputs = severity_info.get("severity_inputs", {})
    gating = severity_info.get("severe_gating_triggers", {})
    notes = []

    critical_unknown_count = 0
    for key in CRITICAL_UNKNOWN_FIELDS:
        item = severity_inputs.get(key, {})
        if is_unknown_value(item.get("score_0_to_4")):
            critical_unknown_count += 1

    should_return_unknown = critical_unknown_count >= 3
    severity_info.setdefault("unknown_flags", {})
    severity_info["unknown_flags"]["critical_unknown_count"] = critical_unknown_count
    severity_info["unknown_flags"]["should_return_unknown"] = "YES" if should_return_unknown else "NO"

    if should_return_unknown:
        notes.append(f"Returned UNKNOWN because {critical_unknown_count} critical factors are UNKNOWN.")
        severity_info.setdefault("derived_by_python", {})
        severity_info["derived_by_python"]["notes_uncertainty"] = " ".join(notes)
        return "UNKNOWN", "UNKNOWN", severity_info["derived_by_python"]["notes_uncertainty"]

    for trigger_name in SEVERE_TRIGGER_FIELDS:
        if is_yes_value(gating.get(trigger_name)):
            notes.append(f"Severe override triggered by {trigger_name}.")
            severity_info.setdefault("derived_by_python", {})
            severity_info["derived_by_python"]["notes_uncertainty"] = " ".join(notes)
            return 100, "SEVERE", severity_info["derived_by_python"]["notes_uncertainty"]

    weighted_sum = 0.0
    for key, weight in SEVERITY_WEIGHTS.items():
        item = severity_inputs.get(key, {})
        raw_score = item.get("score_0_to_4", "UNKNOWN")

        if is_unknown_value(raw_score):
            continue

        try:
            score = float(raw_score)
        except Exception:
            continue

        bounded_score = max(0.0, min(4.0, score))
        weighted_sum += weight * (bounded_score / 4.0)

    severity_score = round(100 * weighted_sum)

    if severity_score <= 29:
        category = "MINOR"
    elif severity_score <= 65:
        category = "MODERATE"
    else:
        category = "SEVERE"

    notes_text = " ".join(notes)
    severity_info.setdefault("derived_by_python", {})
    severity_info["derived_by_python"]["notes_uncertainty"] = notes_text
    return severity_score, category, notes_text

class SeverityBar(Flowable):
    def __init__(self, score, width=300, height=14):
        super().__init__()
        self.score = score
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.setStrokeColor(colors.black)
        c.rect(0, 0, self.width, self.height, stroke=1, fill=0)

        if isinstance(self.score, str):
            c.setFont("Helvetica", 8)
            c.drawString(4, 3, "UNKNOWN")
            return

        score = max(0, min(100, int(self.score)))
        fill_width = (score / 100.0) * self.width

        if score >= 66:
            c.setFillColor(colors.red)
        elif score >= 30:
            c.setFillColor(colors.orange)
        else:
            c.setFillColor(colors.green)

        c.rect(0, 0, fill_width, self.height, stroke=0, fill=1)

def make_section(title, pairs):
    rows = [
        [
            Paragraph(f"<b>{name}</b>", field_style),
            Paragraph(fmt(value), value_style),
        ]
        for name, value in pairs
    ]

    if not rows:
        return []

    table = Table(rows, colWidths=[180, 360])

    table_style = [
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]

    # Add thicker separator lines only for the Severity Criteria table
    if title == "Severity Criteria":
        for row_idx in [3, 6, 9, 12, 15, 18, 21]:
            if row_idx < len(rows):
                table_style.append(("LINEABOVE", (0, row_idx), (-1, row_idx), 1.75, colors.black))

    table.setStyle(TableStyle(table_style))

    return [
        Paragraph(f"<b>{title}</b>", styles["Heading2"]),
        table,
        Spacer(1, 12),
    ]

def generate_pdf(image_path, annotated_image_path, model_output, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    story += [
        Paragraph("<b>Crash Summary</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph("<b>Unannotated Image</b>", styles["Heading2"]),
        Spacer(1, 5),
        Image(image_path, width=5.75 * inch, height=3.35 * inch),
        Spacer(1, 11),
    ]

    if annotated_image_path and os.path.exists(annotated_image_path):
        story += [
            Paragraph("<b>Annotated Image</b>", styles["Heading2"]),
            Spacer(1, 5),
            Image(annotated_image_path, width=5.75 * inch, height=3.35 * inch),
            Spacer(1, 12),
        ]

    cam = model_output.get("camera", {})
    incident = model_output.get("incident", {})
    vehicles = model_output.get("vehicles", {})
    people = model_output.get("people", {})
    hazards = model_output.get("hazards", {})
    lane = model_output.get("lane_impact", {})
    location = model_output.get("location_in_view", {})
    rel = location.get("relative_road_position", {})
    verification_details = model_output.get("verification_details", {})

    story += make_section("Incident Information", [
        ("Incident Detected", incident.get("incident_detected", "")),
        ("Status", incident.get("incident_status", "")),
        ("Types", incident.get("incident_types", [])),
        ("Confidence", incident.get("confidence_incident", "")),
        ("Why", incident.get("why", "")),
        ("Actual Accident Verified By Gemini", incident.get("actual_accident_verified_by_gemini", False)),
        ("Accident Verification Status", incident.get("actual_accident_verification_status", "")),
        ("Accident Verification Note", incident.get("actual_accident_verification_note", "")),
    ])

    story += make_section("Verification Details", [
        ("Image Verification", verification_details.get("image_verification", {}).get("confirmed_accident", "")),
        ("Image Verification Note", verification_details.get("image_verification", {}).get("note", "")),
        ("Video Verification", verification_details.get("video_verification", {}).get("confirmed_accident", "")),
        ("Video Verification Note", verification_details.get("video_verification", {}).get("note", "")),
    ])

    sev = model_output.get("severity_info", {})
    score, category, notes = calculate_severity(sev)
    sev.setdefault("derived_by_python", {})
    sev["derived_by_python"]["severity_score_0_to_100"] = score
    sev["derived_by_python"]["severity_category"] = category
    sev["derived_by_python"]["notes_uncertainty"] = notes

    if category == "SEVERE":
        story += [
            Paragraph("<font color='red'><b>SEVERE INCIDENT</b></font>", styles["Heading1"]),
            Spacer(1, 12),
        ]
    elif category == "UNKNOWN":
        story += [
            Paragraph("<font color='gray'><b>SEVERITY UNKNOWN</b></font>", styles["Heading1"]),
            Spacer(1, 12),
        ]

    story += [
        Paragraph("<b>Severity Assessment</b>", styles["Heading2"]),
        Spacer(1, 6),
        SeverityBar(score),
        Spacer(1, 6),
        Paragraph(f"<b>Score:</b> {fmt(score)} &nbsp;&nbsp; <b>Category:</b> {fmt(category)}", styles["Normal"]),
        Spacer(1, 12),
    ]

    severity_inputs = sev.get("severity_inputs", {})
    severe_triggers = sev.get("severe_gating_triggers", {})
    unknown_flags = sev.get("unknown_flags", {})
    derived = sev.get("derived_by_python", {})

    story += make_section("Severity Criteria", [
        ("Lane Blockage Score", severity_inputs.get("lane_blockage", {}).get("score_0_to_4", "")),
        ("Lane Blockage Evidence", severity_inputs.get("lane_blockage", {}).get("evidence", "")),
        ("Lane Blockage Confidence", severity_inputs.get("lane_blockage", {}).get("confidence_0_to_1", "")),

        ("Vehicle Count Score", severity_inputs.get("vehicle_count", {}).get("score_0_to_4", "")),
        ("Vehicle Count Evidence", severity_inputs.get("vehicle_count", {}).get("evidence", "")),
        ("Vehicle Count Confidence", severity_inputs.get("vehicle_count", {}).get("confidence_0_to_1", "")),

        ("Vehicle Type Score", severity_inputs.get("vehicle_type", {}).get("score_0_to_4", "")),
        ("Vehicle Type Evidence", severity_inputs.get("vehicle_type", {}).get("evidence", "")),
        ("Vehicle Type Confidence", severity_inputs.get("vehicle_type", {}).get("confidence_0_to_1", "")),

        ("Hazards Score", severity_inputs.get("hazards", {}).get("score_0_to_4", "")),
        ("Hazards Evidence", severity_inputs.get("hazards", {}).get("evidence", "")),
        ("Hazards Confidence", severity_inputs.get("hazards", {}).get("confidence_0_to_1", "")),

        ("Vehicle Orientation Score", severity_inputs.get("vehicle_orientation", {}).get("score_0_to_4", "")),
        ("Vehicle Orientation Evidence", severity_inputs.get("vehicle_orientation", {}).get("evidence", "")),
        ("Vehicle Orientation Confidence", severity_inputs.get("vehicle_orientation", {}).get("confidence_0_to_1", "")),

        ("Damage / Deformation Score", severity_inputs.get("damage_deformation", {}).get("score_0_to_4", "")),
        ("Damage / Deformation Evidence", severity_inputs.get("damage_deformation", {}).get("evidence", "")),
        ("Damage / Deformation Confidence", severity_inputs.get("damage_deformation", {}).get("confidence_0_to_1", "")),

        ("Debris Extent Score", severity_inputs.get("debris_extent", {}).get("score_0_to_4", "")),
        ("Debris Extent Evidence", severity_inputs.get("debris_extent", {}).get("evidence", "")),
        ("Debris Extent Confidence", severity_inputs.get("debris_extent", {}).get("confidence_0_to_1", "")),

        ("Fire Visible Trigger", severe_triggers.get("fire_visible", "")),
        ("Rollover Confirmed Trigger", severe_triggers.get("rollover_confirmed", "")),
        ("Major Structural Collapse Trigger", severe_triggers.get("major_structural_collapse", "")),
        ("Full Roadway Blocked Trigger", severe_triggers.get("full_roadway_blocked", "")),
        ("Large Debris Field Across Multiple Lanes Trigger", severe_triggers.get("large_debris_field_across_multiple_lanes", "")),

        ("Critical Unknown Count", unknown_flags.get("critical_unknown_count", "")),
        ("Should Return Unknown", unknown_flags.get("should_return_unknown", "")),
        ("Severity Notes / Uncertainty", derived.get("notes_uncertainty", "")),
    ])

    story += make_section("General Information", [
        ("Camera ID", cam.get("id", "")),
        ("Camera Name", cam.get("name", "")),
        ("Description", cam.get("description", "")),
        ("Public URL", cam.get("publicVideoURL", "")),
        ("Latitude", cam.get("lat", "")),
        ("Longitude", cam.get("lon", "")),
        ("City", cam.get("city", "")),
        ("Categories", cam.get("cameraCategories", "")),
        ("Route", f'{cam.get("routePrefix", "")}{cam.get("routeNumber", "")}{cam.get("routeSuffix", "")}'),
        ("Milepost", cam.get("milePost", "")),
        ("Direction", cam.get("direction_of_travel_interpreted", "")),
        ("Direction Confidence", cam.get("confidence_direction", "")),
        ("Weather", cam.get("weather_live", "")),
        ("Lane Count", cam.get("known_lane_count", "")),
        ("Operational Status", cam.get("opStatus", "")),
        ("Camera IP", cam.get("cctvIp", "")),
    ])

    story += make_section("Vehicle Summary", [
        ("Vehicle Count", vehicles.get("count_involved", 0)),
        ("Count Confidence", vehicles.get("confidence_vehicle_count", "")),
        ("Type Confidence", vehicles.get("confidence_vehicle_types", "")),
    ])

    story += make_section("People", [
        ("People Visible", people.get("people_visible_count", 0)),
        ("Injuries", people.get("injuries_visible", "")),
        ("Signs", people.get("injury_signs", [])),
        ("Confidence", people.get("confidence_injuries", "")),
    ])

    story += make_section("Hazards", [
        ("Fire", hazards.get("fire_visible", "")),
        ("Smoke", hazards.get("smoke_visible", "")),
        ("Debris", hazards.get("debris_visible", "")),
        ("Fluid Spill", hazards.get("fluid_spill_possible", "")),
        ("Notes", hazards.get("notes", "")),
        ("Confidence", hazards.get("confidence_hazards", "")),
    ])

    story += make_section("Lane Impact", [
        ("Lanes Affected", lane.get("lanes_affected", [])),
        ("Shoulder", lane.get("shoulder_affected", "")),
        ("Traffic Flow", lane.get("traffic_flow", "")),
        ("Confidence", lane.get("confidence_lane_impact", "")),
        ("Notes", lane.get("lane_impact_notes", "")),
    ])

    story += make_section("Location in View", [
        ("Region", location.get("in_frame_region", "")),
        ("Distance", location.get("distance_bucket", "")),
        ("Lane", rel.get("lane_id", "")),
        ("Shoulder", rel.get("shoulder", "")),
        ("Median", rel.get("median", "")),
        ("Statement", location.get("location_statement", "")),
    ])

    story += make_section("Media Inputs", [
        ("Unannotated Image", model_output.get("media_inputs", {}).get("unannotated_image_path", "")),
        ("Annotated Image", model_output.get("media_inputs", {}).get("annotated_image_path", "")),
        ("Video", model_output.get("media_inputs", {}).get("video_path", "")),
    ])

    doc.build(story)

# =========================================================
# DETECTION: RECORDED VIDEO
# =========================================================

def detect_from_recorded_video(video_path, run_dir, model_path, logo_path):
    print("=" * 80)
    print("RUNNING RECORDED-VIDEO DETECTION")
    print("=" * 80)

    CONF_THRESHOLD = 0.75
    VID_STRIDE = 10 #This was 20 before
    FRAMES_BEFORE = 55
    FRAMES_AFTER = 65
    SHOW_WINDOW = False
    ACCIDENT_CLASS_NAME = "accident"
    LOGO_SCALE = 0.1
    LOGO_MARGIN = 20

    best_raw_frame_path = os.path.join(run_dir, "best_accident_frame_raw.jpg")
    best_annotated_frame_path = os.path.join(run_dir, "best_accident_frame_annotated.jpg")
    clip_path = os.path.join(run_dir, "accident_context_clip_raw.mp4")

    model = YOLO(model_path)
    class_names = model.names
    logo_rgba_original = load_logo_rgba(logo_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    logo_rgba = resize_logo_for_frame(logo_rgba_original, width, LOGO_SCALE)

    best_conf = -1.0
    best_frame_idx = None
    best_raw_frame = None
    best_annotated_frame = None

    frame_idx = 0
    last_display_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % VID_STRIDE == 0:
            results = model(frame, conf=CONF_THRESHOLD, verbose=False)
            result = results[0]
            annotated = result.plot()
            annotated_with_logo = overlay_logo_top_right(annotated, logo_rgba, LOGO_MARGIN)

            last_display_frame = annotated_with_logo.copy()
            display_frame = last_display_frame

            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    cls_name = str(class_names[cls_id]).lower()

                    if cls_name == ACCIDENT_CLASS_NAME and conf > best_conf:
                        best_conf = conf
                        best_frame_idx = frame_idx
                        best_raw_frame = frame.copy()
                        best_annotated_frame = annotated_with_logo.copy()
        else:
            if last_display_frame is not None:
                display_frame = last_display_frame
            else:
                display_frame = overlay_logo_top_right(frame, logo_rgba, LOGO_MARGIN)

        if SHOW_WINDOW:
            cv2.imshow("YOLO Detection", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    if best_frame_idx is None:
        raise RuntimeError("No accident detected in the recorded video.")

    print("Best accident detection:")
    print(f"  Confidence: {best_conf:.4f}")
    print(f"  Frame index: {best_frame_idx}")

    cv2.imwrite(best_raw_frame_path, best_raw_frame)
    cv2.imwrite(best_annotated_frame_path, best_annotated_frame)

    print(f"Saved raw best frame to: {best_raw_frame_path}")
    print(f"Saved annotated best frame to: {best_annotated_frame_path}")

    start_frame = max(0, best_frame_idx - FRAMES_BEFORE)
    end_frame = min(frame_count - 1, best_frame_idx + FRAMES_AFTER)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(clip_path, fourcc, fps, (width, height))

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not reopen video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    current_idx = start_frame

    while current_idx <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        frame_with_logo = overlay_logo_top_right(frame, logo_rgba, LOGO_MARGIN)
        writer.write(frame_with_logo)
        current_idx += 1

    cap.release()
    writer.release()

    print(f"Saved raw accident clip to: {clip_path}")

    return {
        "raw_image_path": best_raw_frame_path,
        "annotated_image_path": best_annotated_frame_path,
        "clip_video_path": clip_path,
    }

# =========================================================
# DETECTION: LIVE STREAM
# =========================================================

def detect_from_live_stream(stream_url, run_dir, model_path, logo_path):
    print("=" * 80)
    print("RUNNING LIVE-STREAM DETECTION")
    print("=" * 80)

    ACCIDENT_CONF_THRESHOLD = 0.90
    VEHICLE_CONF_THRESHOLD = 0.65
    REQUIRED_DETECTIONS = 5
    DETECTION_WINDOW_SEC = 5
    VID_STRIDE = 5 #This was 4 before
    SHOW_WINDOW = False
    PRE_EVENT_SEC = 5
    POST_EVENT_SEC = 5
    BUFFER_SEC = PRE_EVENT_SEC + 2
    COOLDOWN_SEC = 15
    ACCIDENT_CLASS_NAME = "accident"
    VEHICLE_CLASS_NAME = "vehicle"
    MIN_DETECTION_GAP_SEC = 0.5
    LOGO_SCALE = 0.1
    LOGO_MARGIN = 20

    best_raw_frame_path = os.path.join(run_dir, "best_accident_frame_raw.jpg")
    best_annotated_frame_path = os.path.join(run_dir, "best_accident_frame_annotated.jpg")
    clip_path = os.path.join(run_dir, "accident_context_clip_raw.mp4")

    def prune_old_detections(recent_detections, now_ts):
        while recent_detections and (now_ts - recent_detections[0][0] > DETECTION_WINDOW_SEC):
            recent_detections.popleft()

    def save_event_clip(pre_frames, cap_obj, fps_value, out_w, out_h, save_path, post_sec, logo_rgba):
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps_value, (out_w, out_h))

        for _, buffered_frame in pre_frames:
            frame_out = overlay_logo_top_right(buffered_frame.copy(), logo_rgba, LOGO_MARGIN)
            writer.write(frame_out)

        post_frames_to_capture = int(post_sec * fps_value)
        captured = 0

        while captured < post_frames_to_capture:
            ret, frame = cap_obj.read()
            if not ret:
                print("Warning: stream ended or frame read failed while capturing post-event clip.")
                break

            if frame.shape[1] != out_w or frame.shape[0] != out_h:
                frame = cv2.resize(frame, (out_w, out_h))

            frame_out = overlay_logo_top_right(frame.copy(), logo_rgba, LOGO_MARGIN)
            writer.write(frame_out)
            captured += 1

        writer.release()

    model = YOLO(model_path)
    class_names = model.names
    print("Model classes:", class_names)

    logo_rgba_original = load_logo_rgba(logo_path)

    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open stream: {stream_url}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps > 120:
        fps = 30.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if width <= 0 or height <= 0:
        width, height = 1280, 720

    print(f"Using FPS={fps}, size=({width}, {height})")

    logo_rgba = resize_logo_for_frame(logo_rgba_original, width, LOGO_SCALE)

    max_buffer_frames = int(BUFFER_SEC * fps) + 5
    frame_buffer = deque(maxlen=max_buffer_frames)
    recent_detections = deque()

    frame_idx = 0
    last_display_frame = None
    last_saved_time = 0
    last_detection_time = 0

    best_conf = -1.0
    best_raw_frame = None
    best_annotated_frame = None

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame read failed. Waiting briefly and trying again...")
            time.sleep(1)
            cap.release()
            cap = cv2.VideoCapture(stream_url)
            if not cap.isOpened():
                print("Reconnection failed. Retrying...")
                continue
            print("Reconnected to stream.")
            continue

        now_ts = time.time()

        if frame.shape[1] != width or frame.shape[0] != height:
            frame = cv2.resize(frame, (width, height))

        frame_buffer.append((now_ts, frame.copy()))
        display_frame = frame

        if frame_idx % VID_STRIDE == 0:
            results = model(frame, conf=min(ACCIDENT_CONF_THRESHOLD, VEHICLE_CONF_THRESHOLD), verbose=False)
            result = results[0]
            annotated = frame.copy()

            found_accident_this_frame = False
            best_accident_conf_this_frame = -1.0

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    cls_name = str(class_names[cls_id]).lower()

                    x1, y1, x2, y2 = map(int, boxes.xyxy[i].tolist())

                    draw_this_box = False
                    label = None
                    color = (0, 255, 0)

                    if cls_name == VEHICLE_CLASS_NAME and conf >= VEHICLE_CONF_THRESHOLD:
                        draw_this_box = True
                        label = f"{cls_name} {conf:.2f}"
                        color = (255, 90, 0)

                    elif cls_name == ACCIDENT_CLASS_NAME and conf >= ACCIDENT_CONF_THRESHOLD:
                        draw_this_box = True
                        label = f"{cls_name} {conf:.2f}"
                        color = (0, 30, 255)

                        found_accident_this_frame = True
                        best_accident_conf_this_frame = max(best_accident_conf_this_frame, conf)

                    if draw_this_box:
                        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(
                            annotated,
                            label,
                            (x1, max(y1 - 10, 20)),
                            cv2.FONT_HERSHEY_TRIPLEX,
                            0.6,
                            color,
                            1
                        )

            annotated = overlay_logo_top_right(annotated, logo_rgba, LOGO_MARGIN)
            last_display_frame = annotated.copy()
            display_frame = last_display_frame

            if found_accident_this_frame:
                if (now_ts - last_detection_time) >= MIN_DETECTION_GAP_SEC:
                    recent_detections.append((now_ts, best_accident_conf_this_frame))
                    last_detection_time = now_ts

                    print(
                        f"[DETECTION] accident conf={best_accident_conf_this_frame:.3f} "
                        f"| at {format_ts(now_ts)} "
                        f"| detections in window={len(recent_detections)}"
                    )

                    if best_accident_conf_this_frame > best_conf:
                        best_conf = best_accident_conf_this_frame
                        best_raw_frame = frame.copy()
                        best_annotated_frame = annotated.copy()

            prune_old_detections(recent_detections, now_ts)
            in_cooldown = (now_ts - last_saved_time) < COOLDOWN_SEC

            if not in_cooldown and len(recent_detections) >= REQUIRED_DETECTIONS:
                confirmation_ts = time.time()
                pre_frames = [(ts, fr) for ts, fr in frame_buffer if confirmation_ts - ts <= PRE_EVENT_SEC]

                print("\n" + "=" * 70)
                print("[EVENT CONFIRMED]")
                print(f"Confirmation time: {format_ts(confirmation_ts)}")
                print(f"Rule met: {REQUIRED_DETECTIONS} accident detections within {DETECTION_WINDOW_SEC} seconds")
                print("Triggering detections:")
                for det_num, (det_ts, det_conf) in enumerate(recent_detections, start=1):
                    print(f"  {det_num}. time={format_ts(det_ts)} | conf={det_conf:.3f}")

                save_event_clip(
                    pre_frames=pre_frames,
                    cap_obj=cap,
                    fps_value=fps,
                    out_w=width,
                    out_h=height,
                    save_path=clip_path,
                    post_sec=POST_EVENT_SEC,
                    logo_rgba=logo_rgba
                )

                if best_raw_frame is None:
                    best_raw_frame = pre_frames[-1][1].copy() if pre_frames else frame.copy()
                if best_annotated_frame is None:
                    best_annotated_frame = display_frame.copy()

                cv2.imwrite(best_raw_frame_path, best_raw_frame)
                cv2.imwrite(best_annotated_frame_path, best_annotated_frame)

                print(f"Saved raw best frame to: {best_raw_frame_path}")
                print(f"Saved annotated best frame to: {best_annotated_frame_path}")
                print(f"Saved clip: {clip_path}")
                print("=" * 70 + "\n")

                cap.release()
                cv2.destroyAllWindows()

                return {
                    "raw_image_path": best_raw_frame_path,
                    "annotated_image_path": best_annotated_frame_path,
                    "clip_video_path": clip_path,
                }

        else:
            if last_display_frame is not None:
                display_frame = last_display_frame
            else:
                display_frame = overlay_logo_top_right(display_frame, logo_rgba, LOGO_MARGIN)

        if SHOW_WINDOW:
            cv2.imshow("Live YOLO Stream", display_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise RuntimeError("Live-stream detection stopped by user before event confirmation.")

        frame_idx += 1

# =========================================================
# REPORT PIPELINE
# =========================================================

def run_report_pipeline(raw_image_path, annotated_image_path, clip_video_path, source_video_url, output_json_path, output_pdf_path):
    print("=" * 80)
    print("RUNNING AI IMAGE + VIDEO ASSESSMENT WITH GEOJSON LOOKUP")
    print("=" * 80)

    file_exists_or_raise(PROMPT_TXT_PATH, "Prompt file")
    file_exists_or_raise(raw_image_path, "Unannotated image")
    file_exists_or_raise(annotated_image_path, "Annotated image")
    file_exists_or_raise(clip_video_path, "Clip video")
    file_exists_or_raise(GEOJSON_PATH, "GeoJSON file")
    file_exists_or_raise(GETCAMERAS_PATH, "GetCameras file")

    print("\n1. Loading prompt...")
    prompt_template = load_text(PROMPT_TXT_PATH)
    print("   ✓ Prompt loaded")

    print("\n2. Loading GeoJSON...")
    geojson_data = load_json(GEOJSON_PATH)
    print("   ✓ GeoJSON loaded")

    print("\n2b. Loading GetCameras.json...")
    getcameras_data = load_json(GETCAMERAS_PATH)
    print("   ✓ GetCameras.json loaded")

    print("\n3. Looking up camera...")
    camera_id = extract_camera_id_from_m3u8_url(source_video_url)

    if camera_id:
        print(f"   Extracted camera id from stream URL: {camera_id}")
        camera_info = find_camera_in_geojson_by_id(geojson_data, camera_id)
    else:
        print("   No camera id found in stream URL, falling back to URL lookup...")
        camera_info = find_camera_in_geojson_by_url(geojson_data, source_video_url)

    print("   ✓ Camera lookup complete")
    print(f"   Matched camera id: {camera_info.get('id', '')}")
    print(f"   Matched camera name: {camera_info.get('name', '')}")

    # Pull the remaining table fields from GetCameras.json using the same camera id
    extra_camera_info = find_camera_in_getcameras_by_id(getcameras_data, camera_info.get("id", ""))

    camera_info["routePrefix"] = extra_camera_info.get("routePrefix", "")
    camera_info["routeNumber"] = extra_camera_info.get("routeNumber", "")
    camera_info["routeSuffix"] = extra_camera_info.get("routeSuffix", "")
    camera_info["milePost"] = extra_camera_info.get("milePost", "")
    camera_info["opStatus"] = extra_camera_info.get("opStatus", "")
    camera_info["cctvIp"] = extra_camera_info.get("cctvIp", "")
    camera_info["cameraCategories"] = extra_camera_info.get("cameraCategories", [])
    camera_info["known_lane_count"] = extra_camera_info.get("known_lane_count", "")
    camera_info["direction_of_travel_interpreted"] = extra_camera_info.get("direction_of_travel_interpreted", "")
    camera_info["confidence_direction"] = extra_camera_info.get("confidence_direction", "")

    # City and weather are filled separately
    camera_info["city"] = get_city_from_lat_lon(camera_info.get("lat"), camera_info.get("lon"))
    camera_info["weather_live"] = get_weather_condition(camera_info.get("lat"), camera_info.get("lon"))

    # Keep Public URL shown in the report as the actual input stream/url
    camera_info["publicVideoURL"] = source_video_url

    print("\n4. Calling Gemini on unannotated image for accident verification...")
    image_verification = analyze_image_accident_verification(raw_image_path)
    print("   ✓ Image verification complete")

    print("\n5. Calling Gemini on video clip for accident verification...")
    video_verification = analyze_video_accident_verification(clip_video_path)
    print("   ✓ Video verification complete")

    print("\n6. Running main incident analysis on unannotated image...")
    model_output = analyze_main_incident(
        raw_image_path=raw_image_path,
        camera_info=camera_info,
        prompt_template=prompt_template,
        source_video_url=source_video_url,
        annotated_image_path=annotated_image_path,
        clip_video_path=clip_video_path,
    )
    print("   ✓ Main Gemini analysis complete")

    verification_summary = combine_accident_verification(
        image_result=image_verification,
        video_result=video_verification,
    )

    model_output["incident"]["actual_accident_verified_by_gemini"] = verification_summary["actual_accident_verified_by_gemini"]
    model_output["incident"]["actual_accident_verification_status"] = verification_summary["actual_accident_verification_status"]
    model_output["incident"]["actual_accident_verification_note"] = verification_summary["actual_accident_verification_note"]

    model_output["verification_details"] = {
        "image_verification": image_verification,
        "video_verification": video_verification,
    }

    print("\n7. Saving JSON...")
    save_json(model_output, output_json_path)
    print(f"   ✓ JSON saved: {output_json_path}")

    print("\n8. Generating PDF...")
    generate_pdf(
        image_path=raw_image_path,
        annotated_image_path=annotated_image_path,
        model_output=model_output,
        output_path=output_pdf_path,
    )
    print(f"   ✓ PDF generated: {output_pdf_path}")

# =========================================================
# MAIN
# =========================================================

def main():
    print("=" * 80)
    print("ACCIDENT DETECTION + REPORT GENERATION")
    print("=" * 80)

    file_exists_or_raise(YOLO_MODEL_PATH, "YOLO model")
    file_exists_or_raise(LOGO_PATH, "Logo")
    file_exists_or_raise(PROMPT_TXT_PATH, "Prompt file")
    file_exists_or_raise(GEOJSON_PATH, "GeoJSON file")

    print("\nChoose input type:")
    print("1. Recorded video file")
    print("2. Live stream (.m3u8 URL)")
    choice = input("Enter 1 or 2: ").strip()

    if choice not in {"1", "2"}:
        raise ValueError("Invalid choice. Please run again and enter 1 or 2.")

    if choice == "1":
        source_path_or_url = input("\nEnter full path to recorded video file: ").strip()
        source_video_url = input(
            "Enter the CHART camera/public video URL for GeoJSON lookup (or press Enter to leave blank): "
        ).strip()

        run_dir = tempfile.mkdtemp(prefix="accident_run_video_", dir=SCRIPT_DIR)
        print(f"\nWorking folder: {run_dir}")

        detection_outputs = detect_from_recorded_video(
            video_path=source_path_or_url,
            run_dir=run_dir,
            model_path=YOLO_MODEL_PATH,
            logo_path=LOGO_PATH,
        )

    else:
        source_path_or_url = input("\nEnter .m3u8 live stream URL: ").strip()
        source_video_url = source_path_or_url

        run_dir = tempfile.mkdtemp(prefix="accident_run_live_", dir=SCRIPT_DIR)
        print(f"\nWorking folder: {run_dir}")

        detection_outputs = detect_from_live_stream(
            stream_url=source_path_or_url,
            run_dir=run_dir,
            model_path=YOLO_MODEL_PATH,
            logo_path=LOGO_PATH,
        )

    run_report_pipeline(
        raw_image_path=detection_outputs["raw_image_path"],
        annotated_image_path=detection_outputs["annotated_image_path"],
        clip_video_path=detection_outputs["clip_video_path"],
        source_video_url=source_video_url,
        output_json_path=FINAL_JSON_PATH,
        output_pdf_path=FINAL_PDF_PATH,
    )

    print("\nDone.")
    print(f"Final JSON: {FINAL_JSON_PATH}")
    print(f"Final PDF:  {FINAL_PDF_PATH}")

if __name__ == "__main__":
    main()