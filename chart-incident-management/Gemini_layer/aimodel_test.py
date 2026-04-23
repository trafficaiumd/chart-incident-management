import os
import json
import time
import cv2

from dotenv import load_dotenv
from PIL import Image as PILImage
from google import genai
from google.genai import types

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

# =========================
# CONFIG
# =========================

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
MODEL_NAME = "gemini-2.5-flash"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

PROMPT_TXT_PATH = os.path.join(SCRIPT_DIR, "prompt.txt")
OUTPUT_JSON_PATH = os.path.join(SCRIPT_DIR, "incident_report.json")
OUTPUT_PDF_PATH = os.path.join(SCRIPT_DIR, "incident_report.pdf")

# Outputs from Detection From Video.py
RAW_IMAGE_PATH = os.path.join(SCRIPT_DIR, "best_accident_frame_raw.jpg")
ANNOTATED_IMAGE_PATH = os.path.join(SCRIPT_DIR, "best_accident_frame_annotated.jpg")
CLIP_VIDEO_PATH = os.path.join(SCRIPT_DIR, "accident_context_clip_raw.mp4")

# GeoJSON containing camera metadata
GEOJSON_PATH = os.path.join(SCRIPT_DIR, "MDOT_SHA_CHART_Traffic_Cameras.geojson")

# Provide the source camera/video URL here.
# This is used to find the matching camera record in the GeoJSON.
SOURCE_VIDEO_URL = ""

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

# =========================
# STYLES
# =========================

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

# =========================
# BASIC HELPERS
# =========================

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


def safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return default


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
    url = str(url).strip()
    url = url.replace("http://", "https://")
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


# =========================
# GEOJSON LOOKUP
# =========================

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


# =========================
# PROMPTS
# =========================

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


# =========================
# OUTPUT DEFAULTS
# =========================

def default_severity_info():
    return {
        "severity_inputs": {
            "lane_blockage": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
            "vehicle_count": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
            "vehicle_type": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
            "hazards": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
            "vehicle_orientation": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
            "damage_deformation": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
            "debris_extent": {
                "score_0_to_4": "UNKNOWN",
                "evidence": "",
                "confidence_0_to_1": "UNKNOWN",
            },
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
            "direction_of_travel_interpreted": "",
            "confidence_direction": "",
            "weather_notes": "",
            "known_lane_count": "",
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
            "image_verification": {
                "confirmed_accident": "UNKNOWN",
                "note": "",
            },
            "video_verification": {
                "confirmed_accident": "UNKNOWN",
                "note": "",
            },
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
    output["camera"]["opStatus"] = camera_info.get("opStatus", "")
    output["camera"]["commMode"] = camera_info.get("commMode", "")
    output["camera"]["cctvIp"] = camera_info.get("cctvIp", "")
    output["camera"]["cameraCategories"] = camera_info.get("cameraCategories", [])
    output["camera"]["lastCachedDataUpdateTime"] = to_str(camera_info.get("lastCachedDataUpdateTime", ""))

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


# =========================
# GEMINI ANALYSIS
# =========================

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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, raw_image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[image_verification_prompt(), image],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
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

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[video_file, video_verification_prompt()],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
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


# =========================
# SEVERITY
# =========================

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
        notes.append(
            f"Returned UNKNOWN because {critical_unknown_count} critical factors are UNKNOWN."
        )
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


# =========================
# PDF
# =========================

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
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    return [
        Paragraph(f"<b>{title}</b>", styles["Heading2"]),
        table,
        Spacer(1, 12),
    ]


def generate_pdf(image_path, annotated_image_path, model_output, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    story += [
        Paragraph("<b>EVENT SUMMARY</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph("<b>Unannotated Image</b>", styles["Heading2"]),
        Spacer(1, 6),
        Image(image_path, width=6 * inch, height=3.5 * inch),
        Spacer(1, 12),
    ]

    if annotated_image_path and os.path.exists(annotated_image_path):
        story += [
            Paragraph("<b>Annotated Image</b>", styles["Heading2"]),
            Spacer(1, 6),
            Image(annotated_image_path, width=6 * inch, height=3.5 * inch),
            Spacer(1, 12),
        ]

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

    cam = model_output.get("camera", {})
    incident = model_output.get("incident", {})
    vehicles = model_output.get("vehicles", {})
    people = model_output.get("people", {})
    hazards = model_output.get("hazards", {})
    lane = model_output.get("lane_impact", {})
    location = model_output.get("location_in_view", {})
    rel = location.get("relative_road_position", {})
    verification_details = model_output.get("verification_details", {})

    story += make_section("General Information", [
        ("Camera ID", cam.get("id", "")),
        ("Camera Name", cam.get("name", "")),
        ("Description", cam.get("description", "")),
        ("Public URL", cam.get("publicVideoURL", "")),
        ("Latitude", cam.get("lat", "")),
        ("Longitude", cam.get("lon", "")),
        ("Route", f'{cam.get("routePrefix", "")}{cam.get("routeNumber", "")}{cam.get("routeSuffix", "")}'),
        ("Milepost", cam.get("milePost", "")),
        ("Direction", cam.get("direction_of_travel_interpreted", "")),
        ("Direction Confidence", cam.get("confidence_direction", "")),
        ("Weather", cam.get("weather_notes", "")),
        ("Lane Count", cam.get("known_lane_count", "")),
        ("Operational Status", cam.get("opStatus", "")),
        ("Comm Mode", cam.get("commMode", "")),
        ("Camera IP", cam.get("cctvIp", "")),
        ("Categories", cam.get("cameraCategories", [])),
        ("Last Update", cam.get("lastCachedDataUpdateTime", "")),
    ])

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
        ("Source Video URL", model_output.get("media_inputs", {}).get("source_video_url", "")),
    ])

    doc.build(story)


# =========================
# MAIN
# =========================

def run_with_detection_outputs():
    print("=" * 80)
    print("RUNNING AI IMAGE + VIDEO ASSESSMENT WITH GEOJSON LOOKUP")
    print("=" * 80)

    file_exists_or_raise(PROMPT_TXT_PATH, "Prompt file")
    file_exists_or_raise(RAW_IMAGE_PATH, "Unannotated image")
    file_exists_or_raise(ANNOTATED_IMAGE_PATH, "Annotated image")
    file_exists_or_raise(CLIP_VIDEO_PATH, "Clip video")
    file_exists_or_raise(GEOJSON_PATH, "GeoJSON file")

    print("\n1. Loading prompt...")
    prompt_template = load_text(PROMPT_TXT_PATH)
    print("   ✓ Prompt loaded")

    print("\n2. Loading GeoJSON...")
    geojson_data = load_json(GEOJSON_PATH)
    print("   ✓ GeoJSON loaded")

    print("\n3. Looking up camera by source URL...")
    camera_info = find_camera_in_geojson_by_url(geojson_data, SOURCE_VIDEO_URL)
    print("   ✓ Camera lookup complete")
    print(f"   Matched camera id: {camera_info.get('id', '')}")
    print(f"   Matched camera name: {camera_info.get('name', '')}")

    print("\n4. Calling Gemini on unannotated image for accident verification...")
    image_verification = analyze_image_accident_verification(RAW_IMAGE_PATH)
    print("   ✓ Image verification complete")

    print("\n5. Calling Gemini on video clip for accident verification...")
    video_verification = analyze_video_accident_verification(CLIP_VIDEO_PATH)
    print("   ✓ Video verification complete")

    print("\n6. Running main incident analysis on unannotated image...")
    model_output = analyze_main_incident(
        raw_image_path=RAW_IMAGE_PATH,
        camera_info=camera_info,
        prompt_template=prompt_template,
        source_video_url=SOURCE_VIDEO_URL,
        annotated_image_path=ANNOTATED_IMAGE_PATH,
        clip_video_path=CLIP_VIDEO_PATH,
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
    save_json(model_output, OUTPUT_JSON_PATH)
    print(f"   ✓ JSON saved: {OUTPUT_JSON_PATH}")

    print("\n8. Generating PDF...")
    generate_pdf(
        image_path=RAW_IMAGE_PATH,
        annotated_image_path=ANNOTATED_IMAGE_PATH,
        model_output=model_output,
        output_path=OUTPUT_PDF_PATH,
    )
    print(f"   ✓ PDF generated: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    run_with_detection_outputs()