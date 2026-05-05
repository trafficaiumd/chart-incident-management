from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from PIL import Image as PILImage
from google import genai
from google.genai import types
from google.genai.errors import APIError
from yolo_ai_layer.api_enrichment import get_weather_condition, get_city_from_lat_lon
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Flowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def retry_with_backoff(func, max_retries=3, base_delay=2):
    """Retries a function with exponential backoff if a 503 or 429 error occurs."""
    for attempt in range(max_retries + 1):
        try:
            return func()
        except APIError as e:
            if e.code in [503, 429] and attempt < max_retries:
                sleep_time = base_delay * (2 ** attempt)
                print(
                    f"API busy (Error {e.code}). Retrying in {sleep_time} seconds (Attempt {attempt + 1}/{max_retries})..."
                )
                time.sleep(sleep_time)
            else:
                raise e


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TEMP_DIR = PROJECT_ROOT / "data" / "temp"
PROMPT_FILE = PROJECT_ROOT / "live_camera" / "prompt.txt"
DEFAULT_MODEL = "gemini-2.5-flash"
GEOJSON_PATH = PROJECT_ROOT / "live_camera" / "MDOT_SHA_CHART_Traffic_Cameras.geojson"

styles = getSampleStyleSheet()
field_style = ParagraphStyle(name="FieldStyle", fontSize=9, leading=12)
value_style = ParagraphStyle(name="ValueStyle", fontSize=9, leading=12)

SEVERITY_WEIGHTS = {
    "lane_blockage": 0.20,
    "vehicle_count": 0.20,
    "vehicle_type": 0.15,
    "hazards": 0.15,
    "vehicle_orientation": 0.10,
    "damage_deformation": 0.10,
    "debris_extent": 0.10,
}
CRITICAL_UNKNOWN_FIELDS = ["lane_blockage", "vehicle_count", "vehicle_type", "hazards"]
SEVERE_TRIGGER_FIELDS = [
    "fire_visible",
    "rollover_confirmed",
    "major_structural_collapse",
    "full_roadway_blocked",
    "large_debris_field_across_multiple_lanes",
]


def _clean_parse_json(text: str) -> Dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw = "\n".join(lines).strip()
    if raw.lower().startswith("json"):
        raw = raw[4:].strip()
    start = raw.find("{")
    end = raw.rfind("}")
    if start < 0 or end < 0 or end < start:
        return {}
    try:
        return json.loads(raw[start : end + 1])
    except Exception:
        return {}


def _normalize_url(url: str) -> str:
    if not url:
        return ""
    return str(url).strip().replace("http://", "https://").rstrip("/")


def extract_camera_record_from_feature(feature: Dict[str, Any]) -> Dict[str, Any]:
    props = feature.get("properties", {}) if isinstance(feature, dict) else {}
    geom = feature.get("geometry", {}) if isinstance(feature, dict) else {}
    coords = geom.get("coordinates", []) if isinstance(geom, dict) else []
    lon = coords[0] if isinstance(coords, list) and len(coords) > 0 else ""
    lat = coords[1] if isinstance(coords, list) and len(coords) > 1 else ""
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
    }


def find_camera_in_geojson_by_url(geojson_data: Dict[str, Any], source_url: str) -> Dict[str, Any]:
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
    }
    if not source_url:
        return blank
    target = _normalize_url(source_url)
    for feature in geojson_data.get("features", []) if isinstance(geojson_data, dict) else []:
        rec = extract_camera_record_from_feature(feature)
        for c in [_normalize_url(rec.get("publicVideoURL", "")), _normalize_url(rec.get("name", ""))]:
            if c and (c == target or target in c or c in target):
                return rec
    return blank


def image_verification_prompt() -> str:
    return """You are an expert traffic incident analyst verifying an unannotated roadway image.
Be highly analytical. Look for physics and visual anomalies: vehicles at unnatural angles, disabled vehicles on shoulders, debris, smoke, or sudden unnatural bottlenecks. 
If there are strong contextual clues of an incident, even if distant or low resolution, return true.
Return ONLY valid JSON:
{"confirmed_accident": true, "note": "Brief explanation of what you see"}
"""


def video_verification_prompt() -> str:
    return """You are an expert traffic incident analyst verifying an unannotated roadway video clip.
Be highly analytical. Watch the kinematics of the vehicles across time. Look for sudden stops, vehicles resting at perpendicular angles, hazards, or people outside their cars.
If you observe these anomalies, return true.
Return ONLY valid JSON:
{"confirmed_accident": true, "note": "Brief explanation of what you see"}
"""


def analyze_image_accident_verification(client: genai.Client, model_name: str, raw_image_path: Path) -> Dict[str, Any]:
    image = PILImage.open(raw_image_path)
    response = retry_with_backoff(
        lambda: client.models.generate_content(
            model=model_name,
            contents=[image_verification_prompt(), image],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    )
    data = _clean_parse_json(getattr(response, "text", ""))
    return {"confirmed_accident": data.get("confirmed_accident", "UNKNOWN"), "note": data.get("note", "")}


def upload_video_and_wait(client: genai.Client, video_path: Path):
    video_file = retry_with_backoff(lambda: client.files.upload(file=str(video_path)))
    while getattr(video_file, "state", None) and getattr(video_file.state, "name", "") == "PROCESSING":
        time.sleep(2)
        video_file = retry_with_backoff(lambda: client.files.get(name=video_file.name))
    state_name = getattr(getattr(video_file, "state", None), "name", "")
    if state_name and state_name != "ACTIVE":
        raise RuntimeError(f"Video upload failed or not active. State: {state_name}")
    return video_file


def analyze_video_accident_verification(client: genai.Client, model_name: str, video_path: Path) -> Dict[str, Any]:
    video_file = upload_video_and_wait(client, video_path)
    response = retry_with_backoff(
        lambda: client.models.generate_content(
            model=model_name,
            contents=[video_file, video_verification_prompt()],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    )
    data = _clean_parse_json(getattr(response, "text", ""))
    return {"confirmed_accident": data.get("confirmed_accident", "UNKNOWN"), "note": data.get("note", "")}


def combine_accident_verification(image_result: Dict[str, Any], video_result: Dict[str, Any]) -> Dict[str, Any]:
    def norm(v):
        if v is True:
            return True
        if v is False:
            return False
        if isinstance(v, str) and v.strip().upper() == "UNKNOWN":
            return "UNKNOWN"
        return "UNKNOWN"

    img = norm(image_result.get("confirmed_accident", "UNKNOWN"))
    vid = norm(video_result.get("confirmed_accident", "UNKNOWN"))

    if img is True and vid is True:
        return {
            "actual_accident_verified_by_gemini": True,
            "actual_accident_verification_status": "confirmed_by_both",
            "actual_accident_verification_note": f"Image note: {image_result.get('note', '')} Video note: {video_result.get('note', '')}",
        }
    if img is False and vid is False:
        return {
            "actual_accident_verified_by_gemini": False,
            "actual_accident_verification_status": "rejected_by_both",
            "actual_accident_verification_note": f"Image note: {image_result.get('note', '')} Video note: {video_result.get('note', '')}",
        }

    issues = []
    if img is not True:
        issues.append(f"image={img}")
    if vid is not True:
        issues.append(f"video={vid}")

    return {
        "actual_accident_verified_by_gemini": False,
        "actual_accident_verification_status": "needs_review",
        "actual_accident_verification_note": f"Verification mismatch. Image note: {image_result.get('note', '')} Video note: {video_result.get('note', '')}",
    }


def calculate_severity(severity_info: Dict[str, Any]) -> tuple[Any, str, str]:
    if not severity_info:
        return "UNKNOWN", "UNKNOWN", "Missing severity_info"
    inputs = severity_info.get("severity_inputs", {})
    gating = severity_info.get("severe_gating_triggers", {})

    unknowns = 0
    for key in CRITICAL_UNKNOWN_FIELDS:
        if str(inputs.get(key, {}).get("score_0_to_4", "UNKNOWN")).upper() == "UNKNOWN":
            unknowns += 1

    severity_info.setdefault("unknown_flags", {})
    severity_info["unknown_flags"]["critical_unknown_count"] = unknowns
    severity_info["unknown_flags"]["should_return_unknown"] = "YES" if unknowns >= 3 else "NO"

    if unknowns >= 3:
        return "UNKNOWN", "UNKNOWN", f"Returned UNKNOWN because {unknowns} critical factors are UNKNOWN."

    for key in SEVERE_TRIGGER_FIELDS:
        val = gating.get(key, "NO")
        if val is True or str(val).upper() == "YES":
            return 100, "SEVERE", f"Override by {key}"

    weighted_sum = 0.0
    for key, w in SEVERITY_WEIGHTS.items():
        raw = inputs.get(key, {}).get("score_0_to_4", "UNKNOWN")
        if str(raw).upper() == "UNKNOWN":
            continue
        try:
            s = max(0.0, min(4.0, float(raw)))
            weighted_sum += w * (s / 4.0)
        except Exception:
            continue

    score = round(100 * weighted_sum)
    cat = "MINOR" if score <= 29 else "MODERATE" if score <= 65 else "SEVERE"
    return score, cat, ""


class SeverityBar(Flowable):
    def __init__(self, score: Any, width: int = 300, height: int = 14):
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
        v = max(0, min(100, int(self.score)))
        fill = (v / 100.0) * self.width
        c.setFillColor(colors.red if v >= 66 else colors.orange if v >= 30 else colors.green)
        c.rect(0, 0, fill, self.height, stroke=0, fill=1)


def _section(title: str, pairs: list[tuple[str, Any]]) -> list[Any]:
    rows = [[Paragraph(f"<b>{k}</b>", field_style), Paragraph(str(v), value_style)] for k, v in pairs]
    table = Table(rows, colWidths=[180, 360])
    table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), 0.5, colors.black), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    return [Paragraph(f"<b>{title}</b>", styles["Heading2"]), table, Spacer(1, 10)]


def safe_join(val: Any) -> str:
    if isinstance(val, list):
        return ", ".join(str(v) for v in val)
    return str(val) if val else ""


def generate_pdf(image_path: Path, annotated_image_path: Path, model_output: Dict[str, Any], output_path: Path) -> None:
    doc = SimpleDocTemplate(str(output_path), pagesize=letter)
    story: list[Any] = [
        Paragraph("<b>EVENT SUMMARY</b>", styles["Title"]),
        Spacer(1, 12),
        Paragraph("<b>Unannotated Image</b>", styles["Heading2"]),
        Spacer(1, 6),
        Image(str(image_path), width=6 * inch, height=3.5 * inch),
        Spacer(1, 12),
    ]
    if annotated_image_path.exists():
        story += [Paragraph("<b>Annotated Image</b>", styles["Heading2"]), Spacer(1, 6), Image(str(annotated_image_path), width=6 * inch, height=3.5 * inch), Spacer(1, 12)]

    sev = model_output.setdefault("severity_info", {})
    score, cat, notes = calculate_severity(sev)
    sev.setdefault("derived_by_python", {})
    sev["derived_by_python"]["severity_score_0_to_100"] = score
    sev["derived_by_python"]["severity_category"] = cat
    sev["derived_by_python"]["notes_uncertainty"] = notes

    if cat == "SEVERE":
        story += [Paragraph("<font color='red'><b>SEVERE INCIDENT</b></font>", styles["Heading1"]), Spacer(1, 12)]
    elif cat == "UNKNOWN":
        story += [Paragraph("<font color='gray'><b>SEVERITY UNKNOWN</b></font>", styles["Heading1"]), Spacer(1, 12)]

    story += [
        Paragraph("<b>Severity Assessment</b>", styles["Heading2"]),
        Spacer(1, 6),
        SeverityBar(score),
        Spacer(1, 6),
        Paragraph(f"<b>Score:</b> {score} &nbsp;&nbsp; <b>Category:</b> {cat}", styles["Normal"]),
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
    ver_details = model_output.get("verification_details", {})

    story += _section(
        "General Information",
        [
            ("Camera ID", cam.get("id", "")),
            ("Camera Name", cam.get("name", "")),
            ("Public URL", cam.get("publicVideoURL", "")),
            ("Latitude", cam.get("lat", "")),
            ("Longitude", cam.get("lon", "")),
            ("City", cam.get("city", "")),
            ("Weather", cam.get("weather_live", "")),
            ("Categories", safe_join(cam.get("cameraCategories", []))),
            ("Route", f'{cam.get("routePrefix", "")}{cam.get("routeNumber", "")}{cam.get("routeSuffix", "")}'),
            ("Milepost", cam.get("milePost", "")),
            ("Direction", cam.get("direction_of_travel_interpreted", "")),
            ("Direction Confidence", cam.get("confidence_direction", "")),
            ("Lane Count", cam.get("known_lane_count", "")),
            ("Camera IP", cam.get("cctvIp", "")),
            ("Operational Status", cam.get("opStatus", "")),
        ],
    )

    sev_inputs = sev.get("severity_inputs", {})
    sev_triggers = sev.get("severe_gating_triggers", {})
    unknown_flags = sev.get("unknown_flags", {})
    derived = sev.get("derived_by_python", {})

    story += _section(
        "Severity Criteria",
        [
            ("Lane Blockage Score", sev_inputs.get("lane_blockage", {}).get("score_0_to_4", "")),
            ("Lane Blockage Evidence", sev_inputs.get("lane_blockage", {}).get("evidence", "")),
            ("Lane Blockage Confidence", sev_inputs.get("lane_blockage", {}).get("confidence_0_to_1", "")),
            ("Vehicle Count Score", sev_inputs.get("vehicle_count", {}).get("score_0_to_4", "")),
            ("Vehicle Count Evidence", sev_inputs.get("vehicle_count", {}).get("evidence", "")),
            ("Vehicle Count Confidence", sev_inputs.get("vehicle_count", {}).get("confidence_0_to_1", "")),
            ("Vehicle Type Score", sev_inputs.get("vehicle_type", {}).get("score_0_to_4", "")),
            ("Vehicle Type Evidence", sev_inputs.get("vehicle_type", {}).get("evidence", "")),
            ("Vehicle Type Confidence", sev_inputs.get("vehicle_type", {}).get("confidence_0_to_1", "")),
            ("Hazards Score", sev_inputs.get("hazards", {}).get("score_0_to_4", "")),
            ("Hazards Evidence", sev_inputs.get("hazards", {}).get("evidence", "")),
            ("Hazards Confidence", sev_inputs.get("hazards", {}).get("confidence_0_to_1", "")),
            ("Vehicle Orientation Score", sev_inputs.get("vehicle_orientation", {}).get("score_0_to_4", "")),
            ("Vehicle Orientation Evidence", sev_inputs.get("vehicle_orientation", {}).get("evidence", "")),
            ("Vehicle Orientation Confidence", sev_inputs.get("vehicle_orientation", {}).get("confidence_0_to_1", "")),
            ("Damage Score", sev_inputs.get("damage_deformation", {}).get("score_0_to_4", "")),
            ("Damage Evidence", sev_inputs.get("damage_deformation", {}).get("evidence", "")),
            ("Damage Confidence", sev_inputs.get("damage_deformation", {}).get("confidence_0_to_1", "")),
            ("Debris Extent Score", sev_inputs.get("debris_extent", {}).get("score_0_to_4", "")),
            ("Debris Extent Evidence", sev_inputs.get("debris_extent", {}).get("evidence", "")),
            ("Debris Extent Confidence", sev_inputs.get("debris_extent", {}).get("confidence_0_to_1", "")),
            ("Fire Visible Trigger", sev_triggers.get("fire_visible", "")),
            ("Rollover Confirmed", sev_triggers.get("rollover_confirmed", "")),
            ("Major Structural Collapse", sev_triggers.get("major_structural_collapse", "")),
            ("Full Roadway Blocked", sev_triggers.get("full_roadway_blocked", "")),
            ("Large Debris Field", sev_triggers.get("large_debris_field_across_multiple_lanes", "")),
            ("Critical Unknown Count", unknown_flags.get("critical_unknown_count", "")),
            ("Should Return Unknown", unknown_flags.get("should_return_unknown", "")),
            ("Severity Notes / Uncertainty", derived.get("notes_uncertainty", "")),
        ],
    )

    story += _section(
        "Incident Information",
        [
            ("Incident Detected", incident.get("incident_detected", False)),
            ("Status", incident.get("incident_status", "")),
            ("Types", safe_join(incident.get("incident_types", []))),
            ("Confidence", incident.get("confidence_incident", 0.0)),
            ("Why", incident.get("why", "")),
            ("Actual Accident Verified By Gemini", incident.get("actual_accident_verified_by_gemini", "")),
            ("Accident Verification Status", incident.get("actual_accident_verification_status", "")),
            ("Accident Verification Note", incident.get("actual_accident_verification_note", "")),
        ],
    )

    story += _section(
        "Verification Details",
        [
            ("Image Verification", ver_details.get("image_verification", {}).get("confirmed_accident", "")),
            ("Image Note", ver_details.get("image_verification", {}).get("note", "")),
            ("Video Verification", ver_details.get("video_verification", {}).get("confirmed_accident", "")),
            ("Video Note", ver_details.get("video_verification", {}).get("note", "")),
        ],
    )

    story += _section(
        "Vehicle Summary",
        [
            ("Vehicle Count", vehicles.get("count_involved", 0)),
            ("Count Confidence", vehicles.get("confidence_vehicle_count", "")),
            ("Type Confidence", vehicles.get("confidence_vehicle_types", "")),
        ],
    )

    story += _section(
        "People & Injuries",
        [
            ("People Visible", people.get("people_visible_count", 0)),
            ("Injuries Visible", people.get("injuries_visible", "")),
            ("Injury Signs", safe_join(people.get("injury_signs", []))),
        ],
    )

    story += _section(
        "Hazards",
        [
            ("Fire", hazards.get("fire_visible", "")),
            ("Smoke", hazards.get("smoke_visible", "")),
            ("Debris", hazards.get("debris_visible", "")),
            ("Fluid Spill", hazards.get("fluid_spill_possible", "")),
            ("Notes", hazards.get("notes", "")),
        ],
    )

    story += _section(
        "Lane Impact",
        [
            ("Lanes Affected", safe_join(lane.get("lanes_affected", []))),
            ("Shoulder", lane.get("shoulder_affected", "")),
            ("Traffic Flow", lane.get("traffic_flow", "")),
        ],
    )

    story += _section(
        "Location in View",
        [
            ("Region", location.get("in_frame_region", "")),
            ("Distance", location.get("distance_bucket", "")),
            ("Lane", rel.get("lane_id", "")),
            ("Shoulder", rel.get("shoulder", "")),
            ("Statement", location.get("location_statement", "")),
        ],
    )

    story += _section(
        "Media Inputs",
        [
            ("Unannotated Image", str(image_path)),
            ("Annotated Image", str(annotated_image_path)),
        ],
    )

    doc.build(story)


def run_gemini_analysis(
    temp_dir: str | Path = TEMP_DIR,
    model_name: str = DEFAULT_MODEL,
    camera_id: Optional[str] = None,
) -> Dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    RAW_IMAGE_PATH = Path(temp_dir) / "best_accident_frame_raw.jpg"
    ANNOTATED_IMAGE_PATH = Path(temp_dir) / "best_accident_frame_annotated.jpg"
    CLIP_VIDEO_PATH = Path(temp_dir) / "accident_context_clip_raw.webm"
    OUTPUT_JSON_PATH = Path(temp_dir) / "gemini_results.json"
    OUTPUT_PDF_PATH = Path(temp_dir) / "incident_report.pdf"

    if not RAW_IMAGE_PATH.exists():
        raise FileNotFoundError(f"Missing detection artifact: {RAW_IMAGE_PATH}")

    source_video_url = (camera_id or "").strip()
    geojson_data = {}
    if GEOJSON_PATH.exists():
        try:
            geojson_data = json.loads(GEOJSON_PATH.read_text(encoding="utf-8"))
        except Exception:
            geojson_data = {}
    camera_info = find_camera_in_geojson_by_url(geojson_data, source_video_url)
    lat = camera_info.get("lat")
    lon = camera_info.get("lon")
    camera_info["weather_live"] = get_weather_condition(lat, lon) if lat and lon else "UNKNOWN"
    camera_info["city"] = get_city_from_lat_lon(lat, lon) if lat and lon else "UNKNOWN"

    image_ver = analyze_image_accident_verification(client, model_name, RAW_IMAGE_PATH)
    video_ver = analyze_video_accident_verification(client, model_name, CLIP_VIDEO_PATH) if CLIP_VIDEO_PATH.exists() else {"confirmed_accident": "UNKNOWN", "note": "Clip missing."}

    prompt = PROMPT_FILE.read_text(encoding="utf-8", errors="ignore") if PROMPT_FILE.exists() else "Return strict forensic JSON."
    payload = {
        "camera_metadata": camera_info,
        "source_video_url": source_video_url,
        "verification": {"image": image_ver, "video": video_ver},
    }
    main_prompt = f"{prompt}\n\nContext:\n{json.dumps(payload)}\nReturn JSON only."
    raw_image = PILImage.open(RAW_IMAGE_PATH)
    response = retry_with_backoff(
        lambda: client.models.generate_content(
            model=model_name,
            contents=[main_prompt, raw_image],
            config=types.GenerateContentConfig(response_mime_type="application/json"),
        )
    )
    model_output = _clean_parse_json(getattr(response, "text", ""))
    if not model_output:
        model_output = {}

    model_output.setdefault("camera", camera_info)
    model_output.setdefault("incident", {})
    model_output["incident"].setdefault("incident_detected", False)
    model_output["incident"].setdefault("incident_types", ["UNKNOWN"])
    model_output["incident"].setdefault("confidence_incident", 0.0)
    model_output.setdefault("lane_impact", {})
    model_output["lane_impact"].setdefault("lanes_affected", [])
    model_output.setdefault("severity_info", {"severity_inputs": {}, "severe_gating_triggers": {}, "derived_by_python": {}})
    model_output["verification_details"] = {"image_verification": image_ver, "video_verification": video_ver}
    ver_summary = combine_accident_verification(image_ver, video_ver)
    model_output["incident"]["actual_accident_verified_by_gemini"] = ver_summary["actual_accident_verified_by_gemini"]
    model_output["incident"]["actual_accident_verification_status"] = ver_summary["actual_accident_verification_status"]
    model_output["incident"]["actual_accident_verification_note"] = ver_summary["actual_accident_verification_note"]

    score, cat, notes = calculate_severity(model_output.get("severity_info", {}))
    model_output["severity_info"].setdefault("derived_by_python", {})
    model_output["severity_info"]["derived_by_python"]["severity_score_0_to_100"] = score
    model_output["severity_info"]["derived_by_python"]["severity_category"] = cat
    model_output["severity_info"]["derived_by_python"]["notes_uncertainty"] = notes

    OUTPUT_JSON_PATH.write_text(json.dumps(model_output, indent=2), encoding="utf-8")
    generate_pdf(RAW_IMAGE_PATH, ANNOTATED_IMAGE_PATH, model_output, OUTPUT_PDF_PATH)

    return {
        "result_json": str(OUTPUT_JSON_PATH),
        "pdf_path": str(OUTPUT_PDF_PATH),
        "video_path": str(CLIP_VIDEO_PATH),
        "report": {
            "incident_detected": model_output.get("incident", {}).get("incident_detected", False),
            "incident_type": safe_join(model_output.get("incident", {}).get("incident_types", ["UNKNOWN"])),
            "severity": model_output.get("severity_info", {}).get("derived_by_python", {}).get("severity_category", "UNKNOWN"),
            "confidence_incident": model_output.get("incident", {}).get("confidence_incident", 0.0),
            "lanes_blocked": len(model_output.get("lane_impact", {}).get("lanes_affected", [])),
            "raw_payload": model_output,
        },
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Gemini forensic analysis stage.")
    parser.add_argument("--temp_dir", default=str(TEMP_DIR))
    parser.add_argument("--model_name", default=DEFAULT_MODEL)
    parser.add_argument("--camera_id", default="")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(run_gemini_analysis(args.temp_dir, args.model_name, args.camera_id or None))
