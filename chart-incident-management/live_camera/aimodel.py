import os
import json
import tempfile
import requests
import google.generativeai as genai

from dotenv import load_dotenv
from PIL import Image as PILImage

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

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-2.5-flash")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CAMERA_JSON_PATH = os.path.join(SCRIPT_DIR, "GetCameras.json")
PROMPT_TXT_PATH = os.path.join(SCRIPT_DIR, "Prompt.txt")
OUTPUT_PDF_PATH = os.path.join(SCRIPT_DIR, "incident_report.pdf")

CAMERA_INDEX = 0
CAMERA_KEY = None  # use only if your JSON is a dict and you want a specific key

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
# LOADERS
# =========================

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_text(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read()


def get_camera_info(camera_data, index=0, key=None):
    """
    Supports:
    - a single camera dict
    - a list of camera dicts
    - a dict of camera dicts
    """
    if isinstance(camera_data, list):
        return camera_data[index]

    if isinstance(camera_data, dict):
        if key is not None:
            return camera_data[key]

        expected_camera_fields = {
            "cameraCategories", "cctvIp", "commMode", "description", "id",
            "lastCachedDataUpdateTime", "lat", "lon", "milePost", "name",
            "opStatus", "publicVideoURL", "routeNumber", "routePrefix", "routeSuffix"
        }

        if expected_camera_fields.intersection(camera_data.keys()):
            return camera_data

        first_key = next(iter(camera_data))
        return camera_data[first_key]

    raise ValueError("Unsupported camera JSON structure")
# =========================
# FIND CAMERA BY ID
# =========================

def get_camera_by_id(camera_data, target_id):
    if not isinstance(camera_data, list):
        raise ValueError("Expected camera JSON to be a list")

    for cam in camera_data:
        if cam.get("id") == target_id:
            return cam

    raise ValueError(f"Camera id not found: {target_id}")

# =========================
# HELPERS
# =========================

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


def download_snapshot(camera_id):
    url = f"https://chart.maryland.gov/Video/GetSnapshot?cameraId={camera_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(response.content)
    temp.close()
    return temp.name


def build_prompt(prompt_template, camera_info):
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

    return prompt_template.replace("__CAMERA_METADATA__", json.dumps(metadata, indent=2))


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


def normalize_output(model_output, camera_info):
    cam = model_output.setdefault("camera", {})
    cam["id"] = camera_info.get("id", "")
    cam["name"] = camera_info.get("name", "")
    cam["description"] = camera_info.get("description", "")
    cam["publicVideoURL"] = camera_info.get("publicVideoURL", "")
    cam["lat"] = camera_info.get("lat", "")
    cam["lon"] = camera_info.get("lon", "")
    cam["routePrefix"] = camera_info.get("routePrefix", "")
    cam["routeNumber"] = camera_info.get("routeNumber", "")
    cam["routeSuffix"] = camera_info.get("routeSuffix", "")
    cam["milePost"] = camera_info.get("milePost", "")
    cam["opStatus"] = camera_info.get("opStatus", "")
    cam["commMode"] = camera_info.get("commMode", "")
    cam["cctvIp"] = camera_info.get("cctvIp", "")
    cam["cameraCategories"] = camera_info.get("cameraCategories", [])
    cam["lastCachedDataUpdateTime"] = to_str(camera_info.get("lastCachedDataUpdateTime", ""))

    cam.setdefault("direction_of_travel_interpreted", "")
    cam.setdefault("confidence_direction", "")
    cam.setdefault("weather_notes", "")
    cam.setdefault("known_lane_count", "")

    model_output.setdefault("incident", {
        "incident_detected": False,
        "incident_status": "no_incident",
        "incident_types": [],
        "confidence_incident": 0.0,
        "why": "",
    })

    model_output.setdefault("vehicles", {
        "count_involved": 0,
        "confidence_vehicle_count": 0.0,
        "confidence_vehicle_types": 0.0,
        "list": [],
    })

    model_output.setdefault("people", {
        "people_visible_count": 0,
        "injuries_visible": "",
        "injury_signs": [],
        "confidence_injuries": 0.0,
    })

    model_output.setdefault("hazards", {
        "fire_visible": "",
        "smoke_visible": "",
        "debris_visible": "",
        "fluid_spill_possible": "",
        "notes": "",
        "confidence_hazards": 0.0,
    })

    model_output.setdefault("lane_impact", {
        "lanes_affected": [],
        "shoulder_affected": "",
        "traffic_flow": "",
        "confidence_lane_impact": 0.0,
        "lane_impact_notes": "",
    })

    model_output.setdefault("location_in_view", {
        "in_frame_region": "",
        "distance_bucket": "",
        "relative_road_position": {
            "lane_id": "",
            "shoulder": "",
            "median": "",
        },
        "location_statement": "",
    })

    model_output.setdefault("severity_info", {
        "severity_inputs": {
            "vehicle_damage": {
                "score_0_to_4": 0,
                "weight": 0.25,
                "confidence_0_to_1": 0.0,
                "evidence": "",
            },
            "lane_blockage": {
                "score_0_to_4": 0,
                "weight": 0.25,
                "confidence_0_to_1": 0.0,
                "evidence": "",
            },
            "injury_visibility": {
                "score_0_to_4": 0,
                "weight": 0.25,
                "confidence_0_to_1": 0.0,
                "evidence": "",
            },
            "fire_smoke_hazard": {
                "score_0_to_4": 0,
                "weight": 0.25,
                "confidence_0_to_1": 0.0,
                "evidence": "",
            },
        },
        "severe_gating_triggers": {
            "vehicle_fire": False,
            "multiple_vehicle_pileup": False,
            "visible_injury": False,
            "all_lanes_blocked": False,
        },
        "derived_by_python": {
            "severity_score_0_to_100": None,
            "severity_category": None,
            "notes_uncertainty": "",
        },
    })

    return model_output

# =========================
# GEMINI
# =========================

def analyze_image(image_path, camera_info, prompt_template):
    prompt = build_prompt(prompt_template, camera_info)
    image = PILImage.open(image_path)

    response = model.generate_content([prompt, image])
    text = clean_response(response.text)
    output = json.loads(text)

    return normalize_output(output, camera_info)

# =========================
# SEVERITY
# =========================

def calculate_severity(severity_inputs):
    if not severity_inputs:
        return 0, "UNKNOWN"

    weighted_sum = 0.0
    total_weight = 0.0

    for _, item in severity_inputs.items():
        score = safe_float(item.get("score_0_to_4", 0), 0.0)
        weight = safe_float(item.get("weight", 0), 0.0)
        weighted_sum += score * weight
        total_weight += weight

    score_100 = int((weighted_sum / total_weight) * 25) if total_weight > 0 else 0

    if score_100 >= 75:
        return score_100, "SEVERE"
    if score_100 >= 40:
        return score_100, "MODERATE"
    return score_100, "MINOR"


class SeverityBar(Flowable):
    def __init__(self, score, width=300, height=14):
        super().__init__()
        self.score = max(0, min(100, int(score)))
        self.width = width
        self.height = height

    def draw(self):
        c = self.canv
        c.setStrokeColor(colors.black)
        c.rect(0, 0, self.width, self.height, stroke=1, fill=0)

        fill_width = (self.score / 100.0) * self.width
        if self.score >= 75:
            c.setFillColor(colors.red)
        elif self.score >= 40:
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


def generate_pdf(image_path, model_output, output_path):
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    story = []

    story += [
        Paragraph("<b>EVENT SUMMARY</b>", styles["Title"]),
        Spacer(1, 12),
        Image(image_path, width=6 * inch, height=3.5 * inch),
        Spacer(1, 12),
    ]

    sev = model_output.get("severity_info", {})
    score, category = calculate_severity(sev.get("severity_inputs", {}))
    sev.setdefault("derived_by_python", {})
    sev["derived_by_python"]["severity_score_0_to_100"] = score
    sev["derived_by_python"]["severity_category"] = category

    if category == "SEVERE":
        story += [
            Paragraph("<font color='red'><b>SEVERE INCIDENT</b></font>", styles["Heading1"]),
            Spacer(1, 12),
        ]

    story += [
        Paragraph("<b>Severity Assessment</b>", styles["Heading2"]),
        Spacer(1, 6),
        SeverityBar(score),
        Spacer(1, 6),
        Paragraph(f"<b>Score:</b> {score} &nbsp;&nbsp; <b>Category:</b> {category}", styles["Normal"]),
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
    ])

    story += make_section("Vehicle Summary", [
        ("Vehicle Count", vehicles.get("count_involved", 0)),
        ("Count Confidence", vehicles.get("confidence_vehicle_count", "")),
        ("Type Confidence", vehicles.get("confidence_vehicle_types", "")),
    ])

    story.append(Paragraph("<b>Vehicle Details</b>", styles["Heading2"]))
    vehicle_rows = [["Vehicle ID", "Type", "Damage", "Lane"]]
    for v in vehicles.get("list", []):
        road_pos = v.get("road_position", {})
        vehicle_rows.append([
            fmt(v.get("vehicle_id", "")),
            fmt(v.get("type", "")),
            fmt(v.get("damage", "")),
            fmt(road_pos.get("lane_id", "")),
        ])

    vehicle_table = Table(vehicle_rows)
    vehicle_table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.5, colors.black),
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
    ]))
    story += [vehicle_table, Spacer(1, 12)]

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

    severity_pairs = []
    for key, data in sev.get("severity_inputs", {}).items():
        severity_pairs.append((
            key.replace("_", " ").title(),
            f"Score: {data.get('score_0_to_4', '')} "
            f"(Wt: {data.get('weight', '')}), "
            f"Conf: {data.get('confidence_0_to_1', '')}. "
            f"Evidence: {data.get('evidence', '')}"
        ))
    story += make_section("Severity Inputs", severity_pairs)

    gating_pairs = [
        (k.replace("_", " ").title(), v)
        for k, v in sev.get("severe_gating_triggers", {}).items()
    ]
    story += make_section("Severe Gating Triggers", gating_pairs)

    story += make_section("Derived Severity Information", [
        ("Severity Score", sev.get("derived_by_python", {}).get("severity_score_0_to_100", "")),
        ("Category", sev.get("derived_by_python", {}).get("severity_category", "")),
        ("Notes", sev.get("derived_by_python", {}).get("notes_uncertainty", "")),
    ])

    doc.build(story)

# =========================
# MAIN
# =========================

def main():
    camera_data = load_json(CAMERA_JSON_PATH)
    prompt_template = load_text(PROMPT_TXT_PATH)

    TEST_CAMERA_ID = "4800e8aa00ab0059005cd336c4235c0a"
    camera_info = get_camera_by_id(camera_data, TEST_CAMERA_ID)

    camera_id = camera_info.get("id")
    if not camera_id:
        raise ValueError("Camera JSON must contain an 'id' field")

    image_path = download_snapshot(camera_id)
    model_output = analyze_image(image_path, camera_info, prompt_template)
    generate_pdf(image_path, model_output, OUTPUT_PDF_PATH)

    print(f"PDF generated: {OUTPUT_PDF_PATH}")


if __name__ == "__main__":
    main()