import os
import json
import tempfile
import requests
import google.generativeai as genai
import cv2

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
#CAMERA_JSON_PATH = os.path.join(SCRIPT_DIR, "MDOT_SHA_CHART_Traffic_Cameras.geojson")
PROMPT_TXT_PATH = os.path.join(SCRIPT_DIR, "prompt.txt")
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

# def load_json(json_path):
#     with open(json_path, "r", encoding="utf-8") as f:
#         return json.load(f)


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

# def test_geojson_camera_links(geojson_data, timeout=15):
    results = []

    features = geojson_data.get("features", [])

    for feature in features:
        props = feature.get("properties", {})

        camera_id = props.get("ID", "")
        location = props.get("location", "")
        video_url = props.get("CCTVPublicURL", "")
        stream_url = props.get("url", "")
        hls_url = props.get("hlsurl", "")

        urls_to_test = [
            ("CCTVPublicURL", video_url),
            ("url", stream_url),
            ("hlsurl", hls_url),
        ]

        for url_name, url in urls_to_test:
            if not url:
                results.append({
                    "camera_id": camera_id,
                    "location": location,
                    "url_type": url_name,
                    "url": url,
                    "status": "missing",
                    "error": "No URL provided"
                })
                continue

            try:
                response = requests.get(url, timeout=timeout, stream=True)
                response.raise_for_status()

                results.append({
                    "camera_id": camera_id,
                    "location": location,
                    "url_type": url_name,
                    "url": url,
                    "status": "ok",
                    "error": ""
                })

            except requests.exceptions.RequestException as e:
                results.append({
                    "camera_id": camera_id,
                    "location": location,
                    "url_type": url_name,
                    "url": url,
                    "status": "error",
                    "error": str(e)
                })

    return results
# def print_link_test_results(results):
    for item in results:
        print("=" * 80)
        print(f"Camera ID : {item['camera_id']}")
        print(f"Location  : {item['location']}")
        print(f"URL Type  : {item['url_type']}")
        print(f"URL       : {item['url']}")
        print(f"Status    : {item['status']}")
        print(f"Error     : {item['error']}")
# =========================
# FIND CAMERA BY ID
# =========================

# def get_camera_by_id(camera_data, target_id):
    """
    Find a camera by ID from either:
    - Simple array: [{id: "...", ...}, ...]
    - GeoJSON: {features: [{properties: {ID: "...", ...}}, ...]}
    """
    # Handle simple array of cameras
    if isinstance(camera_data, list):
        for cam in camera_data:
            if cam.get("id") == target_id or cam.get("ID") == target_id:
                return cam
    
    # Handle GeoJSON format
    if isinstance(camera_data, dict):
        features = camera_data.get("features", [])
        for feature in features:
            props = feature.get("properties", {})
            if props.get("ID") == target_id or props.get("id") == target_id:
                return props
    
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
    url = f"https://chart.maryland.gov/Video/{camera_id}"
    print("Trying:", url)

    response = requests.get(url, timeout=30)

    if response.status_code == 404:
        raise Exception(f"404 Not Found for snapshot URL: {url}")

    response.raise_for_status()

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    temp.write(response.content)
    temp.close()
    return temp.name

def get_file_type(path):
    ext = os.path.splitext(path)[1].lower()
    image_exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    video_exts = {".mp4", ".mov", ".avi", ".mkv", ".m4v"}

    if ext in image_exts:
        return "image"
    if ext in video_exts:
        return "video"
    return "unknown"


def extract_frame_from_video(video_path, sample_position=0.5):
    """
    Extract one sampled frame from a video.

    sample_position:
        0.0 = first frame
        0.5 = middle frame
        1.0 = last frame (approx)
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_count <= 0:
        cap.release()
        raise ValueError(f"Could not determine frame count for video: {video_path}")

    target_frame = max(0, min(frame_count - 1, int(frame_count * sample_position)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)

    success, frame = cap.read()
    cap.release()

    if not success or frame is None:
        raise ValueError(f"Could not read sampled frame from video: {video_path}")

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=".jpg")
    cv2.imwrite(temp.name, frame)
    return temp.name

def download_media_from_url(url, timeout=20):
    """
    Download directly fetchable media from HTTP/HTTPS URLs.
    Returns a local temp file path.
    """
    if not url:
        raise ValueError("Empty URL")

    if url.startswith("rtmp://"):
        raise ValueError(f"RTMP not supported by requests: {url}")

    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()

    content_type = (response.headers.get("Content-Type") or "").lower()

    if "image" in content_type:
        suffix = ".jpg"
    elif "video" in content_type or url.endswith(".m3u8"):
        suffix = ".mp4"
    else:
        suffix = ".bin"

    temp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)

    for chunk in response.iter_content(chunk_size=8192):
        if chunk:
            temp.write(chunk)

    temp.close()
    return temp.name


def get_local_media_from_user():
    user_path = input("Enter the file path to an image or video: ").strip()

    if not os.path.exists(user_path):
        raise FileNotFoundError(f"File not found: {user_path}")

    file_type = get_file_type(user_path)

    if file_type == "image":
        return user_path

    if file_type == "video":
        return extract_frame_from_video(user_path, sample_position=0.5)

    raise ValueError("Unsupported file type. Use an image or video file.")


def get_input_media_from_urls(url_info):
    """
    url_info example:
    {
        "url": "rtmp://...",
        "CCTVPublicURL": "https://...",
        "hlsurl": "https://.../playlist.m3u8"
    }
    """
    urls_to_try = [
        ("CCTVPublicURL", url_info.get("CCTVPublicURL", "")),
        ("hlsurl", url_info.get("hlsurl", "")),
        ("url", url_info.get("url", "")),
    ]

    errors = []

    for url_type, url in urls_to_try:
        if not url:
            errors.append(f"{url_type}: missing")
            continue

        try:
            print(f"Trying {url_type}: {url}")
            downloaded_path = download_media_from_url(url)

            file_type = get_file_type(downloaded_path)

            if file_type == "image":
                print(f"Using downloaded image from {url_type}")
                return downloaded_path

            if file_type == "video":
                print(f"Using sampled frame from downloaded video via {url_type}")
                return extract_frame_from_video(downloaded_path, sample_position=0.5)

            # Special handling for HLS playlists
            if url.endswith(".m3u8"):
                raise ValueError("HLS playlist downloaded, but direct frame extraction is not supported from playlist text.")

            raise ValueError(f"Downloaded file from {url_type} is not a supported image/video type")

        except Exception as e:
            errors.append(f"{url_type}: {e}")

    print("\nCould not access any provided URLs.")
    print("Errors:")
    for err in errors:
        print(f"- {err}")

    return get_local_media_from_user()

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
    # Fold GeoJSON camera properties into the standard camera shape
    if isinstance(camera_info, dict) and (
        "ID" in camera_info or "CCTVPublicURL" in camera_info or "location" in camera_info
    ):
        camera_info = {
            "id": camera_info.get("ID", ""),
            "name": camera_info.get("location", ""),
            "description": camera_info.get("location", ""),
            "publicVideoURL": camera_info.get("CCTVPublicURL", ""),
            "lat": camera_info.get("Latitude", ""),
            "lon": camera_info.get("Longitude", ""),
            "milePost": camera_info.get("milePost", ""),
            "routeNumber": camera_info.get("routeNumber", ""),
            "routePrefix": camera_info.get("routePrefix", ""),
            "routeSuffix": camera_info.get("routeSuffix", ""),
            "cameraCategories": camera_info.get("cameraCategories", []),
            "cctvIp": camera_info.get("cctvIp", ""),
            "commMode": camera_info.get("commMode", ""),
            "opStatus": camera_info.get("opStatus", ""),
            "lastCachedDataUpdateTime": camera_info.get("lastCachedDataUpdateTime", ""),
        }

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

    # Skip empty sections - return empty list if no rows
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
    # Mode 1: test every link in the GeoJSON and print status/errors
    geojson_data = load_json(CAMERA_JSON_PATH)
    results = test_geojson_camera_links(geojson_data)

    for item in results:
        print("=" * 80)
        print(f"Camera ID : {item['camera_id']}")
        print(f"Location  : {item['location']}")
        print(f"URL Type  : {item['url_type']}")
        print(f"URL       : {item['url']}")
        print(f"Status    : {item['status']}")
        print(f"Error     : {item['error']}")

    # Optional: save test results
    with open(os.path.join(SCRIPT_DIR, "camera_link_test_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    # Uncomment this block when you want to run one camera through Gemini + PDF
    """
    prompt_template = load_text(PROMPT_TXT_PATH)

    TEST_CAMERA_ID = "7a00a1dc01250075004d823633235daa"

    camera_info = get_camera_by_id(geojson_data, TEST_CAMERA_ID)
    camera_id = camera_info.get("ID") or camera_info.get("id")

    if not camera_id:
        raise ValueError("Camera data must contain an ID")

    image_path = get_input_media(camera_id)
    model_output = analyze_image(image_path, camera_info, prompt_template)
    generate_pdf(image_path, model_output, OUTPUT_PDF_PATH)

    print(f"PDF generated: {OUTPUT_PDF_PATH}")
    """

# def test_pdf_generation_only():
    """Test PDF generation without AI model or JSON reading - uses local test image"""
    print("Testing PDF generation with dummy data and local image...")
    
    # Dummy camera info (no JSON reading)
    camera_info = {
        "id": "7a00a1dc01250075004d823633235daa",
        "name": "Test Camera",
        "description": "Test Location - I-95 North",
        "publicVideoURL": "https://chart.maryland.gov/Video/GetSnapshot?cameraId=7a00a1dc01250075004d823633235daa",
        "lat": "39.2904",
        "lon": "-76.6122",
        "routePrefix": "I",
        "routeNumber": "95",
        "routeSuffix": "",
        "milePost": "12.5",
        "opStatus": "OPERATIONAL",
        "commMode": "Active",
        "cctvIp": "192.168.1.1",
        "cameraCategories": ["traffic"],
        "lastCachedDataUpdateTime": "2026-04-08T10:00:00Z",
    }
    
    # Dummy model output (no AI model call)
    model_output = {
        "camera": {k: v for k, v in camera_info.items()},
        "incident": {
            "incident_detected": True,
            "incident_status": "traffic_congestion",
            "incident_types": ["congestion"],
            "confidence_incident": 0.85,
            "why": "Heavy traffic detected in test image",
        },
        "vehicles": {
            "count_involved": 5,
            "confidence_vehicle_count": 0.9,
            "confidence_vehicle_types": 0.88,
            "list": [
                {"type": "sedan", "count": 3},
                {"type": "truck", "count": 2},
            ],
        },
        "people": {
            "people_visible_count": 0,
            "injuries_visible": "no",
            "injury_signs": [],
            "confidence_injuries": 0.0,
        },
        "hazards": {
            "fire_visible": "no",
            "smoke_visible": "no",
            "debris_visible": "no",
            "fluid_spill_possible": "no",
            "notes": "No hazards detected",
            "confidence_hazards": 0.95,
        },
        "lane_impact": {
            "lanes_affected": ["middle", "right"],
            "shoulder_affected": "no",
            "traffic_flow": "moderate_disruption",
            "confidence_lane_impact": 0.8,
            "lane_impact_notes": "2 of 3 lanes affected",
        },
        "location_in_view": {
            "location_description": "Center of frame",
            "confidence_location": 0.9,
        },
        "severity_info": {
            "severity_inputs": {
                "incident_type": {"score_0_to_4": 2, "weight": 0.3},
                "vehicle_impact": {"score_0_to_4": 1, "weight": 0.2},
                "injuries": {"score_0_to_4": 0, "weight": 0.3},
                "hazards": {"score_0_to_4": 0, "weight": 0.2},
            }
        },
    }
    
    try:
        # Use a local test image instead of downloading
        test_image_path = os.path.join(os.path.dirname(SCRIPT_DIR), "test_images", "image_3.jpg")
        
        if not os.path.exists(test_image_path):
            print(f"Test image not found at: {test_image_path}")
            # List available test images
            test_images_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "test_images")
            if os.path.exists(test_images_dir):
                files = [f for f in os.listdir(test_images_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    test_image_path = os.path.join(test_images_dir, files[0])
                    print(f"Using first available test image: {test_image_path}")
                else:
                    raise FileNotFoundError("No image files found in test_images folder")
            else:
                raise FileNotFoundError(f"test_images folder not found at: {test_images_dir}")
        
        print(f"Using test image: {test_image_path}")
        
        # Generate PDF with dummy data and local image (no AI model call, no JSON reading)
        print("Generating PDF with severity assessment...")
        generate_pdf(test_image_path, model_output, OUTPUT_PDF_PATH)
        print(f"PDF generated successfully: {OUTPUT_PDF_PATH}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def run_with_ai():
    """
    Current test mode:
    - No JSON loader
    - Local image or video input
    - Gemini AI + severity assessment + PDF
    """
    print("=" * 80)
    print("RUNNING AI IMAGE/VIDEO ASSESSMENT")
    print("=" * 80)

    try:
        print("\n1. Loading prompt template...")
        prompt_template = load_text(PROMPT_TXT_PATH)
        print("   ✓ Prompt loaded")

        camera_info = {
            "id": "",
            "name": "Test Input",
            "description": "Local image/video input",
            "publicVideoURL": "",
            "lat": "",
            "lon": "",
            "routePrefix": "",
            "routeNumber": "",
            "routeSuffix": "",
            "milePost": "",
            "opStatus": "",
            "commMode": "",
            "cctvIp": "",
            "cameraCategories": [],
            "lastCachedDataUpdateTime": "",
        }

        print("\n2. Trying camera URLs...")

        url_info = {
            "url": "rtmp://strmr5.sha.maryland.gov/rtplive/0000309302ce009e0052fa36c4235c0a",
            "CCTVPublicURL": "https://chart.maryland.gov/Video/GetVideo/0000309302ce009e0052fa36c4235c0a",
            "hlsurl": "https://strmr5.sha.maryland.gov/rtplive/0000309302ce009e0052fa36c4235c0a/playlist.m3u8",
        }

        media_path = get_input_media_from_urls(url_info)
        print(f"   ✓ Media ready: {media_path}")

        print("\n3. Calling Gemini AI...")
        model_output = analyze_image(media_path, camera_info, prompt_template)
        print("   ✓ AI analysis complete")

        print("\n4. Generating PDF with severity assessment...")
        generate_pdf(media_path, model_output, OUTPUT_PDF_PATH)
        print(f"   ✓ PDF generated: {OUTPUT_PDF_PATH}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":

    
    # For full production pipeline with AI model and Gemini analysis:
    run_with_ai()
