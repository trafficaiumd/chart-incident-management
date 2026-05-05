from __future__ import annotations

import base64
import json
import mimetypes
import sys
import threading
import time
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
from dash import Dash, Input, Output, State, dcc, html, no_update
import dash_bootstrap_components as dbc
from flask import Response, send_from_directory

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from yolo_ai_layer.detection_test import run_detection
from yolo_ai_layer.gemini_ai_test import run_gemini_analysis


TEMP_DIR = PROJECT_ROOT / "data" / "temp"
UPLOAD_DIR = TEMP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


class TerminalLogger:
    def __init__(self, log_path):
        self.log_path = log_path
        self.terminal = sys.stdout
        # Clear log on startup
        with open(self.log_path, 'w') as f: f.write("System Initialized...\n")

    def write(self, message):
        self.terminal.write(message)
        with open(self.log_path, 'a') as f:
            f.write(message)

    def flush(self):
        self.terminal.flush()


LOG_FILE_PATH = TEMP_DIR / "terminal.log"
sys.stdout = TerminalLogger(LOG_FILE_PATH)

LOGO_ABSOLUTE_PATH = Path(
    "/home/group1/chart-incident-management/chart-incident-management/dashboard/__pycache__/logo.png"
)

JOB_LOCK = threading.Lock()
JOB_STATE: Dict[str, Any] = {
    "running": False,
    "stage": "idle",
    "progress": 0,
    "message": "Idle",
    "result": None,
    "error": "",
}
STREAM_FRAME_BYTES = None
STREAM_LOCK = threading.Lock()


def _logo_to_data_uri(image_path: Path) -> str:
    if not image_path.exists():
        return ""
    mime_type, _ = mimetypes.guess_type(str(image_path))
    mime = mime_type or "image/png"
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _decode_upload(contents: str, filename: str) -> Path:
    if "," not in contents:
        raise ValueError("Invalid upload payload.")
    _, payload = contents.split(",", 1)
    raw = base64.b64decode(payload)
    out = UPLOAD_DIR / filename
    out.write_bytes(raw)
    return out.resolve()


def _simulate_camera_ping(camera_id: str) -> bool:
    camera_id = (camera_id or "").strip()
    return bool(camera_id) and len(camera_id) >= 3


def _pipeline_worker(media_path: str, camera_id: Optional[str]) -> None:
    with JOB_LOCK:
        JOB_STATE.update(
            {
                "running": True,
                "stage": "yolo",
                "progress": 20,
                "message": "Running YOLOv11 (detection_test.py)...",
                "result": None,
                "error": "",
            }
        )
    try:
        # Temporary background stream feeder
        def feed_stream(path):
            global STREAM_FRAME_BYTES
            from ultralytics import YOLO
            import numpy as np

            # 1. Load the model specifically for the live visualizer
            try:
                model_path = str(PROJECT_ROOT / "yolo_ai_layer" / "epoch14.pt")
                model = YOLO(model_path)
            except Exception as e:
                print(f"Stream visualizer could not load YOLO: {e}")
                model = None

            cap = cv2.VideoCapture(path)
            while cap.isOpened() and JOB_STATE["running"]:
                ret, frame = cap.read()
                if not ret:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # Loop video
                    continue

                # Resize for consistent streaming
                frame = cv2.resize(frame, (640, 480))
                annotated = frame.copy()

                stage = JOB_STATE["stage"]

                # 2. Draw YOLO Bounding Boxes if model loaded
                if model and stage in ["yolo", "gemini"]:
                    # Run quick inference
                    results = model(annotated, conf=0.5, verbose=False)
                    annotated = results[0].plot()

                # 3. Draw Dynamic HUD Overlays based on Pipeline Stage
                if stage == "yolo":
                    cv2.putText(annotated, "STAGE 1: YOLO VISUAL SCAN ACTIVE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(annotated, "Detecting vehicles & collisions...", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                elif stage == "gemini":
                    # Draw a dark, semi-transparent banner at the bottom for Gemini text
                    overlay = annotated.copy()
                    cv2.rectangle(overlay, (0, 400), (640, 480), (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)

                    cv2.putText(annotated, "STAGE 2: GEMINI FORENSIC ANALYSIS", (20, 430), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    cv2.putText(annotated, "Scanning kinematics, lane impact, and hazard severity...", (20, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                # 4. Encode and yield to the web stream
                _, buffer = cv2.imencode('.jpg', annotated)
                with STREAM_LOCK:
                    STREAM_FRAME_BYTES = buffer.tobytes()

                # Cap the stream framerate
                time.sleep(0.033)

            cap.release()
            with STREAM_LOCK:
                STREAM_FRAME_BYTES = None

        threading.Thread(target=feed_stream, args=(media_path,), daemon=True).start()

        detection_report = run_detection(media_path=media_path, output_dir=str(TEMP_DIR))
        (TEMP_DIR / "detection_summary.json").write_text(json.dumps(detection_report, indent=2), encoding="utf-8")

        with JOB_LOCK:
            JOB_STATE.update(
                {
                    "stage": "gemini",
                    "progress": 65,
                    "message": "Running Gemini AI (gemini_ai_test.py)...",
                }
            )

        gemini_report = run_gemini_analysis(temp_dir=TEMP_DIR, camera_id=camera_id)
        merged = {
            "detection": detection_report,
            "gemini": gemini_report,
            "video_path": detection_report.get("accident_context_clip_raw", ""),
            "pdf_path": gemini_report.get("pdf_path", ""),
        }

        with JOB_LOCK:
            JOB_STATE.update(
                {
                    "running": False,
                    "stage": "done",
                    "progress": 100,
                    "message": "Analysis complete.",
                    "result": merged,
                    "error": "",
                }
            )
    except Exception as exc:
        with JOB_LOCK:
            JOB_STATE.update(
                {
                    "running": False,
                    "stage": "error",
                    "progress": 100,
                    "message": "Pipeline failed.",
                    "result": None,
                    "error": str(exc),
                }
            )


def _landing_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.Img(src=_logo_to_data_uri(LOGO_ABSOLUTE_PATH), style={"height": "72px", "marginRight": "14px"}),
            html.Div([
                html.H2("CHART Transportation System", style={"color": "white", "margin": "0"}),
                html.Div("AI-Powered Incident Detection & Forensic Analysis", style={"color": "#98a2b3", "fontSize": "14px", "marginTop": "2px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "padding": "14px 18px", "backgroundColor": "#5A2CA0", "borderRadius": "10px", "marginBottom": "18px"}),
        html.Div([
            dbc.Row([
                dbc.Col(html.Div([
                    html.H5("1. Source Media", style={"color": "#e5e7eb"}),
                    dcc.Upload(
                        id="media-upload",
                        children=html.Div(["Drag and Drop or ", html.A("Select Video", style={"color": "#a78bfa"})]),
                        multiple=False, accept="video/*,image/*",
                        style={"width": "100%", "height": "60px", "lineHeight": "60px", "borderWidth": "1px", "borderStyle": "dashed", "borderColor": "#7c3aed", "backgroundColor": "#101726", "borderRadius": "10px", "textAlign": "center", "color": "#cbd5e1", "cursor": "pointer"}
                    ),
                    html.Div(id="upload-filename", className="mt-2", style={"color": "#d1d5db", "fontSize": "13px"}),
                    dbc.Progress(id="upload-progress", value=0, max=100, className="mt-2", color="success", style={"height": "8px", "backgroundColor": "#0f172a"}),
                    html.Div(id="media-preview-container", style={"marginTop": "12px", "textAlign": "center"}),
                ], style={"backgroundColor": "#151b2b", "border": "1px solid #2a3441", "borderRadius": "12px", "padding": "24px", "height": "100%"}), md=6),

                dbc.Col(html.Div([
                    html.H5("2. System Metadata Lookup", style={"color": "#e5e7eb"}),
                    html.Div("Enter Camera ID for Weather & GeoJSON routing:", style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "8px"}),
                    dbc.Input(id="camera-id-input", placeholder="e.g., CHART Camera ID or Public URL", type="text", style={"backgroundColor": "#101726", "border": "1px solid #334155", "color": "#e2e8f0"}),
                    dbc.Button("Validate Camera", id="add-camera-btn", className="mt-3 w-100", color="secondary", outline=True),
                    html.Div(id="camera-status-text", className="mt-2", style={"color": "#22c55e", "fontSize": "13px"}),
                    dbc.Progress(id="camera-progress", value=0, max=100, className="mt-2", color="success", style={"height": "8px", "backgroundColor": "#0f172a"}),
                ], style={"backgroundColor": "#151b2b", "border": "1px solid #2a3441", "borderRadius": "12px", "padding": "24px", "height": "100%"}), md=6),
            ], className="mb-4"),

            dbc.Button("Run Analysis", id="run-analysis-btn", size="lg", className="w-100", style={"fontWeight": "700", "color": "white", "border": "none", "background": "linear-gradient(135deg, #4c1d95 0%, #7c3aed 100%)", "padding": "16px", "boxShadow": "0 8px 20px rgba(124, 58, 237, 0.35)", "borderRadius": "10px"}),
        ], style={"maxWidth": "850px", "margin": "26px auto 0 auto"}),
    ])


def _analysis_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.H3("Live Pipeline Processing", style={"color": "white", "fontWeight": "800"}),
            html.Div("YOLOv11 & Gemini Multimodal Analysis Sequence", style={"color": "#a78bfa", "marginBottom": "20px"}),
            html.Div(id="processing-status-text", style={"color": "#f8fafc", "fontSize": "18px", "fontWeight": "700", "marginBottom": "10px"}),
            dbc.Progress(id="processing-progress", value=0, max=100, color="success", style={"height": "16px", "backgroundColor": "#0f172a", "borderRadius": "999px"}),
        ], style={"maxWidth": "800px", "margin": "0 auto 24px auto", "textAlign": "center"}),

        html.Div([
            html.Div("Live AI Vision Feed", style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "10px", "textAlign": "center"}),
            html.Img(src="/video_feed", style={"width": "100%", "maxWidth": "800px", "display": "block", "margin": "0 auto", "borderRadius": "10px", "border": "2px solid #7c3aed", "boxShadow": "0 10px 25px rgba(124, 58, 237, 0.2)"})
        ], style={"backgroundColor": "#151b2b", "padding": "24px", "borderRadius": "16px", "border": "1px solid #2a3441", "maxWidth": "900px", "margin": "0 auto"})
    ], style={"paddingTop": "40px"})


def _verdict_layout() -> html.Div:
    return html.Div([
        html.Div([
            html.Img(src=_logo_to_data_uri(LOGO_ABSOLUTE_PATH), style={"height": "72px", "marginRight": "14px"}),
            html.Div([
                html.H2("CHART Transportation System", style={"color": "white", "margin": "0"}),
                html.Div("AI-Powered Incident Detection & Forensic Analysis", style={"color": "#98a2b3", "fontSize": "14px", "marginTop": "2px"}),
            ]),
        ], style={"display": "flex", "alignItems": "center", "padding": "14px 18px", "backgroundColor": "#5A2CA0", "borderRadius": "10px", "marginBottom": "18px"}),

        html.Div([
            html.Div("ACTIVE INCIDENT ALERT", style={"color": "#fecaca", "fontWeight": "700", "fontSize": "14px"}),
            html.Div("Incident Detected / Type of Accident", style={"color": "white", "fontSize": "24px", "fontWeight": "800", "marginTop": "4px"}),
            html.Div(id="verdict-summary", className="mt-2", style={"color": "#ffe4e6", "fontSize": "14px"}),
        ], style={"backgroundColor": "#4a1120", "border": "1px solid #7f1d1d", "borderRadius": "12px", "padding": "18px 20px", "marginBottom": "16px"}),

        dbc.Row([
            dbc.Col(html.Div([
                html.Div("Forensic 10-Second Clip", style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "10px"}),
                html.Video(id="result-video", controls=True, src="", autoPlay=True, muted=True, style={"width": "100%", "maxHeight": "320px", "borderRadius": "10px", "border": "1px solid #2a3441", "backgroundColor": "#0f172a", "marginBottom": "16px"}),
                dbc.Modal([dbc.ModalHeader("Incident Segment"), dbc.ModalBody(html.Video(id="modal-video", controls=True, src="", autoPlay=True, muted=True, style={"width": "100%"}))], id="video-modal", is_open=False, size="xl"),

                html.Div("Live Pipeline Terminal Logs", style={"color": "#94a3b8", "fontSize": "13px", "marginBottom": "10px"}),
                html.Pre(id="terminal-output", style={"width": "100%", "height": "200px", "backgroundColor": "#000000", "color": "#00ff00", "border": "1px solid #334155", "borderRadius": "8px", "padding": "12px", "overflowY": "scroll", "fontSize": "12px", "fontFamily": "monospace"}),
                dcc.Interval(id="terminal-poller", interval=1000, n_intervals=0)
            ], style={"backgroundColor": "#151b2b", "borderRadius": "12px", "border": "1px solid #2a3441", "padding": "20px", "height": "100%"}), md=7),

            dbc.Col(html.Div([
                html.Div("AI Telemetry", style={"color": "#e2e8f0", "fontWeight": "700", "marginBottom": "14px"}),
                html.Div([
                    html.Div("Severity Score", style={"color": "#94a3b8", "fontSize": "12px"}),
                    html.Div("CALCULATING...", id="severity-badge", style={"display": "inline-block", "marginTop": "8px", "padding": "8px 12px", "borderRadius": "999px", "backgroundColor": "#334155", "color": "#f8fafc", "fontWeight": "800", "fontSize": "13px"}),
                ], style={"marginBottom": "20px"}),
                html.Div([
                    html.Div("Confidence", style={"color": "#94a3b8", "fontSize": "12px", "marginBottom": "8px"}),
                    html.Div(style={"height": "14px", "backgroundColor": "#0f172a", "borderRadius": "999px", "overflow": "hidden", "border": "1px solid #2a3441"}, children=[html.Div(id="confidence-bar", style={"height": "100%", "width": "0%", "background": "linear-gradient(90deg, #334155, #475569)", "transition": "width 1s ease-in-out"})]),
                    html.Div("Analyzing...", id="confidence-text", style={"color": "#94a3b8", "fontWeight": "700", "marginTop": "8px"}),
                ]),
                html.Div("Live Environmental Data", style={"color": "#e2e8f0", "fontWeight": "700", "marginTop": "30px", "marginBottom": "10px"}),
                html.Div(id="env-weather", style={"color": "#38bdf8", "fontSize": "13px", "fontWeight": "600", "marginBottom": "6px"}),
                html.Div(id="env-city", style={"color": "#a78bfa", "fontSize": "13px", "fontWeight": "600"}),

                html.Div([
                    html.A(dbc.Button("Export Official CHART Report", className="w-100 mb-2", style={"backgroundColor": "#7c3aed", "border": "none", "color": "white", "fontWeight": "700", "padding": "12px"}), id="export-report-link", href="", target="_blank", download="incident_report.pdf"),
                    dbc.Button("Back to Landing / Run New", id="back-to-landing-btn", className="w-100", color="secondary", outline=True, style={"padding": "12px"}),
                ], style={"marginTop": "40px"})
            ], style={"backgroundColor": "#151b2b", "borderRadius": "12px", "border": "1px solid #2a3441", "padding": "24px", "height": "100%"}), md=5),
        ], className="g-4"),
    ], style={"maxWidth": "1100px", "margin": "0 auto"})


app: Dash = Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP], suppress_callback_exceptions=True)
server = app.server


@server.route("/temp_media/<path:filename>")
def serve_temp_media(filename):
    # This securely serves files directly from the data/temp directory
    return send_from_directory(str(TEMP_DIR), filename)


def generate_video_stream():
    """Yields multipart JPEG frames for the live dashboard feed."""
    global STREAM_FRAME_BYTES

    # Generate a blank placeholder frame if nothing is running
    blank_img = cv2.UMat(480, 640, cv2.CV_8UC3).get()
    blank_frame = cv2.imencode('.jpg', blank_img)[1].tobytes()

    while True:
        with STREAM_LOCK:
            current_frame = STREAM_FRAME_BYTES

        if current_frame is None:
            frame_to_yield = blank_frame
            time.sleep(0.1)
        else:
            frame_to_yield = current_frame
            # Cap the stream framerate to ~30 FPS to save bandwidth
            time.sleep(0.033)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')


@server.route('/video_feed')
def video_feed():
    """Flask route to serve the multipart video stream."""
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = dbc.Container(
    [
        dcc.Store(id="view-store", data="landing"),
        dcc.Store(id="uploaded-media-path"),
        dcc.Store(id="camera-status-store", data={"connected": False, "camera_id": ""}),
        dcc.Store(id="pipeline-result-store"),
        dcc.Interval(id="pipeline-poller", interval=1000, n_intervals=0, disabled=False),
        html.Div(id="landing-view-container", children=_landing_layout(), style={"display": "block"}),
        html.Div(id="analysis-view-container", children=_analysis_layout(), style={"display": "none"}),
        html.Div(id="verdict-view-container", children=_verdict_layout(), style={"display": "none"}),
    ],
    fluid=True,
    style={"paddingTop": "14px", "paddingBottom": "14px", "backgroundColor": "#0b0f19", "minHeight": "100vh"},
)


@app.callback(
    Output("landing-view-container", "style"),
    Output("analysis-view-container", "style"),
    Output("verdict-view-container", "style"),
    Input("view-store", "data"),
)
def render_page(view_name: str):
    if view_name == "verdict":
        return {"display": "none"}, {"display": "none"}, {"display": "block"}
    if view_name == "analysis":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}
    return {"display": "block"}, {"display": "none"}, {"display": "none"}


@app.callback(
    Output("uploaded-media-path", "data"),
    Output("upload-progress", "value"),
    Output("upload-filename", "children"),
    Output("media-preview-container", "children"),
    Input("media-upload", "contents"),
    State("media-upload", "filename"),
    prevent_initial_call=True,
)
def on_upload(contents: Optional[str], filename: Optional[str]):
    if not contents or not filename:
        return no_update, 0, "No file uploaded.", no_update
    saved = _decode_upload(contents, filename)
    preview: Any
    if contents.startswith("data:image"):
        preview = html.Img(
            src=contents,
            style={
                "maxHeight": "180px",
                "borderRadius": "8px",
                "border": "1px solid #2a3441",
            },
        )
    else:
        preview = html.Div(
            "Video Ready for Analysis",
            style={
                "display": "inline-block",
                "padding": "8px 12px",
                "borderRadius": "8px",
                "border": "1px solid #2a3441",
                "backgroundColor": "#101726",
                "color": "#cbd5e1",
            },
        )
    return str(saved), 100, f"Uploaded: {saved.name}", preview


@app.callback(
    Output("camera-status-store", "data"),
    Output("camera-status-text", "children"),
    Output("camera-progress", "value"),
    Input("add-camera-btn", "n_clicks"),
    State("camera-id-input", "value"),
    prevent_initial_call=True,
)
def on_add_camera(n_clicks: int, camera_id: Optional[str]):
    if not n_clicks:
        return no_update, no_update, no_update
    valid = _simulate_camera_ping(camera_id or "")
    if not valid:
        return {"connected": False, "camera_id": ""}, "Camera validation failed.", 0
    return {"connected": True, "camera_id": camera_id}, f"Connected: {camera_id}", 100


@app.callback(
    Output("view-store", "data", allow_duplicate=True),
    Output("pipeline-poller", "disabled", allow_duplicate=True),
    Output("processing-status-text", "children", allow_duplicate=True),
    Input("run-analysis-btn", "n_clicks"),
    State("uploaded-media-path", "data"),
    State("camera-status-store", "data"),
    prevent_initial_call=True,
)
def start_pipeline(n_clicks: int, media_path: Optional[str], camera_state: Optional[Dict[str, Any]]):
    if not n_clicks:
        return no_update, no_update, no_update
    if not media_path:
        return "landing", False, "Upload media first."
    with JOB_LOCK:
        if JOB_STATE["running"]:
            return "analysis", False, "Pipeline is already running."
    camera_id = (camera_state or {}).get("camera_id")
    threading.Thread(target=_pipeline_worker, args=(media_path, camera_id), daemon=True).start()
    return "analysis", False, "Initializing Pipeline..."


@app.callback(
    Output("processing-progress", "value"),
    Output("processing-status-text", "children"),
    Output("pipeline-result-store", "data"),
    Output("view-store", "data", allow_duplicate=True),
    Output("pipeline-poller", "disabled"),
    Input("pipeline-poller", "n_intervals"),
    State("view-store", "data"),
    prevent_initial_call=True,
)
def poll_pipeline(_n: int, view_name: str):
    with JOB_LOCK:
        progress = JOB_STATE["progress"]
        status = JOB_STATE["message"]
        stage = JOB_STATE["stage"]
        result = JOB_STATE["result"]
        error = JOB_STATE["error"]

    if view_name != "analysis":
        return no_update, no_update, no_update, no_update, True

    if stage == "error":
        return progress, f"Error: {error}", no_update, no_update, True

    if stage == "done" and result:
        return progress, "Analysis Complete. Generating Verdict...", result, "verdict", True

    return progress, status, no_update, no_update, False


@app.callback(
    Output("result-video", "src"),
    Output("modal-video", "src"),
    Output("verdict-summary", "children"),
    Output("export-report-link", "href"),
    Output("severity-badge", "children"),
    Output("severity-badge", "style"),
    Output("confidence-bar", "style"),
    Output("confidence-text", "children"),
    Output("env-weather", "children"),
    Output("env-city", "children"),
    Input("pipeline-result-store", "data"),
)
def render_results(result_data: Optional[Dict[str, Any]]):
    # Default fallback states
    default_badge_style = {
        "display": "inline-block", "marginTop": "8px", "padding": "8px 12px",
        "borderRadius": "999px", "fontWeight": "800", "fontSize": "13px", "letterSpacing": "0.3px",
        "backgroundColor": "#334155", "color": "#f8fafc"
    }
    default_bar_style = {"height": "100%", "width": "0%", "background": "linear-gradient(90deg, #334155, #475569)", "transition": "width 1s ease-in-out"}
    
    if not result_data:
        return "", "", "No results yet.", "", "UNKNOWN", default_badge_style, default_bar_style, "0% Confidence", "Weather: N/A", "City: N/A"

    video_src = ""
    pdf_src = ""
    
    # Bypass cache with timestamps
    import time
    video_path_str = result_data.get("video_path", "")
    if video_path_str:
        video_src = f"/temp_media/{Path(video_path_str).name}?t={int(time.time())}"
        
    pdf_path_str = result_data.get("pdf_path", "")
    if pdf_path_str:
        pdf_src = f"/temp_media/{Path(pdf_path_str).name}?t={int(time.time())}"

    gemini_data = result_data.get("gemini", {})
    report = gemini_data.get("report", {})
    raw_payload = report.get("raw_payload", {})
    
    # 1. Summary Text
    summary = (
        f"Incident Detected: {report.get('incident_detected', False)} | "
        f"Type: {report.get('incident_type', 'UNKNOWN')} | "
        f"Severity: {report.get('severity', 'UNKNOWN')} | "
        f"Lanes Blocked: {report.get('lanes_blocked', 0)}"
    )

    # 2. Severity Badge Logic
    severity_cat = str(report.get("severity", "UNKNOWN")).upper()
    severity_score = raw_payload.get("severity_info", {}).get("derived_by_python", {}).get("severity_score_0_to_100", "N/A")
    badge_text = f"{severity_cat} ({severity_score})" if severity_score != "N/A" else severity_cat
    
    badge_style = default_badge_style.copy()
    if severity_cat == "SEVERE":
        badge_style.update({"backgroundColor": "#7f1d1d", "color": "#fee2e2"}) # Red
    elif severity_cat == "MODERATE":
        badge_style.update({"backgroundColor": "#9a3412", "color": "#ffedd5"}) # Orange
    elif severity_cat == "MINOR":
        badge_style.update({"backgroundColor": "#166534", "color": "#dcfce3"}) # Green
        
    # 3. Confidence Bar Logic
    conf_float = float(report.get("confidence_incident", 0.0))
    conf_pct = int(conf_float * 100)
    
    bar_style = default_bar_style.copy()
    bar_style["width"] = f"{conf_pct}%"
    
    if conf_pct >= 80:
        bar_style["background"] = "linear-gradient(90deg, #22c55e, #16a34a)" # Green
    elif conf_pct >= 50:
        bar_style["background"] = "linear-gradient(90deg, #f59e0b, #d97706)" # Yellow/Orange
    else:
        bar_style["background"] = "linear-gradient(90deg, #ef4444, #dc2626)" # Red
        
    conf_text = f"AI Confidence: {conf_pct}%"

    # 4. Environmental Data
    camera_data = raw_payload.get("camera", {})
    weather_text = f"Weather: {camera_data.get('weather_live', 'UNKNOWN')}"
    city_text = f"Location: {camera_data.get('city', 'UNKNOWN')}"

    return video_src, video_src, summary, pdf_src, badge_text, badge_style, bar_style, conf_text, weather_text, city_text


@app.callback(
    Output("video-modal", "is_open"),
    Input("result-video", "n_clicks"),
    State("video-modal", "is_open"),
    prevent_initial_call=True,
)
def toggle_video_modal(_clicks: int, is_open: bool):
    return not is_open


@app.callback(
    Output("view-store", "data"),
    Output("pipeline-poller", "disabled", allow_duplicate=True),
    Output("processing-progress", "value", allow_duplicate=True),
    Output("processing-status-text", "children", allow_duplicate=True),
    Output("pipeline-result-store", "data", allow_duplicate=True),
    Output("upload-progress", "value", allow_duplicate=True),
    Output("upload-filename", "children", allow_duplicate=True),
    Output("media-preview-container", "children", allow_duplicate=True),
    Output("uploaded-media-path", "data", allow_duplicate=True),
    Input("back-to-landing-btn", "n_clicks"),
    prevent_initial_call=True,
)
def back_to_landing(n_clicks: int):
    if not n_clicks:
        return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

    # 1. Reset the backend job state so the poller doesn't rebound
    with JOB_LOCK:
        JOB_STATE.update({
            "running": False,
            "stage": "idle",
            "progress": 0,
            "message": "Idle",
            "result": None,
            "error": "",
        })

    # 2. Reset the UI elements and transition to landing
    return (
        "landing",          # view-store
        False,              # pipeline-poller disabled
        0,                  # processing-progress
        "",                 # processing-status-text
        None,               # pipeline-result-store
        0,                  # upload-progress
        "No file uploaded.",# upload-filename
        "",                 # media-preview-container
        None                # uploaded-media-path
    )


@app.callback(
    Output("terminal-output", "children"),
    Input("terminal-poller", "n_intervals")
)
def update_terminal(n):
    try:
        with open(LOG_FILE_PATH, 'r') as f:
            logs = f.read()[-2000:]
        return logs
    except:
        return "Waiting for system logs..."


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
