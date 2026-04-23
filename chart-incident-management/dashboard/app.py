"""Single-file Dash application for CHART TIM demo."""

import base64
import json
import os
import signal
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import dash
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update

sys.path.append(str(Path(__file__).resolve().parent.parent))
from utils.pdf_generator import generate_incident_pdf

BASE_DIR = Path(__file__).resolve().parent
LIVE_INCIDENTS_FILE = BASE_DIR / "data" / "live_incidents.json"
TRAINING_FEEDBACK_FILE = BASE_DIR / "data" / "training_feedback_log.json"
OPERATOR_FEEDBACK_FILE = BASE_DIR / "data" / "operator_feedback.json"
STATUS_FILE = BASE_DIR / "data" / "status.json"
PROCESS_META_FILE = BASE_DIR / "data" / "active_process.json"
PROJECT_ROOT = BASE_DIR.parent
ASSETS_DIR = BASE_DIR / "assets"

if (ASSETS_DIR / "Traffic.AI_shield.png").exists():
    SHIELD_LOGO_SRC = "/assets/Traffic.AI_shield.png"
elif (ASSETS_DIR / "image_0.png").exists():
    SHIELD_LOGO_SRC = "/assets/image_0.png"
else:
    SHIELD_LOGO_SRC = ""


def _read_live_incidents():
    if not LIVE_INCIDENTS_FILE.exists():
        return []
    try:
        data = json.loads(LIVE_INCIDENTS_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []
    return data if isinstance(data, list) else []


def _image_to_b64(path: str):
    p = Path(str(path or ""))
    if not p.exists() or not p.is_file():
        return None
    return "data:image/jpeg;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _video_src(path: str):
    p = Path(str(path or ""))
    if not p.exists() or not p.is_file():
        return ""
    try:
        rel = p.resolve().relative_to(ASSETS_DIR.resolve())
        return f"/assets/{str(rel).replace(os.sep, '/')}"
    except Exception:
        pass
    return "data:video/mp4;base64," + base64.b64encode(p.read_bytes()).decode("ascii")


def _append_feedback(entry):
    TRAINING_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if TRAINING_FEEDBACK_FILE.exists():
        try:
            old = json.loads(TRAINING_FEEDBACK_FILE.read_text(encoding="utf-8"))
            if isinstance(old, list):
                data = old
        except Exception:
            pass
    data.append(entry)
    TRAINING_FEEDBACK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _append_operator_correction(entry):
    OPERATOR_FEEDBACK_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = []
    if OPERATOR_FEEDBACK_FILE.exists():
        try:
            old = json.loads(OPERATOR_FEEDBACK_FILE.read_text(encoding="utf-8"))
            if isinstance(old, list):
                data = old
        except Exception:
            pass
    data.append(entry)
    OPERATOR_FEEDBACK_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _incident_confidence(inc):
    if "confidence" in inc:
        try:
            return float(inc["confidence"])
        except Exception:
            pass
    try:
        forensic = inc.get("forensic_report", {})
        incident = forensic.get("incident", {}) if isinstance(forensic, dict) else {}
        if "confidence_incident" in incident:
            return float(incident.get("confidence_incident", 0.0))
    except Exception:
        pass
    sev = str(inc.get("severity", "med")).lower()
    return {"high": 0.9, "med": 0.72, "low": 0.55}.get(sev, 0.7)


def _stop_active_pipeline_if_any() -> None:
    if not PROCESS_META_FILE.exists():
        return
    try:
        pm = json.loads(PROCESS_META_FILE.read_text(encoding="utf-8"))
        pid = int(pm.get("pid", -1))
    except Exception:
        pid = -1
    if pid > 0 and _is_pid_running(pid):
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def _read_status():
    if not STATUS_FILE.exists():
        return {"state": "INGESTING"}
    try:
        data = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {"state": "INGESTING"}


def _is_pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _start_pipeline_for_media(media_file_path: str) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / "accident_guard_yolo26.py"), "--video_path", str(media_file_path)]
    proc = subprocess.Popen(cmd, cwd=str(PROJECT_ROOT))
    PROCESS_META_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROCESS_META_FILE.write_text(
        json.dumps(
            {
                "pid": proc.pid,
                "media_file": str(media_file_path),
                "started_at": datetime.now().isoformat(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return int(proc.pid)


def _format_bytes(n: int) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024.0:.1f} KB"
    return f"{n / (1024.0 * 1024.0):.2f} MB"


def create_layout():
    return html.Div(
        [
            html.Div(
                [
                    html.Div(
                        [
                            html.Img(
                                src=SHIELD_LOGO_SRC,
                                style={"height": "52px", "marginRight": "12px", "display": "none" if not SHIELD_LOGO_SRC else "block"},
                            ),
                            html.H2("CHART Traffic Incident Management", style={"margin": 0, "color": "white"}),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        id="header-status-bar",
                        style={"marginTop": "8px", "fontSize": "12px", "color": "white", "borderTop": "1px solid rgba(255,255,255,0.5)", "paddingTop": "6px"},
                        children=f"Live Status | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Live Incidents: 0",
                    ),
                ],
                style={"backgroundColor": "#3D1A78", "padding": "12px 18px", "position": "sticky", "top": 0, "zIndex": 10},
            ),
            html.Div(
                id="landing-container",
                children=[
                    html.Div(
                        [
                            html.Img(
                                id="landing-logo",
                                src=SHIELD_LOGO_SRC,
                                style={"width": "240px", "marginBottom": "18px", "display": "none" if not SHIELD_LOGO_SRC else "block", "marginLeft": "auto", "marginRight": "auto"},
                            ),
                            html.H3("Analyze Media", style={"color": "white"}),
                            html.Div(
                                id="upload-section",
                                children=[
                                    dcc.Upload(
                                        id="media-upload",
                                        children=html.Div("Drag and drop media here, or click to select a file", style={"fontWeight": "600"}),
                                        style={
                                            "width": "100%",
                                            "height": "64px",
                                            "lineHeight": "64px",
                                            "borderWidth": "1px",
                                            "borderStyle": "dashed",
                                            "borderRadius": "8px",
                                            "textAlign": "center",
                                            "color": "#ddd",
                                            "marginBottom": "10px",
                                            "cursor": "pointer",
                                            "backgroundColor": "#171b28",
                                        },
                                        multiple=False,
                                    ),
                                    dcc.Input(id="media-path-input", type="text", placeholder="Path to media file", style={"width": "100%", "padding": "10px", "marginBottom": "10px"}),
                                    dbc.Progress(id="upload-verify-progress", value=0, label="Awaiting upload...", color="secondary", style={"height": "20px", "display": "none", "marginBottom": "8px"}),
                                ],
                            ),
                            html.Div(
                                id="processing-tracker-section",
                                style={"display": "none", "marginBottom": "10px"},
                                children=[
                                    html.Div(id="processing-progress-label", style={"color": "#b8c1ec", "marginBottom": "6px"}),
                                    dbc.Progress(id="processing-progress-bar", value=0, label="0%", color="info", style={"height": "22px"}),
                                ],
                            ),
                            html.Div(id="uploaded-file-details", style={"color": "#b8c1ec", "fontSize": "12px", "marginBottom": "8px"}),
                            html.Button(
                                "Run Analysis",
                                id="run-analysis-btn",
                                n_clicks=0,
                                disabled=True,
                                style={"width": "100%", "padding": "12px", "fontWeight": "bold", "color": "white", "border": "none", "borderRadius": "8px", "background": "linear-gradient(90deg,#7f5af0,#2cb67d)", "boxShadow": "0 0 20px rgba(127,90,240,0.6)"},
                            ),
                            html.Pre(
                                id="processing-log",
                                style={"marginTop": "12px", "backgroundColor": "#121212", "color": "#9ef0b8", "padding": "10px", "borderRadius": "8px", "minHeight": "120px", "whiteSpace": "pre-wrap"},
                            ),
                        ],
                        style={"width": "min(680px, 94%)", "margin": "40px auto"},
                    )
                ],
                style={"minHeight": "calc(100vh - 90px)", "backgroundColor": "#0c0f17", "display": "block"},
            ),
            html.Div(
                id="verdict-container",
                style={"display": "none", "backgroundColor": "#0f1320", "minHeight": "calc(100vh - 90px)", "padding": "14px"},
                children=[
                    html.Div(id="verdict-banner"),
                    html.Div(id="limp-mode-indicator", style={"display": "none"}),
                    html.Div(
                        [
                            html.Button("Export Official CHART Report", id="export-chart-report-btn", n_clicks=0),
                            html.Button("Back to Landing / New Analysis", id="back-to-landing-btn", n_clicks=0, style={"marginLeft": "8px"}),
                        ],
                        style={"display": "flex", "justifyContent": "flex-end", "margin": "8px 0"},
                    ),
                    html.Button("⚠️ Action Required", id="action-required-btn", n_clicks=0, style={"display": "none", "marginBottom": "8px"}),
                    html.Div(id="action-required-banner", style={"display": "none"}),
                    html.Div(
                        [
                            html.Div([html.H4("Temporal Montage Evidence", style={"color": "white"}), html.Img(id="evidence-montage-image", style={"width": "100%", "borderRadius": "8px"})], style={"flex": "2", "backgroundColor": "#161b2e", "padding": "10px", "borderRadius": "8px"}),
                            html.Div([html.H4("Hero Frame (Peak Impact)", style={"color": "white"}), html.Img(id="hero-frame-image", style={"width": "100%", "borderRadius": "8px"})], style={"flex": "2", "backgroundColor": "#161b2e", "padding": "10px", "borderRadius": "8px"}),
                            html.Div(
                                [
                                    html.H4("AI Insight", style={"color": "white"}),
                                    html.Pre(id="ai-insight-text", style={"whiteSpace": "pre-wrap", "color": "#ddd"}),
                                    html.Div(id="severity-score-chip", style={"color": "#ddd", "marginBottom": "6px", "fontWeight": "bold"}),
                                    dcc.Graph(id="confidence-gauge", config={"displayModeBar": False}),
                                    html.Div(id="classification-badges"),
                                ],
                                style={"flex": "1.2", "backgroundColor": "#161b2e", "padding": "10px", "borderRadius": "8px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "10px"},
                    ),
                    html.Div(
                        [
                            html.Div(id="dispatch-card", style={"flex": "1", "backgroundColor": "#161b2e", "padding": "10px", "borderRadius": "8px", "color": "#ddd"}),
                            html.Div(id="vms-sign", style={"flex": "1", "backgroundColor": "black", "color": "#f9d342", "padding": "16px", "borderRadius": "8px", "fontFamily": "monospace", "fontSize": "22px"}),
                        ],
                        style={"display": "flex", "gap": "10px", "marginTop": "10px"},
                    ),
                    html.Div(
                        [
                            html.Button("Replay Analysis Segment", id="replay-analysis-btn", n_clicks=0),
                            html.Button("✅ Confirm AI Verdict", id="confirm-ai-btn", n_clicks=0, style={"marginLeft": "6px"}),
                            html.Button("✏️ Modify Verdict", id="modify-ai-btn", n_clicks=0, style={"marginLeft": "6px"}),
                            html.Button("🛠 Correct AI", id="correct-ai-btn", n_clicks=0, style={"marginLeft": "6px"}),
                            html.Div(id="operator-feedback-status", style={"marginTop": "8px", "color": "#ddd"}),
                        ],
                        style={"marginTop": "12px"},
                    ),
                    html.H4("Incident History", style={"color": "white", "marginTop": "18px"}),
                    dcc.Input(id="incident-search-input", type="text", placeholder="Search incidents (type, hazard, timestamp)...", style={"width": "100%", "padding": "8px", "marginBottom": "8px"}),
                    html.Div(id="incident-history-container", style={"maxHeight": "240px", "overflowY": "auto", "display": "flex", "flexDirection": "column", "gap": "8px"}),
                    dcc.Download(id="incident-pdf-download"),
                ],
            ),
            dcc.Store(id="ui-phase-store", data={"phase": "landing", "started_at": None}),
            dcc.Store(id="selected-incident-store"),
            dcc.Interval(id="refresh-interval", interval=2000, n_intervals=0),
            dcc.Interval(id="incident-log-interval", interval=2000, n_intervals=0),
            dcc.Interval(id="processing-interval", interval=500, n_intervals=0, disabled=True),
            dbc.Modal(
                id="replay-modal",
                is_open=False,
                children=html.Div(
                    [
                        html.H4("Replay (5-second segment)"),
                        html.Video(id="replay-video", controls=True, preload="auto", playsInline=True, style={"width": "100%", "backgroundColor": "#000", "minHeight": "200px"}),
                        dcc.Textarea(id="modify-verdict-notes", placeholder="Enter operator correction notes here...", style={"width": "100%", "height": "90px", "marginTop": "10px"}),
                        html.Div([html.Button("Save Notes", id="save-notes-btn", n_clicks=0), html.Span(id="save-notes-status", style={"marginLeft": "10px", "color": "#1f4b99"})], style={"marginTop": "8px", "display": "flex", "alignItems": "center"}),
                        html.Button("Close", id="close-replay-modal-btn", n_clicks=0),
                    ],
                    style={"backgroundColor": "white", "padding": "10px"},
                ),
            ),
            dbc.Modal(
                id="correct-ai-modal",
                is_open=False,
                children=html.Div(
                    [
                        html.H4("Correct AI Classification"),
                        dcc.Dropdown(
                            id="corrected-incident-type",
                            options=[
                                {"label": "COLLISION", "value": "COLLISION"},
                                {"label": "ACTIVE_HAZARD", "value": "ACTIVE_HAZARD"},
                                {"label": "DISABLED_VEHICLE", "value": "DISABLED_VEHICLE"},
                                {"label": "UNDER_REVIEW", "value": "UNDER_REVIEW"},
                                {"label": "NEAR_MISS", "value": "NEAR_MISS"},
                            ],
                            placeholder="Select corrected incident type",
                            style={"marginBottom": "8px"},
                        ),
                        dcc.Dropdown(
                            id="corrected-severity",
                            options=[
                                {"label": "LOW", "value": "low"},
                                {"label": "MEDIUM", "value": "med"},
                                {"label": "HIGH", "value": "high"},
                            ],
                            placeholder="Select corrected severity",
                            style={"marginBottom": "8px"},
                        ),
                        html.Button("Save Correction", id="save-correction-btn", n_clicks=0),
                        html.Button("Close", id="close-correct-ai-modal-btn", n_clicks=0, style={"marginLeft": "8px"}),
                        html.Div(id="correct-ai-status", style={"marginTop": "8px", "fontWeight": "bold"}),
                    ],
                    style={"backgroundColor": "white", "padding": "12px"},
                ),
            ),
        ],
        style={"fontFamily": "Inter, sans-serif"},
    )


app = dash.Dash(
    __name__,
    title="CHART Traffic Incident Management",
    update_title="Loading...",
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.BOOTSTRAP, "https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"],
)
app.layout = create_layout()


@app.callback(Output("uploaded-file-details", "children"), [Input("media-upload", "filename"), Input("media-upload", "contents"), Input("media-path-input", "value")])
def show_uploaded_file_details(filename, contents, media_path):
    if media_path:
        p = Path(media_path)
        if p.exists() and p.is_file():
            try:
                size = _format_bytes(p.stat().st_size)
            except Exception:
                size = "unknown size"
            return f"Selected path: {p.name} | {size}"
        return f"Path entered: {media_path} (not found)"
    if filename and contents:
        try:
            _, content_string = contents.split(",", 1)
            size_raw = int((len(content_string) * 3) / 4)
            return f"Uploaded file: {filename} | {_format_bytes(size_raw)}"
        except Exception:
            return f"Uploaded file: {filename}"
    return "No file uploaded yet."


@app.callback(
    [Output("run-analysis-btn", "disabled"), Output("upload-verify-progress", "style"), Output("upload-verify-progress", "value"), Output("upload-verify-progress", "label"), Output("upload-verify-progress", "color")],
    [Input("media-upload", "filename"), Input("media-upload", "contents"), Input("media-path-input", "value")],
)
def update_upload_verification(filename, contents, media_path):
    confirmed = False
    label = "Awaiting upload..."
    if media_path:
        p = Path(media_path)
        if p.exists() and p.is_file():
            confirmed = True
            label = f"Path confirmed: {p.name}"
        else:
            label = "Path not found"
    elif filename and contents:
        confirmed = True
        label = f"Uploaded: {filename}"
    if confirmed:
        return False, {"height": "20px", "display": "block", "marginBottom": "8px"}, 100, label, "success"
    return True, {"height": "20px", "display": "none", "marginBottom": "8px"}, 0, label, "secondary"


@app.callback(Output("header-status-bar", "children"), Input("refresh-interval", "n_intervals"))
def update_header(_):
    return f"Live Status | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Live Incidents: {len(_read_live_incidents())}"


@app.callback(
    [
        Output("replay-modal", "is_open"),
        Output("correct-ai-modal", "is_open"),
        Output("correct-ai-status", "children"),
    ],
    [
        Input("replay-analysis-btn", "n_clicks"),
        Input("action-required-btn", "n_clicks"),
        Input("modify-ai-btn", "n_clicks"),
        Input("close-replay-modal-btn", "n_clicks"),
        Input("correct-ai-btn", "n_clicks"),
        Input("close-correct-ai-modal-btn", "n_clicks"),
        Input("save-correction-btn", "n_clicks"),
    ],
    [
        State("replay-modal", "is_open"),
        State("correct-ai-modal", "is_open"),
        State("corrected-incident-type", "value"),
        State("corrected-severity", "value"),
        State("selected-incident-store", "data"),
    ],
    prevent_initial_call=True,
)
def modal_and_correction_router(
    replay_clicks,
    action_clicks,
    modify_clicks,
    close_replay,
    correct_clicks,
    close_correct,
    save_correction,
    replay_open,
    correct_open,
    corrected_type,
    corrected_severity,
    selected,
):
    trig = callback_context.triggered_id if callback_context.triggered else None
    status = ""
    if trig in {"replay-analysis-btn", "action-required-btn", "modify-ai-btn"}:
        replay_open = True
    elif trig == "close-replay-modal-btn":
        replay_open = False
    if trig == "correct-ai-btn":
        correct_open = True
    elif trig == "close-correct-ai-modal-btn":
        correct_open = False
    elif trig == "save-correction-btn":
        incident = selected or (_read_live_incidents()[-1] if _read_live_incidents() else {})
        if incident:
            _append_operator_correction(
                {
                    "timestamp": datetime.now().isoformat(),
                    "montage_path": str(incident.get("montage_path", "")),
                    "hero_frame_path": str(incident.get("hero_frame_path", "")),
                    "original_incident_type": str(incident.get("incident_type", "unknown")),
                    "original_severity": str(incident.get("severity", "unknown")),
                    "corrected_incident_type": str(corrected_type or incident.get("incident_type", "unknown")),
                    "corrected_severity": str(corrected_severity or incident.get("severity", "unknown")),
                }
            )
            status = "Correction saved for model fine-tuning."
    return replay_open, correct_open, status


@app.callback(Output("operator-feedback-status", "children"), [Input("confirm-ai-btn", "n_clicks"), Input("modify-ai-btn", "n_clicks")], [State("selected-incident-store", "data")])
def operator_feedback(confirm_clicks, modify_clicks, selected):
    trig = callback_context.triggered_id if callback_context.triggered else None
    if trig not in {"confirm-ai-btn", "modify-ai-btn"} or not selected:
        return ""
    if trig == "confirm-ai-btn":
        _append_feedback({"timestamp": datetime.now().isoformat(), "action": "confirm", "incident": selected})
        return "Saved: AI verdict confirmed."
    _append_feedback({"timestamp": datetime.now().isoformat(), "action": "modify", "original": selected, "human_corrected": {"severity": "med", "resources": ["CHART Patrol", "DOT"]}})
    return "Saved: human-modified verdict logged for training."


@app.callback(Output("save-notes-status", "children"), Input("save-notes-btn", "n_clicks"), [State("modify-verdict-notes", "value"), State("selected-incident-store", "data")], prevent_initial_call=True)
def save_operator_notes(n_clicks, notes_value, selected):
    if not n_clicks:
        return ""
    notes = str(notes_value or "").strip()
    if not notes:
        return "Enter notes before saving."
    _append_feedback({"timestamp": datetime.now().isoformat(), "action": "save_notes", "notes": notes, "incident": selected or {}})
    return "Notes saved to training feedback log."


@app.callback(Output("incident-pdf-download", "data"), Input("export-chart-report-btn", "n_clicks"), State("selected-incident-store", "data"), prevent_initial_call=True)
def export_pdf(n_clicks, selected):
    if not n_clicks:
        return no_update
    inc = selected or (_read_live_incidents()[-1] if _read_live_incidents() else None)
    if not inc:
        return no_update
    pdf_path = generate_incident_pdf(inc)
    return dcc.send_file(pdf_path) if pdf_path else no_update


server = app.server

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
