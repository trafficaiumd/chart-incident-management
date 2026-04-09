"""Workflow-driven callbacks for TIM dashboard."""

import base64
import json
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
import plotly.graph_objects as go

from utils.pdf_generator import generate_incident_pdf

LIVE_INCIDENTS_FILE = Path(__file__).parent / "data" / "live_incidents.json"
TRAINING_FEEDBACK_FILE = Path(__file__).parent / "data" / "training_feedback_log.json"
STATUS_FILE = Path(__file__).parent / "data" / "status.json"
PROCESS_META_FILE = Path(__file__).parent / "data" / "active_process.json"
PROJECT_ROOT = Path(__file__).parent.parent
PROCESS_STAGES = [
    "Detect initial impact (YOLO26 Guard)...",
    "Building temporal context window...",
    "Generating temporal montage...",
    "Sending montage to Gemma 4 Judge...",
    "Parsing JSON verdict and dispatch plan...",
]


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


def _incident_confidence(inc):
    if "confidence" in inc:
        try:
            return float(inc["confidence"])
        except Exception:
            pass
    sev = str(inc.get("severity", "med")).lower()
    return {"high": 0.9, "med": 0.72, "low": 0.55}.get(sev, 0.7)


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
    """
    Start Guard/Judge pipeline in background for a single media file.
    """
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


def register_callbacks(app):
    @app.callback(
        Output("uploaded-file-details", "children"),
        [Input("media-upload", "filename"), Input("media-upload", "contents"), Input("media-path-input", "value")],
    )
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
        [
            Output("run-analysis-btn", "disabled"),
            Output("upload-verify-progress", "style"),
            Output("upload-verify-progress", "value"),
            Output("upload-verify-progress", "label"),
            Output("upload-verify-progress", "color"),
        ],
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
        count = len(_read_live_incidents())
        return f"Live Status | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Live Incidents: {count}"

    @app.callback(
        [
            Output("ui-phase-store", "data"),
            Output("processing-log", "children"),
            Output("selected-incident-store", "data"),
            Output("landing-container", "style"),
            Output("verdict-container", "style"),
            Output("upload-section", "style"),
            Output("processing-tracker-section", "style"),
            Output("processing-interval", "disabled"),
            Output("processing-progress-bar", "value"),
            Output("processing-progress-bar", "label"),
            Output("processing-progress-bar", "color"),
            Output("processing-progress-label", "children"),
            Output("landing-logo", "style"),
        ],
        [
            Input("run-analysis-btn", "n_clicks"),
            Input("refresh-interval", "n_intervals"),
            Input("processing-interval", "n_intervals"),
            Input({"type": "view-report-btn", "index": ALL}, "n_clicks"),
        ],
        [
            State("ui-phase-store", "data"),
            State("selected-incident-store", "data"),
            State("media-upload", "filename"),
            State("media-upload", "contents"),
            State("media-path-input", "value"),
        ],
    )
    def manage_phase(
        run_clicks,
        n_intervals,
        _proc_tick,
        _view_clicks,
        phase_data,
        selected,
        upload_filename,
        upload_contents,
        media_path,
    ):
        phase_data = phase_data or {"phase": "landing", "started_at": None}
        selected = selected or {}
        incidents = _read_live_incidents()
        trig = callback_context.triggered_id if callback_context.triggered else None

        if trig == "run-analysis-btn" and run_clicks:
            media_file_path = None
            if media_path:
                p = Path(media_path)
                if p.exists() and p.is_file():
                    media_file_path = str(p.resolve())
            elif upload_filename and upload_contents:
                try:
                    _, content_string = upload_contents.split(",", 1)
                    decoded = base64.b64decode(content_string)
                    suffix = Path(upload_filename).suffix or ".mp4"
                    up_dir = Path(__file__).parent / "data" / "uploads"
                    up_dir.mkdir(parents=True, exist_ok=True)
                    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir=str(up_dir)) as tf:
                        tf.write(decoded)
                        media_file_path = tf.name
                except Exception:
                    media_file_path = None

            if media_file_path:
                # Reset output state for a clean run.
                LIVE_INCIDENTS_FILE.parent.mkdir(parents=True, exist_ok=True)
                LIVE_INCIDENTS_FILE.write_text("[]\n", encoding="utf-8")
                STATUS_FILE.write_text(
                    json.dumps(
                        {"state": "INGESTING", "detail": "Extracting Video Frames...", "ts": datetime.now().isoformat()},
                        indent=2,
                    ),
                    encoding="utf-8",
                )
                # Avoid duplicate launch if a prior run is still alive.
                already_running = False
                if PROCESS_META_FILE.exists():
                    try:
                        pm = json.loads(PROCESS_META_FILE.read_text(encoding="utf-8"))
                        already_running = _is_pid_running(int(pm.get("pid", -1)))
                    except Exception:
                        already_running = False
                if not already_running:
                    _start_pipeline_for_media(media_file_path)
                phase_data = {"phase": "processing", "started_at": n_intervals}
            else:
                logs = ["Unable to start analysis: media not found or upload decode failed."]

        if isinstance(trig, dict) and trig.get("type") == "view-report-btn":
            idx = int(trig.get("index", -1))
            rev = list(reversed(incidents))
            if 0 <= idx < len(rev):
                selected = rev[idx]
                phase_data = {"phase": "verdict", "started_at": phase_data.get("started_at")}

        logs = ["Awaiting analysis run..."]
        prog_value, prog_label, prog_color, stage_label = 0, "0%", "info", "Idle"
        logo_style = {
            "width": "240px",
            "marginBottom": "18px",
            "display": "block",
            "marginLeft": "auto",
            "marginRight": "auto",
        }
        if phase_data.get("phase") == "processing":
            state_map = {
                "INGESTING": (25, "Extracting Video Frames...", "warning"),
                "GUARDING": (50, "YOLO26 Perception Active...", "info"),
                "JUDGING": (75, "Gemma 4 reasoning on physics...", "primary"),
                "FINALIZING": (100, "Generating CHART Report...", "success"),
            }
            status = _read_status()
            st = str(status.get("state", "INGESTING")).upper()
            prog_value, stage_label, prog_color = state_map.get(st, (25, "Extracting Video Frames...", "warning"))
            prog_label = f"{prog_value}%"
            logs = [f"[{st}] {stage_label}"]
            if prog_value < 100:
                logo_style["animation"] = "pulse 1.2s infinite"
            if prog_value >= 100:
                if incidents:
                    selected = incidents[-1]
                phase_data["phase"] = "verdict"

        if phase_data.get("phase") == "verdict":
            landing_style = {"display": "none"}
            verdict_style = {"display": "block", "backgroundColor": "#0f1320", "minHeight": "calc(100vh - 90px)", "padding": "14px"}
            upload_style = {"display": "none"}
            tracker_style = {"display": "none"}
            proc_disabled = True
        else:
            landing_style = {"minHeight": "calc(100vh - 90px)", "backgroundColor": "#0c0f17", "display": "block"}
            verdict_style = {"display": "none"}
            processing = phase_data.get("phase") == "processing"
            upload_style = {"display": "none"} if processing else {"display": "block"}
            tracker_style = {"display": "block", "marginBottom": "10px"} if processing else {"display": "none", "marginBottom": "10px"}
            proc_disabled = not processing
        return (
            phase_data,
            "\n".join(logs),
            selected,
            landing_style,
            verdict_style,
            upload_style,
            tracker_style,
            proc_disabled,
            prog_value,
            prog_label,
            prog_color,
            stage_label,
            logo_style,
        )

    @app.callback(
        [
            Output("verdict-banner", "children"),
            Output("verdict-banner", "style"),
            Output("evidence-montage-image", "src"),
            Output("ai-insight-text", "children"),
            Output("confidence-gauge", "figure"),
            Output("classification-badges", "children"),
            Output("dispatch-card", "children"),
            Output("vms-sign", "children"),
            Output("action-required-banner", "children"),
            Output("action-required-banner", "style"),
            Output("replay-video", "src"),
            Output("incident-history-container", "children"),
        ],
        [Input("refresh-interval", "n_intervals"), Input("incident-log-interval", "n_intervals")],
        [State("selected-incident-store", "data")],
    )
    def render_verdict(_a, _b, selected):
        incidents = _read_live_incidents()
        inc = selected or (incidents[-1] if incidents else {})
        incident_type = str(inc.get("incident_type", "UNKNOWN")).upper()
        severity = str(inc.get("severity", "med")).lower()
        hazards = [str(x) for x in inc.get("hazards", [])]
        vehicles = [str(x) for x in inc.get("vehicle_types", [])]
        confidence = _incident_confidence(inc)

        red = incident_type == "COLLISION"
        banner_style = {
            "padding": "12px",
            "fontSize": "20px",
            "fontWeight": "bold",
            "borderRadius": "8px",
            "marginBottom": "10px",
            "color": "white",
            "backgroundColor": "#8B0000" if red else "#B36B00",
        }
        insight = inc.get("verdict", "No AI insight yet.")
        img_src = _image_to_b64(inc.get("montage_path", ""))
        gauge = go.Figure(
            go.Indicator(mode="gauge+number", value=confidence * 100.0, gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#B00020" if confidence > 0.85 else "#2E7D32"}})
        )
        badges = [html.Span(v, style={"backgroundColor": "#2c3455", "color": "white", "padding": "3px 8px", "borderRadius": "12px", "marginRight": "6px"}) for v in (vehicles + hazards)]
        dispatch = html.Div(
            [
                html.H4("Dispatch Plan", style={"marginTop": 0}),
                html.P("Gemma 4 observed fire/hazard cues." if "fire" in [h.lower() for h in hazards] else "Deploy standard CHART response."),
                html.Ul([html.Li(f) for f in [f"Type: {incident_type}", f"Severity: {severity}", f"Lanes Blocked: {inc.get('lanes_blocked', 0)}"]]),
            ]
        )
        vms_txt = "ACCIDENT AHEAD\nREDUCE SPEED\nFOLLOW MERGE"
        if incident_type == "ACTIVE_HAZARD":
            vms_txt = "HAZARD AHEAD\nPROCEED WITH CAUTION\nEXPECT DELAYS"
        elif incident_type == "DISABLED_VEHICLE":
            vms_txt = "STALLED VEHICLE AHEAD\nMOVE OVER\nEXPECT DELAYS"

        uncertain = confidence < 0.70 or bool(inc.get("uncertainty", False))
        ar_style = {
            "display": "block" if uncertain else "none",
            "backgroundColor": "#ffcc00",
            "padding": "10px",
            "fontWeight": "bold",
            "borderRadius": "8px",
            "marginBottom": "8px",
        }
        ar_txt = "⚠️ ACTION REQUIRED: AI confidence below threshold, operator review needed."

        history_cards = []
        for i, item in enumerate(reversed(incidents[-30:])):
            t = str(item.get("timestamp", "unknown"))
            it = str(item.get("incident_type", "UNKNOWN"))
            pill_color = "#8B0000" if it == "COLLISION" else "#B36B00"
            history_cards.append(
                html.Div(
                    [
                        html.Span(it, style={"backgroundColor": pill_color, "color": "white", "padding": "2px 8px", "borderRadius": "10px", "marginRight": "6px"}),
                        html.Span(t, style={"color": "#ddd", "marginRight": "8px"}),
                        html.Button("View Report", id={"type": "view-report-btn", "index": i}, n_clicks=0),
                    ],
                    style={"backgroundColor": "#1b2137", "padding": "8px", "borderRadius": "6px"},
                )
            )

        return (
            f"{incident_type} Decision Support",
            banner_style,
            img_src,
            insight,
            gauge,
            badges,
            dispatch,
            vms_txt,
            ar_txt,
            ar_style,
            inc.get("replay_clip_path", ""),
            history_cards or [html.Div("No incidents yet.", style={"color": "#aaa"})],
        )

    @app.callback(
        Output("replay-modal", "is_open"),
        [Input("replay-analysis-btn", "n_clicks"), Input("close-replay-modal-btn", "n_clicks")],
        [State("replay-modal", "is_open")],
    )
    def replay_modal(open_clicks, close_clicks, is_open):
        trig = callback_context.triggered_id if callback_context.triggered else None
        if trig == "replay-analysis-btn":
            return True
        if trig == "close-replay-modal-btn":
            return False
        return is_open

    @app.callback(
        Output("operator-feedback-status", "children"),
        [Input("confirm-ai-btn", "n_clicks"), Input("modify-ai-btn", "n_clicks")],
        [State("selected-incident-store", "data")],
    )
    def operator_feedback(confirm_clicks, modify_clicks, selected):
        trig = callback_context.triggered_id if callback_context.triggered else None
        if trig not in {"confirm-ai-btn", "modify-ai-btn"} or not selected:
            return ""
        if trig == "confirm-ai-btn":
            _append_feedback({"timestamp": datetime.now().isoformat(), "action": "confirm", "incident": selected})
            return "Saved: AI verdict confirmed."
        corrected = {"severity": "med", "resources": ["CHART Patrol", "DOT"]}
        _append_feedback(
            {
                "timestamp": datetime.now().isoformat(),
                "action": "modify",
                "original": selected,
                "human_corrected": corrected,
            }
        )
        return "Saved: human-modified verdict logged for training."

    @app.callback(Output("incident-pdf-download", "data"), Input("export-chart-report-btn", "n_clicks"), State("selected-incident-store", "data"), prevent_initial_call=True)
    def export_pdf(n_clicks, selected):
        if not n_clicks:
            return no_update
        inc = selected or (_read_live_incidents()[-1] if _read_live_incidents() else None)
        if not inc:
            return no_update
        pdf_path = generate_incident_pdf(inc)
        return dcc.send_file(pdf_path)