"""Workflow-driven callbacks for TIM dashboard."""

import base64
import json
from datetime import datetime
from pathlib import Path

from dash import ALL, Input, Output, State, callback_context, dcc, html, no_update
import plotly.graph_objects as go

from utils.pdf_generator import generate_incident_pdf

LIVE_INCIDENTS_FILE = Path(__file__).parent / "data" / "live_incidents.json"
TRAINING_FEEDBACK_FILE = Path(__file__).parent / "data" / "training_feedback_log.json"
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


def register_callbacks(app):
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
        ],
        [
            Input("run-analysis-btn", "n_clicks"),
            Input("refresh-interval", "n_intervals"),
            Input({"type": "view-report-btn", "index": ALL}, "n_clicks"),
        ],
        [State("ui-phase-store", "data"), State("selected-incident-store", "data")],
    )
    def manage_phase(run_clicks, n_intervals, _view_clicks, phase_data, selected):
        phase_data = phase_data or {"phase": "landing", "started_at": None}
        selected = selected or {}
        incidents = _read_live_incidents()
        trig = callback_context.triggered_id if callback_context.triggered else None

        if trig == "run-analysis-btn" and run_clicks:
            phase_data = {"phase": "processing", "started_at": n_intervals}

        if isinstance(trig, dict) and trig.get("type") == "view-report-btn":
            idx = int(trig.get("index", -1))
            rev = list(reversed(incidents))
            if 0 <= idx < len(rev):
                selected = rev[idx]
                phase_data = {"phase": "verdict", "started_at": phase_data.get("started_at")}

        logs = ["Awaiting analysis run..."]
        if phase_data.get("phase") == "processing":
            start = phase_data.get("started_at", n_intervals)
            step = max(0, n_intervals - start)
            upto = min(len(PROCESS_STAGES), step + 1)
            logs = [f"[{i+1}/{len(PROCESS_STAGES)}] {PROCESS_STAGES[i]}" for i in range(upto)]
            if incidents:
                selected = incidents[-1]
                phase_data["phase"] = "verdict"

        if phase_data.get("phase") == "verdict":
            landing_style = {"display": "none"}
            verdict_style = {"display": "block", "backgroundColor": "#0f1320", "minHeight": "calc(100vh - 90px)", "padding": "14px"}
        else:
            landing_style = {"minHeight": "calc(100vh - 90px)", "backgroundColor": "#0c0f17", "display": "block"}
            verdict_style = {"display": "none"}
        return phase_data, "\n".join(logs), selected, landing_style, verdict_style

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