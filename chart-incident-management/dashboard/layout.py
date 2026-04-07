"""Two-phase TIM dashboard layout."""

from datetime import datetime
import dash_bootstrap_components as dbc
from dash import dcc, html


def create_header():
    return html.Div(
        [
            html.Div(
                [
                    html.Img(src="/assets/image_0.png", style={"height": "52px", "marginRight": "12px"}),
                    html.H2("CHART Traffic Incident Management", style={"margin": 0, "color": "white"}),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            html.Div(
                id="header-status-bar",
                style={
                    "marginTop": "8px",
                    "fontSize": "12px",
                    "color": "white",
                    "borderTop": "1px solid rgba(255,255,255,0.5)",
                    "paddingTop": "6px",
                },
                children=f"Live Status | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Live Incidents: 0",
            ),
        ],
        style={"backgroundColor": "#3D1A78", "padding": "12px 18px", "position": "sticky", "top": 0, "zIndex": 10},
    )


def landing_page():
    return html.Div(
        id="landing-container",
        children=[
            html.Div(
                [
                    html.Img(src="/assets/image_0.png", style={"width": "240px", "marginBottom": "18px"}),
                    html.H3("Analyze Media", style={"color": "white"}),
                    dcc.Upload(
                        id="media-upload",
                        children=html.Div(["Drag and drop or ", html.A("select file")]),
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
                        },
                        multiple=False,
                    ),
                    dcc.Input(
                        id="media-path-input",
                        type="text",
                        placeholder="Path to media file",
                        style={"width": "100%", "padding": "10px", "marginBottom": "10px"},
                    ),
                    html.Button(
                        "Run Analysis",
                        id="run-analysis-btn",
                        n_clicks=0,
                        style={
                            "width": "100%",
                            "padding": "12px",
                            "fontWeight": "bold",
                            "color": "white",
                            "border": "none",
                            "borderRadius": "8px",
                            "background": "linear-gradient(90deg,#7f5af0,#2cb67d)",
                            "boxShadow": "0 0 20px rgba(127,90,240,0.6)",
                        },
                    ),
                    html.Pre(
                        id="processing-log",
                        style={
                            "marginTop": "12px",
                            "backgroundColor": "#121212",
                            "color": "#9ef0b8",
                            "padding": "10px",
                            "borderRadius": "8px",
                            "minHeight": "120px",
                            "whiteSpace": "pre-wrap",
                        },
                    ),
                ],
                style={"width": "min(680px, 94%)", "margin": "40px auto"},
            )
        ],
        style={"minHeight": "calc(100vh - 90px)", "backgroundColor": "#0c0f17", "display": "block"},
    )


def verdict_page():
    return html.Div(
        id="verdict-container",
        style={"display": "none", "backgroundColor": "#0f1320", "minHeight": "calc(100vh - 90px)", "padding": "14px"},
        children=[
            html.Div(id="verdict-banner"),
            html.Div(
                [
                    html.Button("Export Official CHART Report", id="export-chart-report-btn", n_clicks=0),
                ],
                style={"display": "flex", "justifyContent": "flex-end", "margin": "8px 0"},
            ),
            html.Div(id="action-required-banner", style={"display": "none"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Temporal Montage Evidence", style={"color": "white"}),
                            html.Img(id="evidence-montage-image", style={"width": "100%", "borderRadius": "8px"}),
                        ],
                        style={"flex": "2", "backgroundColor": "#161b2e", "padding": "10px", "borderRadius": "8px"},
                    ),
                    html.Div(
                        [
                            html.H4("AI Insight", style={"color": "white"}),
                            html.Pre(id="ai-insight-text", style={"whiteSpace": "pre-wrap", "color": "#ddd"}),
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
                    html.Div(id="operator-feedback-status", style={"marginTop": "8px", "color": "#ddd"}),
                ],
                style={"marginTop": "12px"},
            ),
            html.H4("Incident History", style={"color": "white", "marginTop": "18px"}),
            html.Div(id="incident-history-container", style={"maxHeight": "240px", "overflowY": "auto", "display": "flex", "flexDirection": "column", "gap": "8px"}),
            dcc.Download(id="incident-pdf-download"),
        ],
    )


def create_layout():
    return html.Div(
        [
            create_header(),
            landing_page(),
            verdict_page(),
            dcc.Store(id="ui-phase-store", data={"phase": "landing", "started_at": None}),
            dcc.Store(id="selected-incident-store"),
            dcc.Interval(id="refresh-interval", interval=2000, n_intervals=0),
            dcc.Interval(id="incident-log-interval", interval=2000, n_intervals=0),
            dbc.Modal(
                id="replay-modal",
                is_open=False,
                children=html.Div(
                    [
                        html.H4("Replay (3-second segment)"),
                        html.Video(id="replay-video", controls=True, style={"width": "100%"}),
                        html.Button("Close", id="close-replay-modal-btn", n_clicks=0),
                    ],
                    style={"backgroundColor": "white", "padding": "10px"},
                ),
            ),
        ],
        style={"fontFamily": "Inter, sans-serif"},
    )