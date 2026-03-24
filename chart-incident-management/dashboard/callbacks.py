# dashboard/callbacks.py
"""
Dashboard callbacks for interactivity
"""

import base64
import tempfile
from dash import Input, Output, State, callback_context, no_update, html
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from decision_engine.pipeline import DecisionPipeline
from output_layer.chart_formatter import CHARTFormatter
from layout import (
    create_severity_card, create_resources_card,
    create_vms_card, create_risk_card
)
from ai_layer.analyzer import analyze

# Initialize engines
pipeline = DecisionPipeline()
formatter = CHARTFormatter()

# Sample incidents data (replace with your actual data source)
INCIDENTS = {
    'crash_1': {
        'model_output': {
            'incident_type': 'crash',
            'confidence': 0.95,
            'vehicles': {'total_count': 4},
            'lanes': {'blocked': [1, 2]},
            'hazards': {'debris': True},
            'traffic': {'state': 'stopped'},
            'emergency_response': {'police_present': False}
        },
        'context': {'location': 'I-95 NB mile 27.4', 'time_of_day': 'day', 'weather': 'clear'}
    },
    'debris_1': {
        'model_output': {
            'incident_type': 'debris',
            'confidence': 0.85,
            'vehicles': {'total_count': 0},
            'lanes': {'blocked': [3]},
            'hazards': {'debris': True},
            'traffic': {'state': 'slowing'},
            'emergency_response': {}
        },
        'context': {'location': 'I-695 SB mm 12.1', 'time_of_day': 'day', 'weather': 'rain'}
    }
}

def register_callbacks(app):
    """Register all callbacks."""

    @app.callback(
        [Output('severity-card-container', 'children'),
         Output('resources-card-container', 'children'),
         Output('vms-card-container', 'children'),
         Output('risk-card-container', 'children'),
         Output('incident-summary', 'children'),
         Output('incident-history', 'children'),
         Output('incident-data-store', 'data'),
         Output('detail-incident-type', 'children'),
         Output('detail-vehicles', 'children'),
         Output('detail-lanes', 'children'),
         Output('detail-hazards', 'children'),
         Output('analyze-status', 'children'),
        ],
        [Input('incident-selector', 'value'),
         Input('refresh-interval', 'n_intervals'),
         Input('run-analysis-btn', 'n_clicks'),
        ],
        [State('media-upload', 'contents'),
         State('media-upload', 'filename'),
         State('media-path-input', 'value'),
        ],
    )
    def update_dashboard(
        incident_id,
        n_intervals,
        run_clicks,
        upload_contents,
        upload_filename,
        media_path,
    ):
        """Update dashboard: either from incident selector or from Run Analysis."""
        ctx = callback_context
        triggered = ctx.triggered_id if ctx.triggered else None

        # Run Analysis path: load media, build grid, call analyzer, run pipeline
        if triggered == 'run-analysis-btn' and run_clicks:
            media_file_path = None

            if media_path and Path(media_path).exists():
                media_file_path = media_path
            elif upload_contents and upload_filename:
                # Decode base64 upload and save to temp file
                try:
                    content_type, content_string = upload_contents.split(',')
                    decoded = base64.b64decode(content_string)
                    ext = Path(upload_filename).suffix or '.jpg'
                    tmp = tempfile.NamedTemporaryFile(suffix=ext, delete=False)
                    tmp.write(decoded)
                    tmp.close()
                    media_file_path = tmp.name
                except Exception as e:
                    status = html.Span(f"Upload error: {e}", style={"color": "red"})
                    return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, status

            if media_file_path:
                try:
                    model_output = analyze(media_file_path, location_text=Path(media_file_path).name)
                    context = {'location': 'From uploaded media', 'time_of_day': 'day', 'weather': 'clear'}
                    decision = pipeline.process(model_output, context)
                    status = html.Span("Analysis complete", style={"color": "green"})
                except Exception as e:
                    model_output = {
                        'incident_type': 'unknown', 'confidence': 0,
                        'vehicles': {}, 'lanes': {}, 'hazards': {},
                        'traffic': {}, 'emergency_response': {},
                    }
                    context = {'location': 'unknown', 'time_of_day': 'day', 'weather': 'clear'}
                    decision = pipeline.process(model_output, context)
                    status = html.Span(f"Analysis error: {e}", style={"color": "red"})

                incident_block = decision['incident']
                vehicles = model_output.get('vehicles', {})
                lanes = model_output.get('lanes', {})
                hazards = model_output.get('hazards', {})

                def show_unknown(value):
                    if isinstance(value, str) and value.lower() in ("unknown", "none"):
                        return "UNKNOWN"
                    return value

                detail_incident_type = f"Type: {show_unknown(incident_block.get('type'))}"
                detail_vehicles = f"Vehicles (total): {vehicles.get('total_count', 0)}"
                detail_lanes = f"Lanes blocked: {lanes.get('blocked', []) or 'NONE'}"
                detail_hazards = (
                    f"Hazards - fire={hazards.get('fire', False)}, "
                    f"smoke={hazards.get('smoke', False)}, "
                    f"debris={hazards.get('debris', False)}, "
                    f"injuries_visible={hazards.get('injuries_visible', False)}"
                )
                chart_data = formatter.format(decision)
                severity_card = create_severity_card(decision['severity'])
                resources_card = create_resources_card(decision['resources'])
                vms_card = create_vms_card(decision['vms'])
                risk_card = create_risk_card(decision['secondary_risk'])
                summary = html.Div([
                    html.P(f"Location: {context.get('location', 'unknown')}"),
                    html.P(f"Type: {decision['incident']['type']}"),
                    html.P(f"Confidence: {decision['incident']['confidence']:.0%}"),
                    html.P(f"Human Review: {'⚠️ Required' if decision['requires_human_review'] else '✅ Auto-approved'}")
                ])
                history = html.Ul([
                    html.Li(f"10:45 AM - I-95 Crash (RED)"),
                    html.Li(f"10:30 AM - I-695 Debris (YELLOW)"),
                    html.Li(f"10:15 AM - I-270 Fire (RED)")
                ])
                return (
                    severity_card, resources_card, vms_card, risk_card,
                    summary, history, chart_data,
                    detail_incident_type, detail_vehicles, detail_lanes, detail_hazards,
                    status,
                )
            # No file: show hint and fall through to incident selector
            status = html.Span("Select a file or enter a path", style={"color": "#888"})

        # Incident selector / interval path (default)
        incident = INCIDENTS.get(incident_id, INCIDENTS['crash_1'])
        decision = pipeline.process(incident['model_output'], incident['context'])
        incident_block = decision['incident']
        model_output = incident['model_output']
        vehicles = model_output.get('vehicles', {})
        lanes = model_output.get('lanes', {})
        hazards = model_output.get('hazards', {})

        def show_unknown(value):
            if isinstance(value, str) and value.lower() in ("unknown", "none"):
                return "UNKNOWN"
            return value

        detail_incident_type = f"Type: {show_unknown(incident_block.get('type'))}"
        detail_vehicles = f"Vehicles (total): {vehicles.get('total_count', 0)}"
        detail_lanes = f"Lanes blocked: {lanes.get('blocked', []) or 'NONE'}"
        detail_hazards = (
            f"Hazards - fire={hazards.get('fire', False)}, "
            f"smoke={hazards.get('smoke', False)}, "
            f"debris={hazards.get('debris', False)}, "
            f"injuries_visible={hazards.get('injuries_visible', False)}"
        )
        chart_data = formatter.format(decision)
        severity_card = create_severity_card(decision['severity'])
        resources_card = create_resources_card(decision['resources'])
        vms_card = create_vms_card(decision['vms'])
        risk_card = create_risk_card(decision['secondary_risk'])
        summary = html.Div([
            html.P(f"Location: {incident['context']['location']}"),
            html.P(f"Type: {decision['incident']['type']}"),
            html.P(f"Confidence: {decision['incident']['confidence']:.0%}"),
            html.P(f"Human Review: {'⚠️ Required' if decision['requires_human_review'] else '✅ Auto-approved'}")
        ])
        history = html.Ul([
            html.Li(f"10:45 AM - I-95 Crash (RED)"),
            html.Li(f"10:30 AM - I-695 Debris (YELLOW)"),
            html.Li(f"10:15 AM - I-270 Fire (RED)")
        ])
        status = html.Span("", style={"fontSize": "12px"})
        return (
            severity_card, resources_card, vms_card, risk_card,
            summary, history, chart_data,
            detail_incident_type, detail_vehicles, detail_lanes, detail_hazards,
            status,
        )
    
    @app.callback(
        Output('header-timestamp', 'children'),
        Input('refresh-interval', 'n_intervals')
    )
    def update_timestamp(n_intervals):
        """Update the timestamp in header"""
        return f"Real-time Incident Decision Support | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"