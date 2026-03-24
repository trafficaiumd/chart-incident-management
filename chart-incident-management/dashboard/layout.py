# dashboard/layout.py
"""
Dashboard layout components
"""

from dash import dcc, html
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
from datetime import datetime
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))


# Custom CSS
def get_custom_css():
    return html.Link(
        rel='stylesheet',
        href='/assets/style.css'
    )

# Header component
def create_header():
    return html.Div([
        html.Div([
            html.H1("🚦 CHART Traffic Incident Management", 
                   className="header-title"),
            html.P(id="header-timestamp",children=f"Real-time Incident Decision Support | Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  className="header-timestamp")
        ], className="header-container")
    ], className="header-wrapper")

# Severity card
def create_severity_card(severity_data=None):
    if severity_data is None:
        severity_data = {'level': 'GREEN', 'score': 0, 'reasons': []}
    
    level = severity_data['level']
    color_map = {'RED': '#dc3545', 'YELLOW': '#ffc107', 'GREEN': '#28a745'}
    color = color_map.get(level, '#6c757d')
    
    return html.Div([
        html.H3("Severity", className="card-title"),
        html.Div([
            html.Span(level, className=f"severity-badge severity-{level.lower()}"),
            html.Span(f"Score: {severity_data['score']}", className="severity-score")
        ], className="severity-header"),
        html.Ul([
            html.Li(reason) for reason in severity_data['reasons'][:3]
        ], className="reasons-list")
    ], className=f"card severity-card {level.lower()}-card", id="severity-card")

# Resources card
def create_resources_card(resources_data=None):
    if resources_data is None:
        resources_data = {'police': False, 'ems': False, 'fire': False, 
                         'tow': False, 'dot': False, 'hazmat': False, 'reasons': []}
    
    resource_icons = {
        'police': '👮', 'ems': '🚑', 'fire': '🚒', 
        'tow': '🛻', 'dot': '🛣️', 'hazmat': '☣️'
    }
    
    return html.Div([
        html.H3("Resources to Dispatch", className="card-title"),
        html.Div([
            html.Div([
                html.Span(resource_icons.get(resource, '•'), className="resource-icon"),
                html.Span(resource.replace('_', ' ').title(), className="resource-name")
            ], className=f"resource-item {'active' if needed else 'inactive'}")
            for resource, needed in resources_data.items() 
            if resource != 'reasons' and needed
        ], className="resources-grid"),
        html.Div([
            html.H4("Dispatch Reasons", className="subtitle"),
            html.Ul([html.Li(reason) for reason in resources_data.get('reasons', [])])
        ], className="reasons-section")
    ], className="card resources-card")

# VMS card
def create_vms_card(vms_data=None):
    if vms_data is None:
        vms_data = {'primary': 'NO INCIDENT', 'secondary': 'ALL LANES OPEN', 
                   'tertiary': 'DRIVE SAFELY', 'all_messages': []}
    
    return html.Div([
        html.H3("Variable Message Signs", className="card-title"),
        html.Div([
            html.Div([
                html.Div([
                    html.Div(vms_data['primary'], className="vms-line vms-primary"),
                    html.Div(vms_data['secondary'], className="vms-line vms-secondary"),
                    html.Div(vms_data['tertiary'], className="vms-line vms-tertiary")
                ], className="vms-display")
            ], className="vms-container")
        ], className="vms-wrapper")
    ], className="card vms-card")

def create_incident_details_card():
    return html.Div(
        [
            html.H3("Incident Details", className="card-title"),
            html.Ul(
                [
                    html.Li(id="detail-incident-type"),
                    html.Li(id="detail-vehicles"),
                    html.Li(id="detail-lanes"),
                    html.Li(id="detail-hazards"),
                ],
                className="details-list",
            ),
        ],
        className="card details-card",
    )
# Risk card
def create_risk_card(risk_data=None):
    if risk_data is None:
        risk_data = {'risk_level': 'LOW', 'risk_score': 0, 'factors': []}
    
    level = risk_data['risk_level']
    color_map = {'HIGH': '#dc3545', 'MEDIUM': '#ffc107', 'LOW': '#28a745'}
    color = color_map.get(level, '#6c757d')
    
    # Create gauge chart
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=risk_data['risk_score'],
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 14}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 40], 'color': '#e9f7e9'},
                {'range': [40, 70], 'color': '#fff3cd'},
                {'range': [70, 100], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': risk_data['risk_score']
            }
        }
    ))
    
    gauge.update_layout(height=200, margin=dict(l=10, r=10, t=30, b=10))
    
    return html.Div([
        html.H3("Secondary Crash Risk", className="card-title"),
        html.Div([
            html.Span(level, className=f"risk-badge risk-{level.lower()}"),
        ], className="risk-header"),
        dcc.Graph(figure=gauge, config={'displayModeBar': False}),
        html.Div([
            html.H4("Risk Factors", className="subtitle"),
            html.Ul([html.Li(factor) for factor in risk_data['factors'][:3]])
        ], className="factors-section")
    ], className=f"card risk-card {level.lower()}-card")

# Analyze Media panel
def create_analyze_media_panel():
    """Panel for loading image/video, building grid, running analysis."""
    return html.Div([
        html.H3("Analyze Media", className="card-title"),
        dcc.Upload(
            id="media-upload",
            children=html.Div([
                "Drag and drop or ",
                html.A("Select file", href="#"),
            ]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "marginBottom": "10px",
            },
            multiple=False,
        ),
        html.Div([
            html.Label("Or enter path:", style={"fontSize": "12px"}),
            dcc.Input(
                id="media-path-input",
                type="text",
                placeholder="e.g. tests/datasets/.../test/image.jpg",
                style={"width": "100%", "marginTop": "4px", "padding": "6px"},
            ),
        ], style={"marginBottom": "10px"}),
        html.Button(
            "Run Analysis",
            id="run-analysis-btn",
            n_clicks=0,
            style={
                "width": "100%",
                "padding": "10px",
                "backgroundColor": "#007bff",
                "color": "white",
                "border": "none",
                "borderRadius": "5px",
                "cursor": "pointer",
                "fontWeight": "bold",
            },
        ),
        html.Div(id="analyze-status", className="analyze-status", style={"marginTop": "8px", "fontSize": "12px"}),
    ], className="card analyze-media-card", style={"marginBottom": "16px"})


# Incident selector
def create_incident_selector(incidents=None):
    if incidents is None:
        incidents = [
            {'id': 'crash_1', 'label': 'I-95 NB Crash (RED)', 'type': 'crash'},
            {'id': 'debris_1', 'label': 'I-695 Debris (YELLOW)', 'type': 'debris'},
            {'id': 'fire_1', 'label': 'I-270 Fire (RED)', 'type': 'fire'}
        ]
    
    return html.Div([
        html.H3("Active Incidents", className="card-title"),
        dcc.Dropdown(
            id='incident-selector',
            options=[{'label': inc['label'], 'value': inc['id']} for inc in incidents],
            value='crash_1',
            clearable=False,
            className="incident-dropdown"
        ),
        html.Div(id='incident-summary', className="incident-summary")
    ], className="card selector-card")

# History panel
def create_history_panel():
    return html.Div([
        html.H3("Recent Incidents", className="card-title"),
        html.Div(id='incident-history', className="history-list")
    ], className="card history-card")

# Main layout
def create_layout():
    return html.Div([
        get_custom_css(),
        create_header(),
        
        # Main content row
        html.Div([
            # Left column - Analyze Media + Incident selector
            html.Div([
                create_analyze_media_panel(),
                create_incident_selector()
            ], className="col-left"),
            
            # Center column - Main incident details
            html.Div([
                html.Div([
                    html.Div(id='severity-card-container'),
                    html.Div(id='resources-card-container')
                ], className="row-top"),
                html.Div([
                    html.Div(id='vms-card-container'),
                    create_incident_details_card()
                ], className="row-bottom")
            ], className="col-center"),
            
            # Right column - Risk and history
            html.Div([
                html.Div(id='risk-card-container'),
                create_history_panel()
            ], className="col-right")
        ], className="main-row"),
        
        # Store components for data
        dcc.Store(id='incident-data-store'),
        dcc.Interval(
            id='refresh-interval',
            interval=30000,  # 30 seconds
            n_intervals=0
        )
    ], className="app-container")