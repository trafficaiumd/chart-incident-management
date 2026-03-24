# dashboard/app.py
"""
Main Dash application for CHART Traffic Incident Management
"""

import dash
from dash import dcc, html
import plotly.io as pio
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import your layout and callbacks
from layout import create_layout
from callbacks import register_callbacks

# Initialize the Dash app - ADD suppress_callback_exceptions=True
app = dash.Dash(
    __name__,
    title="CHART Traffic Incident Management",
    update_title="Loading...",
    suppress_callback_exceptions=True,  # 👈 ADD THIS LINE
    external_stylesheets=[
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap'
    ]
)

# Set the app layout
app.layout = create_layout()

# Register all callbacks
register_callbacks(app)

# For deployment
server = app.server

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8050)