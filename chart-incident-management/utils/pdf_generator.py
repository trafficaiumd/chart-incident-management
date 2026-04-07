"""
PDF generator for incident reports.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _to_pretty_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, indent=2)
    return str(value)


def generate_incident_pdf(incident_data: Dict[str, Any], output_path: str | None = None) -> str:
    """
    Build a one-page CHART-style PDF report for a single incident.
    """
    if output_path is None:
        fd, tmp_path = tempfile.mkstemp(prefix="incident_report_", suffix=".pdf")
        Path(tmp_path).unlink(missing_ok=True)
        output_path = tmp_path
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output), pagesize=letter)
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "ChartHeader",
        parent=styles["Heading1"],
        fontSize=18,
        textColor=colors.HexColor("#0B3D91"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "ChartSubtitle",
        parent=styles["Normal"],
        fontSize=10,
        textColor=colors.HexColor("#333333"),
        spaceAfter=12,
    )

    elements = []
    elements.append(Paragraph("CHART Official Incident Report", title_style))
    elements.append(Paragraph("Coordinated Highways Action Response Team", subtitle_style))

    montage_path = incident_data.get("montage_path")
    if montage_path:
        p = Path(str(montage_path))
        if p.exists() and p.is_file():
            elements.append(Paragraph("2x3 Evidence Montage", styles["Heading3"]))
            elements.append(Image(str(p), width=6.8 * inch, height=3.4 * inch))
            elements.append(Spacer(1, 0.18 * inch))

    # Clean tabular summary from JSON verdict fields.
    ordered_keys = [
        "incident_type",
        "timestamp",
        "collision_confirmed",
        "severity",
        "lanes_blocked",
        "hazards",
        "vehicle_types",
        "verdict",
    ]
    rows = [["Field", "Value"]]
    for key in ordered_keys:
        if key in incident_data:
            rows.append([key, _to_pretty_value(incident_data.get(key))])

    # Include any extra keys not in default order.
    for key in sorted(incident_data.keys()):
        if key not in ordered_keys:
            rows.append([key, _to_pretty_value(incident_data.get(key))])

    table = Table(rows, colWidths=[1.9 * inch, 4.8 * inch], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0B3D91")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.whitesmoke),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ALIGN", (0, 0), (-1, -1), "LEFT"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.HexColor("#F7F9FC")]),
            ]
        )
    )
    elements.append(Paragraph("Incident Summary", styles["Heading3"]))
    elements.append(table)

    doc.build(elements)
    return str(output)

