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
from reportlab.platypus import Flowable, Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def _to_pretty_value(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return _safe_truncate(json.dumps(value, indent=2))
    return _safe_truncate(str(value))


def _safe_truncate(text: Any, max_chars: int = 400) -> str:
    s = str(text or "")
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 3] + "..."


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


class SeverityBar(Flowable):
    def __init__(self, score: float, width: float = 300.0, height: float = 14.0):
        super().__init__()
        self.score = max(0.0, min(100.0, float(score or 0.0)))
        self.width = width
        self.height = height

    def draw(self):
        self.canv.setStrokeColor(colors.black)
        self.canv.rect(0, 0, self.width, self.height, stroke=1, fill=0)
        if self.score >= 75:
            fill = colors.red
        elif self.score >= 40:
            fill = colors.orange
        else:
            fill = colors.green
        fill_width = self.width * (self.score / 100.0)
        self.canv.setFillColor(fill)
        self.canv.rect(0, 0, fill_width, self.height, stroke=0, fill=1)


def _make_section_table(title: str, pairs: list[tuple[str, Any]], styles) -> list[Any]:
    if not pairs:
        return []
    rows = [["Field", "Value"]]
    for k, v in pairs:
        rows.append([k, _to_pretty_value(v)])
    table = Table(rows, colWidths=[2.0 * inch, 4.7 * inch], repeatRows=1)
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
    return [Paragraph(title, styles["Heading3"]), table, Spacer(1, 0.12 * inch)]


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

    try:
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
        elements.append(Paragraph("Maryland DOT / CHART Official Incident Report", title_style))
        elements.append(Paragraph("Coordinated Highways Action Response Team - Forensic Traffic AI", subtitle_style))

        montage_path = incident_data.get("montage_path")
        if montage_path:
            p = Path(str(montage_path))
            if p.exists() and p.is_file():
                elements.append(Paragraph("2x3 Evidence Montage", styles["Heading3"]))
                elements.append(Image(str(p), width=6.8 * inch, height=3.4 * inch))
                elements.append(Spacer(1, 0.18 * inch))
        hero_path = incident_data.get("hero_frame_path")
        if hero_path:
            p = Path(str(hero_path))
            if p.exists() and p.is_file():
                elements.append(Paragraph("Hero Frame (Peak Impact)", styles["Heading3"]))
                elements.append(Image(str(p), width=6.8 * inch, height=3.2 * inch))
                elements.append(Spacer(1, 0.18 * inch))

        forensic = incident_data.get("forensic_report", {}) if isinstance(incident_data.get("forensic_report", {}), dict) else {}
        cam = forensic.get("camera", {}) if isinstance(forensic.get("camera", {}), dict) else {}
        veh = forensic.get("vehicles", {}) if isinstance(forensic.get("vehicles", {}), dict) else {}
        hazards_obj = forensic.get("hazards", {}) if isinstance(forensic.get("hazards", {}), dict) else {}
        lane = forensic.get("lane_impact", {}) if isinstance(forensic.get("lane_impact", {}), dict) else {}
        sev = forensic.get("severity_info", {}) if isinstance(forensic.get("severity_info", {}), dict) else {}
        sev_inputs = sev.get("severity_inputs", {}) if isinstance(sev.get("severity_inputs", {}), dict) else {}
        derived = incident_data.get("derived_by_python", {}) if isinstance(incident_data.get("derived_by_python", {}), dict) else {}
        score = _safe_float(derived.get("severity_score_0_to_100", incident_data.get("severity_score_0_to_100", 0.0)), 0.0)

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
        elements.append(Spacer(1, 0.16 * inch))

        elements.append(Paragraph("Severity Assessment", styles["Heading3"]))
        elements.append(SeverityBar(score))
        elements.append(Spacer(1, 0.06 * inch))
        elements.append(
            Paragraph(
                f"Severity Score: {score:.1f}/100 | Category: {_to_pretty_value(derived.get('severity_category', incident_data.get('severity_category', 'unknown')))}",
                styles["Normal"],
            )
        )
        elements.append(Spacer(1, 0.14 * inch))

        elements += _make_section_table(
            "Camera Metadata",
            [
                ("Camera ID", cam.get("id", "unknown")),
                ("Name", cam.get("name", "unknown")),
                ("Milepost", cam.get("milePost", "unknown")),
                ("Route Number", cam.get("routeNumber", "unknown")),
                ("Latitude", cam.get("lat", "unknown")),
                ("Longitude", cam.get("lon", "unknown")),
                ("Direction", cam.get("direction", "unknown")),
            ],
            styles,
        )
        elements += _make_section_table(
            "Vehicle Summary",
            [
                ("Count Involved", veh.get("count_involved", 0)),
                ("Vehicle Types", incident_data.get("vehicle_types", [])),
                ("Vehicle List", veh.get("list", [])),
            ],
            styles,
        )
        elements += _make_section_table(
            "Lane Impact",
            [
                ("Lanes Blocked", lane.get("lanes_blocked", incident_data.get("lanes_blocked", 0))),
                ("Blocked Lanes Description", lane.get("blocked_lanes_description", "unknown")),
            ],
            styles,
        )
        elements += _make_section_table(
            "Hazards",
            [
                ("Hazard Tokens", incident_data.get("hazards", [])),
                ("Fire Visible", hazards_obj.get("fire_visible", "unknown")),
                ("Smoke Visible", hazards_obj.get("smoke_visible", "unknown")),
                ("Debris Visible", hazards_obj.get("debris_visible", "unknown")),
                ("Hazard Confidence", hazards_obj.get("confidence_hazards", 0.0)),
            ],
            styles,
        )
        elements += _make_section_table(
            "Severity Inputs",
            [
                ("Vehicle Damage", sev_inputs.get("vehicle_damage", {})),
                ("Lane Blockage", sev_inputs.get("lane_blockage", {})),
                ("Injury Visibility", sev_inputs.get("injury_visibility", {})),
                ("Fire/Smoke Hazard", sev_inputs.get("fire_smoke_hazard", {})),
                ("Severe Gating Triggers", sev.get("severe_gating_triggers", {})),
            ],
            styles,
        )

        elements.append(Paragraph("Gemma Reasoning", styles["Heading3"]))
        reasoning = (
            incident_data.get("forensic_report", {})
            .get("incident", {})
            .get("why", incident_data.get("verdict", "unknown"))
        )
        elements.append(Paragraph(_to_pretty_value(reasoning), styles["Normal"]))

        doc.build(elements)
        return str(output)
    except Exception as e:
        print(f"[REPORT] warning: generate_incident_pdf failed: {e}", flush=True)
        return ""


def generate_batch_summary_pdf(incidents: list[Dict[str, Any]], output_path: str | None = None) -> str:
    """
    Build a compact summary PDF for a multi-video batch run.
    """
    if output_path is None:
        fd, tmp_path = tempfile.mkstemp(prefix="batch_incident_summary_", suffix=".pdf")
        Path(tmp_path).unlink(missing_ok=True)
        output_path = tmp_path
    output = Path(output_path).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    doc = SimpleDocTemplate(str(output), pagesize=letter)
    styles = getSampleStyleSheet()
    elements = [
        Paragraph("CHART Batch Incident Summary Report", styles["Title"]),
        Spacer(1, 0.12 * inch),
        Paragraph("Detected incidents across video_dir processing run.", styles["Normal"]),
        Spacer(1, 0.14 * inch),
    ]

    rows = [["#", "Timestamp", "Incident Type", "Severity", "Score (0-100)", "Confidence"]]
    for i, inc in enumerate(incidents or [], start=1):
        rows.append(
            [
                str(i),
                _to_pretty_value(inc.get("timestamp", "unknown")),
                _to_pretty_value(inc.get("incident_type", "unknown")),
                _to_pretty_value(inc.get("severity_category", inc.get("severity", "unknown"))),
                _to_pretty_value(inc.get("severity_score_0_to_100", "unknown")),
                _to_pretty_value(inc.get("confidence", "unknown")),
            ]
        )

    if len(rows) == 1:
        rows.append(["-", "unknown", "none_detected", "unknown", "0", "0.0"])

    table = Table(rows, colWidths=[0.4 * inch, 1.5 * inch, 1.4 * inch, 1.2 * inch, 1.1 * inch, 1.0 * inch], repeatRows=1)
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
    elements.append(table)
    doc.build(elements)
    return str(output)

