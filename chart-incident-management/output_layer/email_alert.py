# output_layer/email_alert.py
#this is optional and can be left as a placeholder 
"""
Email alerts for critical incidents
"""

import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import List, Dict, Any
import json

class EmailAlert:
    """
    Send email alerts for HIGH severity incidents
    """
    
    def __init__(self, smtp_server: str = "localhost", port: int = 25):
        self.smtp_server = smtp_server
        self.port = port
        self.recipients = []
    
    def add_recipient(self, email: str):
        """Add recipient to alert list"""
        self.recipients.append(email)
    
    def send_alert(self, incident: Dict[str, Any], chart_data: Dict[str, Any]):
        """Send alert email for critical incident"""
        
        if not self.recipients:
            print("No recipients configured")
            return
        
        # Only send for RED severity or HIGH secondary risk
        severity = incident.get('severity', {}).get('level')
        risk = incident.get('secondary_risk', {}).get('risk_level')
        
        if severity != 'RED' and risk != 'HIGH':
            return
        
        # Create message
        msg = MIMEMultipart()
        msg['Subject'] = f"🚨 CHART ALERT: {incident['incident']['type'].upper()} - {severity} Severity"
        msg['From'] = "chart-alerts@mdot.state"
        msg['To'] = ", ".join(self.recipients)
        
        # Build email body
        body = f"""
CHART INCIDENT ALERT
====================
Time: {incident['timestamp']}
Location: {incident.get('context', {}).get('location', 'unknown')}

INCIDENT: {incident['incident']['type']}
Severity: {severity} ({incident['severity']['score']})
Secondary Risk: {risk} ({incident['secondary_risk']['score']})

RESOURCES DISPATCHED:
{self._format_resources(incident['resources'])}

VMS MESSAGES:
{chr(10).join(incident['vms']['all_messages'])}

Requires Human Review: {incident['requires_human_review']}

Full Incident Data:
{json.dumps(chart_data, indent=2)}
        """
        
        msg.attach(MIMEText(body, 'plain'))
        
        # Send
        try:
            with smtplib.SMTP(self.smtp_server, self.port) as server:
                server.send_message(msg)
            print(f"Alert sent to {len(self.recipients)} recipients")
        except Exception as e:
            print(f"Failed to send alert: {e}")
    
    def _format_resources(self, resources: Dict) -> str:
        """Format resources for email"""
        lines = []
        for resource, needed in resources.items():
            if resource != 'reasons' and needed:
                lines.append(f"  - {resource.upper()}")
        return "\n".join(lines) if lines else "  None"


def test_email():
    """Test email alert"""
    from decision_engine.pipeline import DecisionPipeline
    
    pipeline = DecisionPipeline()
    
    crash_output = {
        'incident_type': 'crash',
        'confidence': 0.95,
        'vehicles': {'total_count': 4},
        'lanes': {'blocked': [1, 2]},
        'hazards': {'debris': True},
        'traffic': {'state': 'stopped'},
        'emergency_response': {}
    }
    
    decision = pipeline.process(crash_output)
    
    alert = EmailAlert()
    alert.add_recipient("operator@chart.gov")
    alert.send_alert(decision, decision)  # Simplified for test

if __name__ == "__main__":
    test_email()