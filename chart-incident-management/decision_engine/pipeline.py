# decision_engine/pipeline.py
"""
Complete Decision Pipeline
Integrates all decision engines into one flow
"""
import sys
from pathlib import Path

# Add parent directory to path so we can import sibling modules
sys.path.append(str(Path(__file__).parent.parent))

from decision_engine.severity import calculate_severity
from decision_engine.resources import recommend_resources
from decision_engine.vms_generator import generate_vms_messages
from decision_engine.secondary_risk import analyze_secondary_risk

class DecisionPipeline:
    """
    End-to-end decision engine for CHART
    Takes model output, returns complete recommendations
    """
    
    def __init__(self):
        self.version = "1.0.0"
        
    def process(self, model_output, context=None):
        """
        Process model output through all decision engines
        
        Args:
            model_output: Dictionary from AI model (complex schema)
            context: Optional dict with time/weather/location
        
        Returns:
            dict: Complete decision package for CHART operator
        """
        if context is None:
            context = {
                'time_of_day': 'day',
                'weather': 'clear',
                'location': 'unknown'
            }
        
        # Run all engines
        severity = calculate_severity(model_output)
        resources = recommend_resources(model_output)
        vms = generate_vms_messages(model_output)
        risk = analyze_secondary_risk(
            model_output,
            context.get('time_of_day', 'day'),
            context.get('weather', 'clear')
        )
        
        # Compile complete response
        decision = {
            'timestamp': self._get_timestamp(),
            'incident': {
                'type': model_output.get('incident_type', 'unknown'),
                'confidence': model_output.get('confidence', 0.5)
            },
            'severity': severity,
            'resources': resources,
            'vms': vms,
            'secondary_risk': risk,
            'context': context,
            'requires_human_review': self._needs_review(severity, risk)
        }
        
        return decision
    
    def _get_timestamp(self):
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _needs_review(self, severity, risk):
        """Determine if human review is needed"""
        # Always review HIGH severity or HIGH secondary risk
        if severity['level'] == 'RED' or risk['risk_level'] == 'HIGH':
            return True
        
        # Review if conflicting signals
        if severity['level'] == 'YELLOW' and risk['risk_level'] == 'HIGH':
            return True
        
        return False
    
    def format_for_display(self, decision):
        """Format decision for console/UI display"""
        lines = []
        lines.append("="*60)
        lines.append("CHART INCIDENT DECISION PACKAGE")
        lines.append("="*60)
        lines.append(f"Time: {decision['timestamp']}")
        lines.append(f"Incident: {decision['incident']['type']} (confidence: {decision['incident']['confidence']:.0%})")
        lines.append(f"Severity: {decision['severity']['level']} ({decision['severity']['score']})")
        
        lines.append("\n🚨 RESOURCES TO DISPATCH:")
        for resource, needed in decision['resources'].items():
            if resource != 'reasons' and needed:
                lines.append(f"  • {resource.upper()}")
        
        lines.append("\n📋 REASONS:")
        for reason in decision['resources'].get('reasons', []):
            lines.append(f"  • {reason}")
        
        lines.append("\n💬 VMS MESSAGES:")
        lines.append(f"  ╔══════════════════════════════╗")
        lines.append(f"  ║  {decision['vms']['primary']:<26} ║")
        lines.append(f"  ║  {decision['vms']['secondary']:<26} ║")
        lines.append(f"  ║  {decision['vms']['tertiary']:<26} ║")
        lines.append(f"  ╚══════════════════════════════╝")
        
        lines.append(f"\n⚠️ SECONDARY RISK: {decision['secondary_risk']['risk_level']} ({decision['secondary_risk']['risk_score']})")
        lines.append("  Factors:")
        for factor in decision['secondary_risk']['factors'][:3]:  # Top 3 factors
            lines.append(f"    • {factor}")
        
        if decision['requires_human_review']:
            lines.append("\n👤 REQUIRES HUMAN REVIEW")
        
        return "\n".join(lines)


def test_pipeline():
    """Test the complete pipeline with sample data"""
    
    pipeline = DecisionPipeline()
    
    # Test case 1: Night crash
    crash_output = {
        'incident_type': 'crash',
        'confidence': 0.95,
        'vehicles': {
            'total_count': 4,
            'involved_in_incident': [
                {'type': 'pickup_truck', 'damaged': True}
            ],
            'nearby_traffic': [
                {'type': 'bus', 'damaged': False},
                {'type': 'pickup', 'damaged': False}
            ]
        },
        'lanes': {
            'total': 4,
            'blocked': [1, 2],
            'shoulder_blocked': False
        },
        'hazards': {
            'fire': False,
            'smoke': False,
            'debris': True,
            'injuries_visible': False,
            'spill': False
        },
        'traffic': {
            'state': 'stopped',
            'queue_length_meters': 800
        },
        'emergency_response': {
            'police_present': False,
            'officers_count': 0
        }
    }
    
    context = {
        'time_of_day': 'night',
        'weather': 'clear',
        'location': 'I-95 NB mile 27.4'
    }
    
    print("\n" + "="*60)
    print("TEST 1: Night Crash Scenario")
    print("="*60)
    
    decision = pipeline.process(crash_output, context)
    print(pipeline.format_for_display(decision))
    
    # Test case 2: Day debris
    debris_output = {
        'incident_type': 'debris',
        'confidence': 0.85,
        'vehicles': {
            'total_count': 0,
            'involved_in_incident': [],
            'nearby_traffic': []
        },
        'lanes': {
            'total': 4,
            'blocked': [3],
            'shoulder_blocked': False
        },
        'hazards': {
            'fire': False,
            'smoke': False,
            'debris': True,
            'injuries_visible': False,
            'spill': False
        },
        'traffic': {
            'state': 'slowing',
            'queue_length_meters': 200
        },
        'emergency_response': {
            'police_present': False
        }
    }
    
    context = {
        'time_of_day': 'day',
        'weather': 'rain',
        'location': 'I-695 SB mm 12.1'
    }
    
    print("\n" + "="*60)
    print("TEST 2: Day Debris in Rain")
    print("="*60)
    
    decision = pipeline.process(debris_output, context)
    print(pipeline.format_for_display(decision))

if __name__ == "__main__":
    test_pipeline()