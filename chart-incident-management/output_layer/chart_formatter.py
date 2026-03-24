# output_layer/chart_formatter.py
"""
CHART System Formatter
Formats decision output to match CHART's expected schema
"""

import json
from datetime import datetime
from typing import Dict, Any, List


class CHARTFormatter:
    """
    Formats internal decision package to CHART's external schema
    """
    
    def __init__(self):
        self.version = "1.0.0"
        self.agency = "CHART"
    
    def format(self, decision_package: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert internal decision package to CHART format
        
        Args:
            decision_package: Output from DecisionPipeline.process()
        
        Returns:
            dict: CHART-compatible incident report
        """
        
        # Extract data
        incident = decision_package.get('incident', {})
        severity = decision_package.get('severity', {})
        resources = decision_package.get('resources', {})
        vms = decision_package.get('vms', {})
        risk = decision_package.get('secondary_risk', {})
        context = decision_package.get('context', {})
        
        # Generate CHART incident ID
        incident_id = self._generate_incident_id()
        
        # Build CHART-compatible output
        chart_output = {
            "incident_id": incident_id,
            "timestamp": decision_package.get('timestamp', datetime.now().isoformat()),
            "agency": self.agency,
            "status": "active",
            "location": self._format_location(context),
            "incident": {
                "type": incident.get('type', 'unknown'),
                "confidence": incident.get('confidence', 0.0),
                "description": self._generate_description(decision_package)
            },
            "severity": {
                "level": severity.get('level', 'GREEN'),
                "score": severity.get('score', 0),
                "reasons": severity.get('reasons', [])
            },
            "response": {
                "dispatch": self._format_dispatch(resources),
                "lane_closures": self._format_lane_closures(decision_package),
                "vms_messages": vms.get('all_messages', []),
                "estimated_duration": self._estimate_duration(decision_package)
            },
            "secondary_risk": {
                "level": risk.get('risk_level', 'LOW'),
                "score": risk.get('risk_score', 0),
                "factors": risk.get('factors', [])[:3]  # Top 3 factors
            },
            "requires_human_review": decision_package.get('requires_human_review', False),
            "review_status": "pending" if decision_package.get('requires_human_review') else "auto_approved",
            "version": self.version
        }
        
        return chart_output
    
    def _generate_incident_id(self) -> str:
        """Generate unique incident ID in CHART format"""
        date_str = datetime.now().strftime("%Y%m%d")
        sequence = datetime.now().strftime("%H%M%S")
        return f"CHART-{date_str}-{sequence}"
    
    def _format_location(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Format location data for CHART"""
        return {
            "roadway": context.get('location', 'unknown').split(' ')[0] if ' ' in context.get('location', '') else context.get('location', 'unknown'),
            "direction": self._extract_direction(context.get('location', '')),
            "mile_marker": self._extract_mile_marker(context.get('location', '')),
            "description": context.get('location', 'unknown')
        }
    
    def _extract_direction(self, location: str) -> str:
        """Extract direction from location string (e.g., 'I-95 NB' -> 'NB')"""
        parts = location.split()
        for part in parts:
            if part in ['NB', 'SB', 'EB', 'WB']:
                return part
        return 'unknown'
    
    def _extract_mile_marker(self, location: str) -> str:
        """Extract mile marker from location string"""
        import re
        match = re.search(r'mile\s*(\d+(?:\.\d+)?)', location.lower())
        return match.group(1) if match else 'unknown'
    
    def _generate_description(self, package: Dict[str, Any]) -> str:
        """Generate human-readable incident description"""
        incident_type = package.get('incident', {}).get('type', 'unknown')
        severity = package.get('severity', {}).get('level', 'UNKNOWN')
        lanes = package.get('vms', {}).get('secondary', '')
        
        desc = f"{severity} severity {incident_type}"
        if lanes and 'LANE' in lanes:
            desc += f" with {lanes.lower()}"
        
        return desc
    
    def _format_dispatch(self, resources: Dict[str, Any]) -> List[str]:
        """Format dispatch list for CHART"""
        dispatch = []
        for resource, needed in resources.items():
            if resource != 'reasons' and needed:
                # Convert internal names to CHART standard
                chart_name = {
                    'police': 'Law Enforcement',
                    'ems': 'EMS',
                    'fire': 'Fire/Rescue',
                    'tow': 'Tow Truck',
                    'dot': 'DOT Maintenance',
                    'hazmat': 'Hazmat Team'
                }.get(resource, resource.upper())
                dispatch.append(chart_name)
        return dispatch
    
    def _format_lane_closures(self, package: Dict[str, Any]) -> Dict[str, Any]:
        """Format lane closure information"""
        # Extract from VMS or incident data
        vms_secondary = package.get('vms', {}).get('secondary', '')
        
        closed_lanes = []
        if 'LANE' in vms_secondary:
            # Parse "LANE 1 CLOSED" or "LANES 1, 2 CLOSED"
            import re
            numbers = re.findall(r'\d+', vms_secondary)
            closed_lanes = [int(n) for n in numbers]
        
        return {
            "closed_lanes": closed_lanes,
            "total_lanes": 4,  # Default, could be extracted
            "shoulder_closed": 'SHOULDER' in vms_secondary,
            "taper_length_ft": self._calculate_taper_length(closed_lanes)
        }
    
    def _calculate_taper_length(self, closed_lanes: List[int]) -> int:
        """Calculate taper length in feet (speed limit * lanes * 10)"""
        speed_limit = 55  # Default highway speed
        lanes_affected = len(closed_lanes) if closed_lanes else 0
        return speed_limit * max(lanes_affected, 1) * 10
    
    def _estimate_duration(self, package: Dict[str, Any]) -> str:
        """Estimate incident duration based on severity"""
        severity = package.get('severity', {}).get('level', 'GREEN')
        
        duration_map = {
            'RED': '2-4 hours',
            'YELLOW': '1-2 hours',
            'GREEN': '30-60 minutes'
        }
        
        return duration_map.get(severity, 'unknown')


def test_chart_formatter():
    """Test the CHART formatter with sample data"""
    
    from decision_engine.pipeline import DecisionPipeline
    
    # Create sample decision
    pipeline = DecisionPipeline()
    
    crash_output = {
        'incident_type': 'crash',
        'confidence': 0.95,
        'vehicles': {
            'total_count': 4,
            'involved_in_incident': [{'type': 'pickup_truck', 'damaged': True}],
            'nearby_traffic': [{'type': 'bus', 'damaged': False}]
        },
        'lanes': {'total': 4, 'blocked': [1, 2], 'shoulder_blocked': False},
        'hazards': {'fire': False, 'smoke': False, 'debris': True, 'injuries_visible': False},
        'traffic': {'state': 'stopped', 'queue_length_meters': 800},
        'emergency_response': {'police_present': False}
    }
    
    context = {
        'time_of_day': 'night',
        'weather': 'clear',
        'location': 'I-95 NB mile 27.4'
    }
    
    decision = pipeline.process(crash_output, context)
    
    # Format for CHART
    formatter = CHARTFormatter()
    chart_output = formatter.format(decision)
    
    print("="*70)
    print("CHART FORMATTED OUTPUT")
    print("="*70)
    print(json.dumps(chart_output, indent=2))
    
    return chart_output

if __name__ == "__main__":
    test_chart_formatter()