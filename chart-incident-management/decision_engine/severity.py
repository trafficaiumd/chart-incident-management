# decision_engine/severity.py
"""
CHART Severity Scoring Engine
RED / YELLOW / GREEN based on incident characteristics
"""

def calculate_severity(incident_data):
    """
    Determine severity level based on CHART rules
    
    Args:
        incident_data: Dictionary from model output following complex schema
    
    Returns:
        dict: {
            'level': 'RED'|'YELLOW'|'GREEN',
            'reasons': [list of rules that triggered],
            'score': int (0-100)
        }
    """
    reasons = []
    score = 0
    
    # Extract data (with safe defaults)
    vehicles = incident_data.get('vehicles', {})
    hazards = incident_data.get('hazards', {})
    lanes = incident_data.get('lanes', {})
    incident_type = incident_data.get('incident_type', 'unknown')
    
    # ===== RED SEVERITY RULES =====
    
    # Fire is always RED
    if hazards.get('fire', False):
        reasons.append("RED: Fire present")
        score += 100
    
    # School bus involved is RED
    nearby = vehicles.get('nearby_traffic', [])
    involved = vehicles.get('involved_in_incident', [])
    all_vehicles = nearby + involved
    
    for v in all_vehicles:
        if v.get('type') == 'school_bus':
            reasons.append("RED: School bus involved")
            score += 100
            break
    
    # 3+ vehicles involved is RED
    if vehicles.get('total_count', 0) >= 3:
        reasons.append("RED: 3+ vehicles involved")
        score += 90
    
    # 2+ lanes blocked is RED
    blocked_lanes = lanes.get('blocked', [])
    if len(blocked_lanes) >= 2:
        reasons.append(f"RED: {len(blocked_lanes)} lanes blocked")
        score += 80
    
    # ===== YELLOW SEVERITY RULES =====
    
    # Debris is at least YELLOW
    if hazards.get('debris', False) and score < 70:
        reasons.append("YELLOW: Debris in roadway")
        score = max(score, 60)
    
    # Single lane blocked is YELLOW
    if len(blocked_lanes) == 1 and score < 60:
        reasons.append(f"YELLOW: Lane {blocked_lanes[0]} blocked")
        score = max(score, 50)
    
    # 2 vehicles is YELLOW
    if vehicles.get('total_count', 0) == 2 and score < 50:
        reasons.append("YELLOW: Two vehicles involved")
        score = max(score, 40)
    
    # ===== GREEN =====
    # No issues or minor issues only
    
    if score == 0:
        if incident_type == 'none':
            reasons.append("GREEN: No incident detected")
        else:
            reasons.append("GREEN: Minor incident, no significant hazards")
        score = 10
    
    # Determine final level
    if score >= 80:
        level = 'RED'
    elif score >= 40:
        level = 'YELLOW'
    else:
        level = 'GREEN'
    
    return {
        'level': level,
        'reasons': reasons,
        'score': score
    }


def test_severity():
    """Quick test function"""
    # Test with crash data
    crash_data = {
        'incident_type': 'crash',
        'vehicles': {'total_count': 4},
        'hazards': {'fire': False, 'debris': True},
        'lanes': {'blocked': [1]}
    }
    
    result = calculate_severity(crash_data)
    print(f"Crash severity: {result['level']}")
    for reason in result['reasons']:
        print(f"  - {reason}")
    
    # Test with fire
    fire_data = {
        'incident_type': 'fire',
        'vehicles': {'total_count': 1},
        'hazards': {'fire': True, 'debris': False},
        'lanes': {'blocked': []}
    }
    
    result = calculate_severity(fire_data)
    print(f"\nFire severity: {result['level']}")
    for reason in result['reasons']:
        print(f"  - {reason}")

if __name__ == "__main__":
    test_severity()