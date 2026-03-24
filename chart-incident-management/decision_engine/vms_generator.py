# decision_engine/vms_generator.py
"""
Variable Message Sign Generator
Creates message boards for drivers based on incident
"""

def generate_vms_messages(incident_data):
    """
    Generate VMS messages for driver notification
    
    Args:
        incident_data: Dictionary from model output
    
    Returns:
        dict: {
            'primary': str,    # Top line (most important)
            'secondary': str,  # Middle line
            'tertiary': str,   # Bottom line
            'all_messages': [str],  # All lines as list
            'template_used': str
        }
    """
    
    incident_type = incident_data.get('incident_type', 'unknown')
    lanes = incident_data.get('lanes', {})
    hazards = incident_data.get('hazards', {})
    traffic = incident_data.get('traffic', {})
    
    blocked = lanes.get('blocked', [])
    shoulder = lanes.get('shoulder_blocked', False)
    
    # ===== TEMPLATES BY INCIDENT TYPE =====
    
    templates = {
        'crash': {
            'primary': "ACCIDENT AHEAD",
            'secondary': _format_lanes_blocked(blocked, shoulder),
            'tertiary': _format_traffic_advice(traffic)
        },
        'fire': {
            'primary': "FIRE AHEAD",
            'secondary': "EMERGENCY VEHICLES",
            'tertiary': "KEEP BACK 500 FT"
        },
        'debris': {
            'primary': "DEBRIS IN ROAD",
            'secondary': _format_lanes_blocked(blocked, shoulder),
            'tertiary': "USE CAUTION"
        },
        'breakdown': {
            'primary': "STOPPED VEHICLE",
            'secondary': _format_lanes_blocked(blocked, shoulder),
            'tertiary': "MERGE SAFELY"
        },
        'none': {
            'primary': "NORMAL TRAFFIC",
            'secondary': "NO INCIDENTS",
            'tertiary': "DRIVE SAFELY"
        }
    }
    
    # Get base template
    template = templates.get(incident_type, templates['none']).copy()
    
    # Override for special conditions
    if hazards.get('smoke', False):
        template['secondary'] = "SMOKE REDUCES VISIBILITY"
        template['tertiary'] = "REDUCE SPEED"
    
    if hazards.get('spill', False):
        template['secondary'] = "HAZMAT SPILL"
        template['tertiary'] = "FOLLOW DETOUR"
    
    # Format lane information
    if 'LANE' in template['secondary'] and not template['secondary']:
        # If lane formatting returned empty, use generic
        template['secondary'] = "USE ALTERNATE LANES"
    
    return {
        'primary': template['primary'],
        'secondary': template['secondary'],
        'tertiary': template['tertiary'],
        'all_messages': [template['primary'], template['secondary'], template['tertiary']],
        'template_used': incident_type
    }


def _format_lanes_blocked(blocked_lanes, shoulder_blocked):
    """Helper to format lane blockage text"""
    if not blocked_lanes and not shoulder_blocked:
        return "ALL LANES OPEN"
    
    if len(blocked_lanes) == 1:
        lane_num = blocked_lanes[0]
        
        return f"LANE {lane_num} CLOSED"
    
    if len(blocked_lanes) >= 2:
        lanes_str = ", ".join(str(l) for l in blocked_lanes)
        return f"LANES {lanes_str} CLOSED"
    
    if shoulder_blocked:
        return "SHOULDER CLOSED"
    
    return ""


def _format_traffic_advice(traffic_data):
    """Helper to format traffic advice"""
    state = traffic_data.get('state', 'unknown')
    queue = traffic_data.get('queue_length_meters', 0)
    
    if state == 'stopped':
        return "EXPECT STANDSTILL"
    elif state == 'slowing':
        if queue > 200:
            return f"LONG DELAYS AHEAD"
        else:
            return "EXPECT DELAYS"
    else:
        return "MERGE LEFT"


def test_vms_generator():
    """Test the VMS generator"""
    
    test_cases = [
        {
            'name': 'Multi-lane crash',
            'data': {
                'incident_type': 'crash',
                'lanes': {'blocked': [1, 2], 'shoulder_blocked': False},
                'hazards': {'smoke': False, 'spill': False},
                'traffic': {'state': 'slowing', 'queue_length_meters': 300}
            }
        },
        {
            'name': 'Single lane crash',
            'data': {
                'incident_type': 'crash',
                'lanes': {'blocked': [2], 'shoulder_blocked': False},
                'hazards': {'smoke': False, 'spill': False},
                'traffic': {'state': 'slowing', 'queue_length_meters': 100}
            }
        },
        {
            'name': 'Fire with smoke',
            'data': {
                'incident_type': 'fire',
                'lanes': {'blocked': [], 'shoulder_blocked': True},
                'hazards': {'smoke': True, 'spill': False},
                'traffic': {'state': 'stopped', 'queue_length_meters': 500}
            }
        },
        {
            'name': 'Debris only',
            'data': {
                'incident_type': 'debris',
                'lanes': {'blocked': [3], 'shoulder_blocked': False},
                'hazards': {'smoke': False, 'spill': False},
                'traffic': {'state': 'flowing', 'queue_length_meters': 0}
            }
        },
        {
            'name': 'Hazmat spill',
            'data': {
                'incident_type': 'crash',
                'lanes': {'blocked': [1], 'shoulder_blocked': False},
                'hazards': {'smoke': False, 'spill': True},
                'traffic': {'state': 'slowing', 'queue_length_meters': 200}
            }
        }
    ]
    
    print("="*70)
    print("VMS GENERATOR TESTS")
    print("="*70)
    
    for test in test_cases:
        print(f"\n📋 {test['name']}")
        print("-" * 50)
        
        result = generate_vms_messages(test['data'])
        
        print(f"  ╔══════════════════════════════╗")
        print(f"  ║  {result['primary']:<26} ║")
        print(f"  ║  {result['secondary']:<26} ║")
        print(f"  ║  {result['tertiary']:<26} ║")
        print(f"  ╚══════════════════════════════╝")
        
        print(f"  Template: {result['template_used']}")

if __name__ == "__main__":
    test_vms_generator()