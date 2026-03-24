# decision_engine/resources.py
"""
Resource Recommendation Engine
Determines which emergency services to dispatch
"""

def recommend_resources(incident_data):
    """
    Recommend resources based on incident characteristics
    
    Args:
        incident_data: Dictionary from model output
    
    Returns:
        dict: {
            'police': bool,
            'ems': bool,
            'fire': bool,
            'tow': bool,
            'dot': bool,
            'hazmat': bool,
            'reasons': [str]
        }
    """
    resources = {
        'police': False,
        'ems': False,
        'fire': False,
        'tow': False,
        'dot': False,
        'hazmat': False,
        'reasons': []
    }
    
    # Extract data
    incident_type = incident_data.get('incident_type', 'unknown')
    vehicles = incident_data.get('vehicles', {})
    hazards = incident_data.get('hazards', {})
    lanes = incident_data.get('lanes', {})
    emergency = incident_data.get('emergency_response', {})
    
    # ===== POLICE =====
    # Police for any incident (except if already on scene)
    if incident_type != 'none':
        if not emergency.get('police_present', False):
            resources['police'] = True
            resources['reasons'].append("Police: Incident reported")
        else:
            resources['reasons'].append("Police: Already on scene")
    
    # ===== EMS (Ambulance) =====
    # EMS for injuries, fire, or severe crashes
    if hazards.get('injuries_visible', False):
        resources['ems'] = True
        resources['reasons'].append("EMS: Visible injuries")
    
    if hazards.get('fire', False):
        resources['ems'] = True
        resources['reasons'].append("EMS: Fire present")
    
    if vehicles.get('total_count', 0) >= 3:
        resources['ems'] = True
        resources['reasons'].append("EMS: Multi-vehicle crash")
    
    # ===== FIRE =====
    # Fire department for fire, smoke, or hazmat
    if hazards.get('fire', False):
        resources['fire'] = True
        resources['reasons'].append("Fire: Active fire")
    
    if hazards.get('smoke', False):
        resources['fire'] = True
        resources['reasons'].append("Fire: Smoke visible")
    
    # ===== TOW =====
    # Tow truck for damaged vehicles
    involved = vehicles.get('involved_in_incident', [])
    for vehicle in involved:
        if vehicle.get('damaged', False):
            resources['tow'] = True
            resources['reasons'].append("Tow: Damaged vehicle")
            break
    
    # Also tow for any incident with vehicles (unless it's just debris)
    if incident_type in ['crash', 'breakdown'] and vehicles.get('total_count', 0) > 0:
        resources['tow'] = True
        if "Tow: Damaged vehicle" not in resources['reasons']:
            resources['reasons'].append("Tow: Vehicles involved")
    
    # ===== DOT (Department of Transportation) =====
    # DOT for lane closures, debris, or traffic control
    if lanes.get('blocked', []):
        resources['dot'] = True
        resources['reasons'].append("DOT: Lane closure needed")
    
    if hazards.get('debris', False):
        resources['dot'] = True
        resources['reasons'].append("DOT: Debris removal")
    
    # ===== HAZMAT =====
    # Hazmat for tankers, spills, or hazardous materials
    all_vehicles = vehicles.get('nearby_traffic', []) + vehicles.get('involved_in_incident', [])
    for vehicle in all_vehicles:
        if vehicle.get('type') in ['tanker', 'hazmat']:
            resources['hazmat'] = True
            resources['reasons'].append("HAZMAT: Tanker vehicle involved")
            break
    
    if hazards.get('spill', False):
        resources['hazmat'] = True
        resources['reasons'].append("HAZMAT: Spill visible")
    
    return resources


def test_resources():
    """Test the resource recommender"""
    
    # Test 1: Multi-vehicle crash
    crash_data = {
        'incident_type': 'crash',
        'vehicles': {
            'total_count': 4,
            'involved_in_incident': [
                {'type': 'pickup_truck', 'damaged': True}
            ],
            'nearby_traffic': []
        },
        'hazards': {'fire': False, 'smoke': False, 'debris': True, 'injuries_visible': False},
        'lanes': {'blocked': [1]},
        'emergency_response': {'police_present': False}
    }
    
    print("="*60)
    print("TEST 1: Multi-vehicle crash")
    print("="*60)
    result = recommend_resources(crash_data)
    for resource, value in result.items():
        if resource != 'reasons':
            print(f"  {resource}: {value}")
    print("\n  Reasons:")
    for reason in result['reasons']:
        print(f"    - {reason}")
    
    # Test 2: Fire with tanker
    fire_data = {
        'incident_type': 'fire',
        'vehicles': {
            'total_count': 1,
            'involved_in_incident': [
                {'type': 'tanker', 'damaged': True}
            ],
            'nearby_traffic': []
        },
        'hazards': {'fire': True, 'smoke': True, 'debris': False, 'injuries_visible': True, 'spill': True},
        'lanes': {'blocked': []},
        'emergency_response': {'police_present': False}
    }
    
    print("\n" + "="*60)
    print("TEST 2: Fire with tanker")
    print("="*60)
    result = recommend_resources(fire_data)
    for resource, value in result.items():
        if resource != 'reasons':
            print(f"  {resource}: {value}")
    print("\n  Reasons:")
    for reason in result['reasons']:
        print(f"    - {reason}")
    
    # Test 3: Debris only
    debris_data = {
        'incident_type': 'debris',
        'vehicles': {
            'total_count': 0,
            'involved_in_incident': [],
            'nearby_traffic': []
        },
        'hazards': {'fire': False, 'smoke': False, 'debris': True, 'injuries_visible': False},
        'lanes': {'blocked': [2]},
        'emergency_response': {'police_present': False}
    }
    
    print("\n" + "="*60)
    print("TEST 3: Debris in roadway")
    print("="*60)
    result = recommend_resources(debris_data)
    for resource, value in result.items():
        if resource != 'reasons':
            print(f"  {resource}: {value}")
    print("\n  Reasons:")
    for reason in result['reasons']:
        print(f"    - {reason}")

if __name__ == "__main__":
    test_resources()