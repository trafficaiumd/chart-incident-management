# decision_engine/secondary_risk.py
"""
Secondary Crash Risk Analyzer
Evaluates risk of additional crashes after initial incident
"""

def analyze_secondary_risk(incident_data, time_of_day="day", weather="clear"):
    """
    Calculate risk of secondary crashes
    
    Args:
        incident_data: Dictionary from model output
        time_of_day: "day", "night", "dawn", "dusk"
        weather: "clear", "rain", "snow", "fog", "ice"
    
    Returns:
        dict: {
            'risk_level': 'HIGH'|'MEDIUM'|'LOW',
            'risk_score': int (0-100),
            'factors': [str],
            'recommendations': [str]
        }
    """
    risk_score = 0
    factors = []
    recommendations = []
    
    # Extract data
    incident_type = incident_data.get('incident_type', 'unknown')
    lanes = incident_data.get('lanes', {})
    traffic = incident_data.get('traffic', {})
    hazards = incident_data.get('hazards', {})
    
    # ===== PRIMARY INCIDENT FACTORS =====
    
    # Incident type
    if incident_type == 'crash':
        risk_score += 25
        factors.append("Primary crash present")
    elif incident_type == 'fire':
        risk_score += 30
        factors.append("Fire reduces visibility")
    
    # Lane blockage
    blocked_count = len(lanes.get('blocked', []))
    if blocked_count >= 2:
        risk_score += 30
        factors.append(f"{blocked_count} lanes blocked")
    elif blocked_count == 1:
        risk_score += 15
        factors.append("Single lane blocked")
    
    # Shoulder blockage
    if lanes.get('shoulder_blocked', False):
        risk_score += 10
        factors.append("Shoulder blocked - no refuge")
    
    # ===== TRAFFIC CONDITIONS =====
    
    traffic_state = traffic.get('state', 'unknown')
    if traffic_state == 'stopped':
        risk_score += 35
        factors.append("Stopped traffic ahead")
        recommendations.append("Advise variable speed limits")
    elif traffic_state == 'slowing':
        risk_score += 20
        factors.append("Traffic slowing suddenly")
        recommendations.append("Display queue warning")
    
    queue_length = traffic.get('queue_length_meters', 0)
    if queue_length > 500:
        risk_score += 15
        factors.append(f"Long queue ({queue_length}m)")
        recommendations.append("Consider ramp metering")
    
    # ===== ENVIRONMENTAL FACTORS =====
    
    # Time of day
    if time_of_day in ['night', 'dusk', 'dawn']:
        risk_score += 25
        factors.append(f"Low light ({time_of_day})")
        recommendations.append("Ensure lighting activated")
    
    # Weather
    weather_risk = {
        'clear': 0,
        'rain': 20,
        'snow': 30,
        'fog': 35,
        'ice': 40
    }
    risk_score += weather_risk.get(weather, 0)
    if weather != 'clear':
        factors.append(f"{weather} conditions")
        recommendations.append(f"Alert drivers to {weather}")
    
    # ===== HAZARDS =====
    
    if hazards.get('smoke', False):
        risk_score += 25
        factors.append("Smoke reduces visibility")
        recommendations.append("Post reduced speed")
    
    if hazards.get('debris', False):
        risk_score += 15
        factors.append("Debris in roadway")
        recommendations.append("Send maintenance crew")
    
    if hazards.get('spill', False):
        risk_score += 30
        factors.append("Hazmat spill")
        recommendations.append("Close affected lanes")
    
    # Cap risk score at 100
    risk_score = min(risk_score, 100)
    # ===== DETERMINE RISK LEVEL =====
    
    if risk_score >= 70:
        risk_level = 'HIGH'
        recommendations.append("ACTIVATE SECONDARY ALERT")
    elif risk_score >= 40:
        risk_level = 'MEDIUM'
        recommendations.append("Monitor closely")
    else:
        risk_level = 'LOW'
        recommendations.append("Routine monitoring")
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'factors': factors,
        'recommendations': list(set(recommendations))  # Remove duplicates
    }


def test_risk_analyzer():
    """Test the secondary risk analyzer"""
    
    test_scenarios = [
        {
            'name': 'Night crash, 2 lanes blocked',
            'data': {
                'incident_type': 'crash',
                'lanes': {'blocked': [1, 2], 'shoulder_blocked': False},
                'traffic': {'state': 'stopped', 'queue_length_meters': 800},
                'hazards': {'smoke': False, 'debris': True, 'spill': False}
            },
            'time': 'night',
            'weather': 'clear'
        },
        {
            'name': 'Day debris, light rain',
            'data': {
                'incident_type': 'debris',
                'lanes': {'blocked': [3], 'shoulder_blocked': False},
                'traffic': {'state': 'slowing', 'queue_length_meters': 200},
                'hazards': {'smoke': False, 'debris': True, 'spill': False}
            },
            'time': 'day',
            'weather': 'rain'
        },
        {
            'name': 'Fog, shoulder fire',
            'data': {
                'incident_type': 'fire',
                'lanes': {'blocked': [], 'shoulder_blocked': True},
                'traffic': {'state': 'slowing', 'queue_length_meters': 300},
                'hazards': {'smoke': True, 'debris': False, 'spill': False}
            },
            'time': 'dawn',
            'weather': 'fog'
        },
        {
            'name': 'Minor breakdown, clear conditions',
            'data': {
                'incident_type': 'breakdown',
                'lanes': {'blocked': [], 'shoulder_blocked': True},
                'traffic': {'state': 'flowing', 'queue_length_meters': 0},
                'hazards': {'smoke': False, 'debris': False, 'spill': False}
            },
            'time': 'day',
            'weather': 'clear'
        }
    ]
    
    print("="*70)
    print("SECONDARY CRASH RISK ANALYSIS")
    print("="*70)
    
    for scenario in test_scenarios:
        print(f"\n📋 {scenario['name']}")
        print("-" * 50)
        
        result = analyze_secondary_risk(
            scenario['data'], 
            scenario['time'], 
            scenario['weather']
        )
        
        # Color code risk level
        level = result['risk_level']
        if level == 'HIGH':
            level_display = "🔴 HIGH"
        elif level == 'MEDIUM':
            level_display = "🟡 MEDIUM"
        else:
            level_display = "🟢 LOW"
        
        print(f"  Risk Level: {level_display}")
        print(f"  Risk Score: {result['risk_score']}/100")
        
        print("\n  Risk Factors:")
        for factor in result['factors']:
            print(f"    • {factor}")
        
        print("\n  Recommendations:")
        for rec in result['recommendations']:
            print(f"    • {rec}")

if __name__ == "__main__":
    test_risk_analyzer()