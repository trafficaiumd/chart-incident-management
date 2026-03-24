"""
Fire dataset loader for test cases.
"""

import os
import json
from typing import List, Dict, Any, Optional
from test_dataset import TestCase


def load_fire_dataset(
    dataset_path: str,
    image_dir: str,
    max_samples: Optional[int] = None
) -> List[TestCase]:
    """
    Load fire incident dataset and convert to test cases.
    
    Args:
        dataset_path: Path to the dataset JSON file
        image_dir: Directory containing the images
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        List of TestCase objects
    """
    with open(dataset_path, 'r') as f:
        data = json.load(f)
    
    test_cases = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break
            
        image_path = os.path.join(image_dir, item.get('image', ''))
        
        # Convert dataset format to unified schema
        expected = {
            "incident_type": item.get('incident_type', 'unknown'),
            "vehicles": {
                "total_count": item.get('vehicle_count', 0),
                "involved_in_incident": item.get('involved_vehicles', []),
                "nearby_traffic": item.get('nearby_vehicles', [])
            },
            "lanes": {
                "blocked": item.get('blocked_lanes', []),
                "shoulder_blocked": item.get('shoulder_blocked', False)
            },
            "hazards": {
                "fire": item.get('fire_present', False),
                "smoke": item.get('smoke_present', False),
                "debris": item.get('debris_present', False),
                "injuries_visible": item.get('injuries_visible', False),
                "spill": item.get('spill_present', False)
            },
            "traffic": {
                "state": item.get('traffic_state', 'unknown')
            },
            "emergency_response": {
                "police_present": item.get('police_present', False),
                "ambulance_present": item.get('ambulance_present', False),
                "fire_truck_present": item.get('fire_truck_present', False),
                "responders_on_scene": item.get('responders_count', 0) > 0,
                "officers_count": item.get('officers_count', 0)
            },
            "pedestrians": {
                "count": item.get('pedestrian_count', 0)
            }
        }
        
        test_cases.append(
            TestCase(
                image_path=image_path,
                expected=expected,
                description=f"Fire incident {i+1}: {item.get('description', '')}"
            )
        )
    
    return test_cases