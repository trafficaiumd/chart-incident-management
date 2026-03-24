# tests/test_dataset.py
import json
from dataclasses import dataclass
from typing import List, Dict, Any
from pathlib import Path


# Get the actual paths
PROJECT_ROOT = Path("/home/group1/chart-incident-management")
REPO_ROOT = PROJECT_ROOT / "chart-incident-management"
TEST_IMAGES_DIR = PROJECT_ROOT / "test_images"

@dataclass
class TestCase:
    image_path: str
    expected: Dict[str, Any]
    description: str
    critical_rules: List[str]

class TestDataset:
    def __init__(self):
        self.cases = []
        
    def add_case(self, image_path, expected, description, critical_rules=None):
        # Handle both full paths and filenames
        if not Path(image_path).exists():
            # Try looking in test_images directory
            full_path = TEST_IMAGES_DIR / Path(image_path).name
            if full_path.exists():
                image_path = str(full_path)
            else:
                print(f"Warning: Image not found: {image_path}")
        
        self.cases.append(TestCase(
            image_path=image_path,
            expected=expected,
            description=description,
            critical_rules=critical_rules or []
        ))
    
    def load_from_json(self, json_path):
        json_path = REPO_ROOT / json_path
        with open(json_path, 'r') as f:
            data = json.load(f)
            for case in data['cases']:
                self.add_case(
                    image_path=case['image'],
                    expected=case['expected'],
                    description=case['description'],
                    critical_rules=case.get('critical_rules', [])
                )
    
    def save_to_json(self, json_path):
        json_path = REPO_ROOT / json_path
        data = {
            'cases': [
                {
                    'image': c.image_path,
                    'expected': c.expected,
                    'description': c.description,
                    'critical_rules': c.critical_rules
                }
                for c in self.cases
            ]
        }
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)

def create_initial_test_set():
    dataset = TestDataset()
    
    # Image_79.jpg - Multi-vehicle crash with emergency response
    # Image_79.jpg - Crash with specific vehicle roles
    dataset.add_case(
        image_path=str(TEST_IMAGES_DIR / "Image_79.jpg"),
        expected={
            "incident_type": "crash",
            "vehicles": {
                "total_count": 4,
                "involved_in_incident": [
                    {
                        "type": "pickup_truck",  # Blue pickup on red truck
                        "color": "blue",
                        "damaged": True,
                        "position": "crash_location"
                    }
                ],
                "nearby_traffic": [
                    {
                        "type": "bus",
                        "color": "blue_white",
                        "damaged": False,
                        "position": "right_side"
                    },
                    {
                        "type": "pickup",
                        "color": "red",
                        "damaged": False,
                        "position": "left_side"
                    },
                    {
                        "type": "bus",
                        "color": "yellow",
                        "damaged": False,
                        "position": "oncoming"
                    }
                ],
                "emergency_vehicles": [],  # No emergency vehicles, just officers
                "pedestrians": {
                    "count": 5,
                    "positions": ["right_side"],  # 3 officers + 1 man
                    "on_roadway": False
                }
            },
            "lanes": {
                "total": 4,
                "blocked": [1],  # Leftmost lane where crash occurred
                "shoulder_blocked": False,
                "affected_lanes": [1]  # Same as blocked
            },
            "hazards": {
                "fire": False,
                "smoke": False,
                "debris": True,
                "injuries_visible": False,
                "spill": False
            },
            "traffic": {
                "state": "slowing",  # Traffic still moving but slowed
                "queue_length_meters": 100,
                "lanes_affected": [1]  # Only left lane affected
            },
            "emergency_response": {
                "police_present": True,
                "officers_count": 3,
                "ambulance_present": False,
                "fire_truck_present": False,
                "responders_on_scene": True
            }
            },

        description="Blue pickup crashed on red truck, blocking left lane. Police on scene. Other traffic passing.",
        critical_rules=[
            "3+ vehicles total = RED",  # 5 total vehicles
            "debris_present = YELLOW",  # At least caution
            "police_on_scene = dispatch_confirmed"
        ]
    )
    # car.jpg - Just a car driving (no incident)
    # In test_dataset.py, update the car.jpg test case:

# car.jpg - Just a car driving (no incident)
    dataset.add_case(
        image_path=str(TEST_IMAGES_DIR / "car.jpg"),
        expected={
            "incident_type": "none",
            "vehicles": {
                "total_count": 1,  # 👈 Changed from "count"
                "involved_in_incident": [],  # 👈 Add this
                "nearby_traffic": [  # 👈 Add this
                    {
                        "type": "car",
                        "color": "blue",
                        "damaged": False,
                        "position": "driving"
                    }
                ],
                "emergency_vehicles": [],  # 👈 Add this
                "pedestrians": {  # 👈 Add this
                    "count": 0,
                    "positions": [],
                    "on_roadway": False
                }
            },
            "lanes": {
                "total": 4,
                "blocked": [],
                "shoulder_blocked": False,
                "affected_lanes": []  # 👈 Add this
            },
            "hazards": {
                "fire": False,
                "smoke": False,
                "debris": False,
                "injuries_visible": False,
                "spill": False  # 👈 Add this
            },
            "traffic": {
                "state": "flowing",
                "queue_length_meters": 0,
                "lanes_affected": []  # 👈 Add this
            },
            "emergency_response": {  # 👈 Add this whole section
                "police_present": False,
                "officers_count": 0,
                "ambulance_present": False,
                "fire_truck_present": False,
                "responders_on_scene": False
            }
        },
        description="Blue Camry driving normally",
        critical_rules=["false_positive"]
    )
    
    return dataset


if __name__ == "__main__":
    dataset = create_initial_test_set()
    dataset.save_to_json("tests/test_cases.json")
    print(f"\nCreated {len(dataset.cases)} test cases")
    print(f"Test cases saved to: {REPO_ROOT}/tests/test_cases.json")