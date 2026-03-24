# tests/ground_truth.py
"""
Manual ground truth for all test images.
Run this once to create your test cases.
"""

from test_dataset import TestDataset, TEST_IMAGES_DIR
import csv
from pathlib import Path

def create_all_test_cases():
    dataset = TestDataset()
    
    # Manual ground truth based on your observations
    ground_truth = [
        {
            "image": "Image_79.jpg",
            "expected": {
                "incident_type": "crash",
                "vehicles": {"count": 5, "types": ["pickup", "truck", "bus", "pickup", "bus"]},
                "lanes": {"total": 4, "blocked": [1], "shoulder_blocked": False},
                "hazards": {"fire": False, "smoke": False, "debris": True, "injuries_visible": False},
                "traffic": {"state": "slowing", "queue_length_meters": 100}
            },
            "description": "Blue pickup crashed into red truck on bridge",
            "critical_rules": ["3+ vehicles = RED"]
        },
        {
            "image": "car.jpg",
            "expected": {
                "incident_type": "none",
                "vehicles": {"count": 1, "types": ["car"]},
                "lanes": {"total": 4, "blocked": [], "shoulder_blocked": False},
                "hazards": {"fire": False, "smoke": False, "debris": False, "injuries_visible": False},
                "traffic": {"state": "flowing", "queue_length_meters": 0}
            },
            "description": "Blue Camry driving normally",
            "critical_rules": ["false_positive"]
        },
        # Add ALL other images here
        # Image_4.jpg - what's in it?
        # Image_21.jpg - what's in it?
        # etc...
    ]
    
    for item in ground_truth:
        dataset.add_case(
            image_path=str(TEST_IMAGES_DIR / item["image"]),
            expected=item["expected"],
            description=item["description"],
            critical_rules=item.get("critical_rules", [])
        )
    
    return dataset

if __name__ == "__main__":
    dataset = create_all_test_cases()
    dataset.save_to_json("tests/test_cases.json")
    print(f"Created {len(dataset.cases)} test cases")