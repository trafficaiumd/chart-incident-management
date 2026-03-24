# tests/model_stubs.py
"""
Simple model stubs for testing the evaluation framework.
Replace with real API calls later.
"""

import random
import time
from typing import Dict, Any

def _base_empty() -> Dict[str, Any]:
    return {
        "incident_type": "unknown",
        "confidence": 0.0,
        "vehicles": {
            "total_count": 0,
            "involved_in_incident": [],
            "nearby_traffic": [],
            "types": [],
        },
        "lanes": {
            "total": 4,
            "blocked": [],
            "shoulder_blocked": False,
        },
        "hazards": {
            "fire": False,
            "smoke": False,
            "debris": False,
            "injuries_visible": False,
            "spill": False,
        },
        "traffic": {
            "state": "unknown",
            "queue_length_meters": 0,
        },
        "emergency_response": {
            "police_present": False,
            "officers_count": 0,
            "ambulance_present": False,
            "fire_truck_present": False,
            "responders_on_scene": False,
        },
        "analysis_failed": False,
    }

def perfect_model(image_path:str) -> Dict[str, Any]:
    """
    Ideal model stub: returns hard‑coded outputs based on filename.

    This is for exercising the evaluation harness, not for production use.
    """
    image_path_lower = image_path.lower()
    base = _base_empty()

    if "79" in image_path_lower:
        # Match the Image_79 crash semantics from tests/test_dataset.py (roughly)
        base.update(
            {
                "incident_type": "crash",
                "confidence": 0.99,
            }
        )
        base["vehicles"] = {
            "total_count": 5,
            "involved_in_incident": [
                {
                    "type": "pickup_truck",
                    "color": "blue",
                    "damaged": True,
                    "position": "crash_location",
                }
            ],
            "nearby_traffic": [
                {
                    "type": "bus",
                    "color": "blue_white",
                    "damaged": False,
                    "position": "right_side",
                },
                {
                    "type": "pickup",
                    "color": "red",
                    "damaged": False,
                    "position": "left_side",
                },
                {
                    "type": "bus",
                    "color": "yellow",
                    "damaged": False,
                    "position": "oncoming",
                },
            ],
            "types": ["pickup_truck", "truck", "bus", "pickup", "bus"],
        }
        base["lanes"] = {
            "total": 4,
            "blocked": [1],
            "shoulder_blocked": False,
        }
        base["hazards"] = {
            "fire": False,
            "smoke": False,
            "debris": True,
            "injuries_visible": False,
            "spill": False,
        }
        base["traffic"] = {
            "state": "slowing",
            "queue_length_meters": 100,
        }
        base["emergency_response"] = {
            "police_present": True,
            "officers_count": 3,
            "ambulance_present": False,
            "fire_truck_present": False,
            "responders_on_scene": True,
        }
        return base

    if "car" in image_path_lower:
        # Match the simple non‑incident car.jpg case
        base.update(
            {
                "incident_type": "none",
                "confidence": 0.99,
            }
        )
        base["vehicles"] = {
            "total_count": 1,
            "involved_in_incident": [],
            "nearby_traffic": [
                {
                    "type": "car",
                    "color": "blue",
                    "damaged": False,
                    "position": "driving",
                }
            ],
            "types": ["car"],
        }
        base["lanes"] = {
            "total": 4,
            "blocked": [],
            "shoulder_blocked": False,
        }
        base["hazards"] = {
            "fire": False,
            "smoke": False,
            "debris": False,
            "injuries_visible": False,
            "spill": False,
        }
        base["traffic"] = {
            "state": "flowing",
            "queue_length_meters": 0,
        }
        base["emergency_response"] = {
            "police_present": False,
            "officers_count": 0,
            "ambulance_present": False,
            "fire_truck_present": False,
            "responders_on_scene": False,
        }
        return base

    # Fallback: no incident / unknown
    base["incident_type"] = "none"
    base["confidence"] = 0.5
    return base


def noisy_model(image_path: str) -> Dict[str, Any]:
    """
    Model stub that introduces random noise on top of perfect_model,
    to exercise error handling and metrics.
    """
    output = perfect_model(image_path)
    time.sleep(0.3)  # Simulate API latency

    # Randomly perturb some fields
    if random.random() < 0.3:
        output["incident_type"] = random.choice(["crash", "debris", "none", "unknown"])

    if random.random() < 0.2:
        # Perturb vehicle count a little, but keep it non‑negative
        c = max(0, output["vehicles"]["total_count"] + random.choice([-1, 1]))
        output["vehicles"]["total_count"] = c

    return output