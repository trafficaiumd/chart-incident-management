#tests/dataset_loaders.py
"""
Dataset loaders for all COCO-based incident datasets.

Datasets (under tests/datasets/):
- Accident and Non-accident label Image Dataset
- Cars on fire.
- School  bus.
- Trucks.v8-training_3.
- motorcycle.
- Road Lane detection
- tanker.   (train only)
"""

from pathlib import Path
from typing import Dict, Any

from test_dataset import TestDataset
from coco_loader import load_coco_dataset


DATASETS_ROOT = Path(__file__).parent / "datasets"


# --------------------------------------------------------------------- #
# Helper: common "empty expected" template
# --------------------------------------------------------------------- #

def _empty_expected(incident_type: str = "none") -> Dict[str, Any]:
    return {
        "incident_type": incident_type,
        "vehicles": {
            "total_count": 0,
            "involved_in_incident": [],
            "nearby_traffic": [],
            "types": [],
        },
        "lanes": {
            "total": 0,
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
        "pedestrians": {
            "count": 0,
        },
    }


# --------------------------------------------------------------------- #
# Accident vs Non-accident
# --------------------------------------------------------------------- #

ACCIDENT_ROOT = DATASETS_ROOT / "Accident and Non-accident label Image Dataset"


def _accident_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    has_accident = False
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        # Adjust this substring to the actual category name in your COCO file
        if "accident" in name:
            has_accident = True
            break

    inc_type = "crash" if has_accident else "none"
    expected = _empty_expected(inc_type)
    return expected


def load_accident_dataset(split: str = "test") -> TestDataset:
    coco_path = ACCIDENT_ROOT / split / "_annotations.coco.json"
    image_root = ACCIDENT_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_accident_label_to_expected,
        description_prefix=f"Accident {split}",
    )


# --------------------------------------------------------------------- #
# Cars on fire
# --------------------------------------------------------------------- #

CARS_FIRE_ROOT = DATASETS_ROOT / "Cars on fire."


def _cars_fire_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    has_fire = False
    has_smoke = False
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        # Adjust these substrings to your actual category names
        if "fire" in name:
            has_fire = True
        if "smoke" in name:
            has_smoke = True

    expected = _empty_expected("fire" if has_fire else "none")
    expected["hazards"]["fire"] = has_fire
    expected["hazards"]["smoke"] = has_smoke
    return expected


def load_cars_on_fire_dataset(split: str = "test") -> TestDataset:
    coco_path = CARS_FIRE_ROOT / split / "_annotations.coco.json"
    image_root = CARS_FIRE_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_cars_fire_label_to_expected,
        description_prefix=f"CarsFire {split}",
    )


# --------------------------------------------------------------------- #
# School bus
# --------------------------------------------------------------------- #

SCHOOL_BUS_ROOT = DATASETS_ROOT / "School  bus."


def _school_bus_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    bus_count = 0
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        # Adjust: category likely like "school bus" or similar
        if "bus" in name:
            bus_count += 1

    expected = _empty_expected("none")
    expected["vehicles"]["total_count"] = bus_count
    expected["vehicles"]["types"] = ["school_bus"] * bus_count if bus_count else []
    return expected


def load_school_bus_dataset(split: str = "test") -> TestDataset:
    coco_path = SCHOOL_BUS_ROOT / split / "_annotations.coco.json"
    image_root = SCHOOL_BUS_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_school_bus_label_to_expected,
        description_prefix=f"SchoolBus {split}",
    )


# --------------------------------------------------------------------- #
# Trucks
# --------------------------------------------------------------------- #

TRUCKS_ROOT = DATASETS_ROOT / "Trucks.v8-training_3."


def _trucks_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    truck_count = 0
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        if "truck" in name:
            truck_count += 1

    expected = _empty_expected("none")
    expected["vehicles"]["total_count"] = truck_count
    expected["vehicles"]["types"] = ["truck"] * truck_count if truck_count else []
    return expected


def load_trucks_dataset(split: str = "test") -> TestDataset:
    coco_path = TRUCKS_ROOT / split / "_annotations.coco.json"
    image_root = TRUCKS_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_trucks_label_to_expected,
        description_prefix=f"Trucks {split}",
    )


# --------------------------------------------------------------------- #
# Motorcycle
# --------------------------------------------------------------------- #

MOTORCYCLE_ROOT = DATASETS_ROOT / "motorcycle."


def _motorcycle_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    moto_count = 0
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        if "motorcycle" in name or "bike" in name:
            moto_count += 1

    expected = _empty_expected("none")
    expected["vehicles"]["total_count"] = moto_count
    expected["vehicles"]["types"] = ["motorcycle"] * moto_count if moto_count else []
    return expected


def load_motorcycle_dataset(split: str = "test") -> TestDataset:
    coco_path = MOTORCYCLE_ROOT / split / "_annotations.coco.json"
    image_root = MOTORCYCLE_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_motorcycle_label_to_expected,
        description_prefix=f"Motorcycle {split}",
    )


# --------------------------------------------------------------------- #
# Road Lane detection
# --------------------------------------------------------------------- #

LANE_ROOT = DATASETS_ROOT / "Road Lane detection"


def _lane_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    # This dataset is lane geometry only; no incidents.
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    lane_like = 0
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        if "lane" in name:
            lane_like += 1

    expected = _empty_expected("none")
    # Crude approximation: number of lane objects == total lanes
    expected["lanes"]["total"] = lane_like or 3
    expected["lanes"]["blocked"] = []
    expected["lanes"]["shoulder_blocked"] = False
    return expected


def load_road_lane_dataset(split: str = "test") -> TestDataset:
    coco_path = LANE_ROOT / split / "_annotations.coco.json"
    image_root = LANE_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_lane_label_to_expected,
        description_prefix=f"Lane {split}",
    )


# --------------------------------------------------------------------- #
# Tanker (train only)
# --------------------------------------------------------------------- #

TANKER_ROOT = DATASETS_ROOT / "tanker."


def _tanker_label_to_expected(sample: Dict[str, Any]) -> Dict[str, Any]:
    anns = sample["annotations"]
    coco = sample["coco"]
    categories = {c["id"]: c["name"].lower() for c in coco.get("categories", [])}

    tanker_count = 0
    for ann in anns:
        name = categories.get(ann["category_id"], "")
        if "tanker" in name or "tank" in name:
            tanker_count += 1

    expected = _empty_expected("none")
    expected["vehicles"]["total_count"] = tanker_count
    expected["vehicles"]["types"] = ["tanker"] * tanker_count if tanker_count else []
    return expected


def load_tanker_dataset(split: str = "train") -> TestDataset:
    """
    Tanker dataset only has 'train' folder.
    Other split values will be forced to 'train'.
    """
    split = "train"
    coco_path = TANKER_ROOT / split / "_annotations.coco.json"
    image_root = TANKER_ROOT / split
    return load_coco_dataset(
        coco_path=coco_path,
        image_root=image_root,
        label_to_expected=_tanker_label_to_expected,
        description_prefix=f"Tanker {split}",
    )


# --------------------------------------------------------------------- #
# Unified loader
# --------------------------------------------------------------------- #

def load_all_datasets(
    accident_split: str = "test",
    fire_split: str = "test",
    school_bus_split: str = "test",
    trucks_split: str = "test",
    motorcycle_split: str = "test",
    lane_split: str = "test",
    tanker_split: str = "train",
) -> Dict[str, TestDataset]:
    """
    Load all datasets into a dict keyed by dataset name.
    """
    datasets: Dict[str, TestDataset] = {}

    datasets["accident"] = load_accident_dataset(accident_split)
    datasets["cars_on_fire"] = load_cars_on_fire_dataset(fire_split)
    datasets["school_bus"] = load_school_bus_dataset(school_bus_split)
    datasets["trucks"] = load_trucks_dataset(trucks_split)
    datasets["motorcycle"] = load_motorcycle_dataset(motorcycle_split)
    datasets["road_lane"] = load_road_lane_dataset(lane_split)
    datasets["tanker"] = load_tanker_dataset(tanker_split)  # always train

    return datasets