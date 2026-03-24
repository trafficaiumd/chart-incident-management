#tests/coco_loader.py
"""
Generic COCO dataset loader utilities.
"""

import json
from pathlib import Path
from typing import Dict, Any, Callable

from test_dataset import TestDataset


def load_coco_dataset(
    coco_path: Path,
    image_root: Path,
    label_to_expected: Callable[[Dict[str, Any]], Dict[str, Any]],
    description_prefix: str,
) -> TestDataset:
    """
    Generic COCO -> TestDataset loader.

    - coco_path: path to _annotations.coco.json
    - image_root: folder containing the images referenced by 'file_name'
    - label_to_expected: function mapping a COCO sample
      { "image": <image_obj>, "annotations": [...], "coco": <full_coco> }
      into your expected incident dict.
    """
    with open(coco_path, "r") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco.get("images", [])}
    anns_by_image: Dict[int, list[Dict[str, Any]]] = {}
    for ann in coco.get("annotations", []):
        anns_by_image.setdefault(ann["image_id"], []).append(ann)

    dataset = TestDataset()

    for image_id, img in images.items():
        file_name = img["file_name"]
        img_path = image_root / file_name
        if not img_path.exists():
            print(f"Warning: image file missing for {img_path}")
            continue

        anns = anns_by_image.get(image_id, [])
        sample = {"image": img, "annotations": anns, "coco": coco}
        expected = label_to_expected(sample)

        dataset.add_case(
            image_path=str(img_path),
            expected=expected,
            description=f"{description_prefix} {file_name}",
            critical_rules=[],
        )

    return dataset